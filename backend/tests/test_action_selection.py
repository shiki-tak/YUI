"""目標の参照と行動選択（フェーズ4 PR7）。

設計書 6 の 3「回答・確認質問・話題提案・調査・待機から適切な行動を選ぶ」。

ここで固定するのは次の4つ。

- 実行できる目標だけが会話へ渡る（期限前・要確認・別の相手のものは渡らない）。
- 選んだ行動を記録する。**待機を選んだことも残す。**
- 話しかけないことがある。「常に話しかけることを自律性の達成条件にしない」。
- 調査は保留する。フェーズ5A が未実装なので「調べた」と言わせない。
"""

from __future__ import annotations

from datetime import timedelta

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agent.goal import create_goal
from app.models import Action, GoalStatus, GoalTrigger, utcnow
from tests.conftest import FakeLLM


async def _speaker(client: AsyncClient, external_id: str | None = None) -> int:
    """相手を1人作る。既定の相手（以降の /api/chat と同じ）で作る。

    別の相手を作るときだけ external_id を渡す。目標は「誰に対するものか」を
    持つので、ここがずれると渡らない。
    """
    body: dict = {"text": "こんにちは"}
    if external_id is not None:
        body["speaker"] = {
            "source": "local_text",
            "external_id": external_id,
            "display_name": external_id,
        }
    response = await client.post("/api/chat", json=body)
    return response.json()["user_message"]["speaker_id"]


async def _goal(session_factory, **kwargs):
    async with session_factory() as session:
        goal = await create_goal(session, status=GoalStatus.ACTIVE.value, **kwargs)
        await session.commit()
        return goal.id


async def test_an_accepted_goal_reaches_the_conversation(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """採用した目標が会話へ渡り、実行記録に残る（完了条件1・5）。"""
    speaker_id = await _speaker(client)
    goal_id = await _goal(
        session_factory,
        content="土曜に見た映画の感想を聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )

    fake_llm.push_action('{"action": "ask", "goal": 1, "reason": "話の流れに合う"}')
    fake_llm.push("そういえば、土曜の映画はどうでしたか？")
    reply = await client.post("/api/chat", json={"text": "ただいま"})

    assert reply.json()["run"]["referenced_goal_ids"] == [goal_id]
    assert reply.json()["run"]["selected_action"] == Action.ASK.value
    # 聞くと決めたときは、聞くよう指示して渡す。
    assert "# 次に話したいこと" in fake_llm.last_system_prompt
    assert "1つだけ" in fake_llm.last_system_prompt


async def test_waiting_is_recorded_and_the_goal_is_not_raised(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """待機を選んだことを記録する（完了条件2）。

    目標は渡す。渡さないことで待機させると、判断ではなく取り上げているだけに
    なる。渡したうえで「いまは持ち出さない」と指示する。
    """
    speaker_id = await _speaker(client)
    goal_id = await _goal(
        session_factory,
        content="土曜に見た映画の感想を聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )

    fake_llm.push_action('{"action": "wait", "goal": null, "reason": "疲れている"}')
    fake_llm.push("お疲れさまでした。ゆっくり休んでくださいね。")
    reply = await client.post("/api/chat", json={"text": "今日は残業で疲れた。もう寝る"})

    run = reply.json()["run"]
    assert run["selected_action"] == Action.WAIT.value
    # 渡してはいる。渡ったことも記録する。
    assert run["referenced_goal_ids"] == [goal_id]
    assert "いまは持ち出さないでください" in fake_llm.last_system_prompt


async def test_a_goal_that_is_not_executable_never_reaches_the_conversation(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """期限前・要確認の目標は渡さない。行動選択にも掛けない。"""
    speaker_id = await _speaker(client)
    await _goal(
        session_factory,
        content="来週の面接の結果を聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
        trigger=GoalTrigger.AFTER_DATE.value,
        due_at=utcnow() + timedelta(days=7),
    )

    reply = await client.post("/api/chat", json={"text": "こんばんは"})
    run = reply.json()["run"]
    assert run["referenced_goal_ids"] is None
    # 目標が無いので、行動選択でモデルを呼ばない（設計書「常時LLMを呼び続けず」）。
    assert run["selected_action"] == Action.ANSWER.value
    assert "# 次に話したいこと" not in fake_llm.last_system_prompt


async def test_research_is_held_and_not_claimed(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """調査は保留する。フェーズ5A が未実装なので「調べた」と言わせない。"""
    speaker_id = await _speaker(client)
    await _goal(
        session_factory,
        content="開発者が気にしていたカメラの新型について調べて伝える",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )

    fake_llm.push_action('{"action": "research", "goal": 1, "reason": "調べないと分からない"}')
    fake_llm.push("ごめんなさい、いまは調べられないんです。")
    reply = await client.post("/api/chat", json={"text": "あのカメラ、新型出た？"})

    assert reply.json()["run"]["selected_action"] == Action.RESEARCH.value
    prompt = fake_llm.last_system_prompt
    assert "いまは調べられません" in prompt
    assert "調べたふりをせず" in prompt


async def test_yui_can_start_a_conversation(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """YUI の側から会話を始められる（完了条件1）。"""
    speaker_id = await _speaker(client)
    goal_id = await _goal(
        session_factory,
        content="土曜に見た映画の感想を聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )
    conversation_id = (await client.get("/api/conversations")).json()[0]["id"]

    fake_llm.push_action('{"action": "ask", "goal": 1, "reason": "期限を過ぎている"}')
    fake_llm.push("こんばんは。土曜の映画、どうでしたか？")
    opened = await client.post(f"/api/conversations/{conversation_id}/open")

    body = opened.json()
    assert body["action"] == Action.ASK.value
    assert body["referenced_goal_ids"] == [goal_id]
    assert body["message"]["content"] == "こんばんは。土曜の映画、どうでしたか？"
    assert body["message"]["speaker_kind"] == "character"


async def test_yui_can_decide_not_to_start(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """話しかけないこともある（完了条件2）。

    設計書「常に話しかけることを自律性の達成条件にはしません」。待機を選んだ
    ときは発言を作らない。**発言が無いことと、失敗は別**である。
    """
    speaker_id = await _speaker(client)
    await _goal(
        session_factory,
        content="土曜に見た映画の感想を聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )
    conversation_id = (await client.get("/api/conversations")).json()[0]["id"]

    fake_llm.push_action('{"action": "wait", "goal": null, "reason": "前回の話の流れで間が悪い"}')
    opened = await client.post(f"/api/conversations/{conversation_id}/open")

    body = opened.json()
    assert opened.status_code == 200
    assert body["action"] == Action.WAIT.value
    assert body["message"] is None
    assert body["reason"] == "前回の話の流れで間が悪い"


async def test_an_unreadable_choice_falls_back_to_waiting(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """行動を読み取れなければ、触れない側へ倒す。

    判断できなかったときに聞きに行くと、相手の様子を見ずに質問することになる。
    会話そのものは止めない。
    """
    speaker_id = await _speaker(client)
    await _goal(
        session_factory,
        content="土曜に見た映画の感想を聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )

    fake_llm.push_action("行動を選べませんでした")
    fake_llm.push("こんばんは。")
    reply = await client.post("/api/chat", json={"text": "ただいま"})

    assert reply.status_code == 200
    assert reply.json()["run"]["selected_action"] == Action.WAIT.value


async def test_asking_without_naming_a_goal_falls_back_to_waiting(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """どの目標か決められないまま聞かない。

    根拠を追えないまま話しかけることになる。完了条件5（行動の根拠を追える）に
    反する。
    """
    speaker_id = await _speaker(client)
    await _goal(
        session_factory,
        content="土曜に見た映画の感想を聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )

    fake_llm.push_action('{"action": "ask", "goal": null, "reason": "なんとなく"}')
    fake_llm.push("こんばんは。")
    reply = await client.post("/api/chat", json={"text": "ただいま"})

    assert reply.json()["run"]["selected_action"] == Action.WAIT.value


async def test_a_goal_for_someone_else_is_not_selected(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """別の相手の目標は、行動選択にも掛からない（設計書 3C）。"""
    alice = await _speaker(client, "alice")
    bob = await _speaker(client, "bob")
    await _goal(
        session_factory,
        content="アリスに映画の感想を聞く",
        subject_speaker_id=alice,
        visible_to_speaker_id=alice,
    )

    reply = await client.post(
        "/api/chat",
        json={
            "text": "こんばんは",
            "speaker": {
                "source": "local_text",
                "external_id": "bob",
                "display_name": "bob",
            },
        },
    )
    assert bob is not None
    run = reply.json()["run"]
    assert run["referenced_goal_ids"] is None
    assert run["selected_action"] == Action.ANSWER.value


# --- 第1回レビューへの対応 --------------------------------------------------


async def test_the_write_lock_is_released_before_the_first_model_call(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """行動選択の待ち時間に、書き込みロックを持たない（指摘1）。

    SQLite は書き込みロックを1つしか持てない。モデルを待つ間に握ると、別の
    会話や記憶の訂正が「database is locked」で失敗する。**モデルを呼ぶ処理を
    足すときは、その前にトランザクションを閉じているかを必ず確かめる。**
    """
    import asyncio

    speaker_id = await _speaker(client)
    await _goal(
        session_factory,
        content="土曜に見た映画の感想を聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )

    fake_llm.entered.clear()
    fake_llm.gate = asyncio.Event()
    fake_llm.push_action('{"action": "ask", "goal": 1, "reason": "聞ける"}')
    fake_llm.push("土曜の映画、どうでしたか？")

    task = asyncio.create_task(client.post("/api/chat", json={"text": "ただいま"}))
    await asyncio.wait_for(fake_llm.entered.wait(), timeout=5)

    # 行動選択を待っている間に、別の接続から書けること。
    async with session_factory() as session:
        written = await create_goal(
            session, content="別の接続から書いた目標", status=GoalStatus.ACTIVE.value
        )
        await session.commit()
        assert written.id is not None

    fake_llm.gate.set()
    fake_llm.gate = None
    assert (await asyncio.wait_for(task, timeout=10)).status_code == 200


async def test_open_is_rejected_while_reflecting(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """振り返り中の会話へ、自発発話を入れない（指摘2）。

    振り返りの対象を固めた後に発言が入ると、抽出した内容と会話ログが食い違う。
    発言の API と同じ検査を通す。
    """
    from app.models import Conversation

    speaker_id = await _speaker(client)
    await _goal(
        session_factory,
        content="土曜に見た映画の感想を聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )
    conversation_id = (await client.get("/api/conversations")).json()[0]["id"]

    async with session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        conversation.reflection_started_at = utcnow()
        await session.commit()

    refused = await client.post(f"/api/conversations/{conversation_id}/open")
    assert refused.status_code == 409
    assert "振り返り中" in refused.json()["detail"]


async def test_only_the_chosen_goal_is_passed_to_the_generation(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """聞くと決めた目標だけを渡す（指摘3）。

    全部渡して生成側に選び直させると、相手の様子を見て選んだ判断が伝わらない。
    触れないときは、持っているものとして全部渡す。
    """
    speaker_id = await _speaker(client)
    first = await _goal(
        session_factory,
        content="土曜に見た映画の感想を聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )
    second = await _goal(
        session_factory,
        content="買ったカメラを使ったか聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )

    fake_llm.push_action('{"action": "ask", "goal": 2, "reason": "カメラの話が続いている"}')
    fake_llm.push("そのカメラ、もう使いましたか？")
    reply = await client.post("/api/chat", json={"text": "カメラの話の続きなんだけど"})

    prompt = fake_llm.last_system_prompt
    assert "買ったカメラを使ったか聞く" in prompt
    assert "土曜に見た映画の感想を聞く" not in prompt
    # 記録も、実際に渡したものにそろえる。
    assert reply.json()["run"]["referenced_goal_ids"] == [second]
    assert first != second


async def test_a_broken_choice_does_not_raise(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """行動の型が不正でも、例外にせず待機へ倒す（指摘4）。"""
    speaker_id = await _speaker(client)
    await _goal(
        session_factory,
        content="土曜に見た映画の感想を聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )

    for broken in ('{"action": [], "goal": 1}', '{"action": {"a": 1}}', '{"action": 3}'):
        fake_llm.push_action(broken)
        fake_llm.push("こんばんは。")
        reply = await client.post("/api/chat", json={"text": "ただいま"})
        assert reply.status_code == 200
        assert reply.json()["run"]["selected_action"] == Action.WAIT.value


async def test_waiting_leaves_a_record(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """話しかけなかったことを記録する（指摘5）。

    発言が無いので run_records は作れない。設計書は「適切な待機も確認する」と
    しており、残さないと適切に待てたのかを測れない。
    """
    from sqlalchemy import select

    from app.models import ActionRecord

    speaker_id = await _speaker(client)
    goal_id = await _goal(
        session_factory,
        content="土曜に見た映画の感想を聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )
    conversation_id = (await client.get("/api/conversations")).json()[0]["id"]

    fake_llm.push_action('{"action": "wait", "goal": null, "reason": "間が悪い"}')
    opened = await client.post(f"/api/conversations/{conversation_id}/open")
    assert opened.json()["message"] is None

    async with session_factory() as session:
        records = list((await session.execute(select(ActionRecord))).scalars())
    proactive = [r for r in records if r.is_proactive]
    assert len(proactive) == 1
    assert proactive[0].action == Action.WAIT.value
    assert proactive[0].reason == "間が悪い"
    assert proactive[0].candidate_goal_ids == [goal_id]
    # 発言していないので、発言との結び付きは無い。
    assert proactive[0].message_id is None


async def test_open_reports_a_model_failure_as_unavailable(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """モデルに繋がらないことと、正しい待機を区別して返す（指摘7）。"""
    from app.llm.base import LLMError

    speaker_id = await _speaker(client)
    await _goal(
        session_factory,
        content="土曜に見た映画の感想を聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )
    conversation_id = (await client.get("/api/conversations")).json()[0]["id"]

    async def failing_chat(*args, **kwargs):
        raise LLMError("接続できません")

    fake_llm.chat = failing_chat  # type: ignore[method-assign]
    response = await client.post(f"/api/conversations/{conversation_id}/open")
    assert response.status_code == 503
    assert "接続できません" in response.json()["detail"]


async def test_open_releases_the_write_lock_when_it_creates_a_speaker(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """未登録の相手を指定しても、行動選択の待ち時間にロックを持たない（指摘1）。

    話者の作成は INSERT である。閉じないまま行動選択の応答を待つと、通常の
    発言で直したのと同じロックが残る。**入口ごとに確かめる。**
    """
    import asyncio

    await _goal(session_factory, content="相手を選ばない話題を出す")
    conversation_id = (
        await client.post("/api/chat", json={"text": "こんにちは"})
    ).json()["conversation_id"]

    fake_llm.entered.clear()
    fake_llm.gate = asyncio.Event()
    fake_llm.push_action('{"action": "ask", "goal": 1, "reason": "話せる"}')
    fake_llm.push("こんばんは。")

    task = asyncio.create_task(
        client.post(
            f"/api/conversations/{conversation_id}/open",
            params={
                "source": "local_text",
                "external_id": "newcomer",
                "display_name": "はじめての人",
            },
        )
    )
    await asyncio.wait_for(fake_llm.entered.wait(), timeout=5)

    async with session_factory() as session:
        written = await create_goal(
            session, content="別の接続から書いた目標", status=GoalStatus.ACTIVE.value
        )
        await session.commit()
        assert written.id is not None

    fake_llm.gate.set()
    fake_llm.gate = None
    assert (await asyncio.wait_for(task, timeout=10)).status_code == 200


async def test_the_action_record_keeps_the_model_that_decided(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """判断に使ったモデルと設定を残す（指摘2）。

    発言の無い待機は run_records が作られない。残さないと、どのモデル・設定が
    待機を選んだのかを後から追えない。モデルを呼ばずに決めた判断は、モデル情報
    なしとして区別する。
    """
    from sqlalchemy import select

    from app.models import ActionRecord

    speaker_id = await _speaker(client)
    await _goal(
        session_factory,
        content="土曜に見た映画の感想を聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )
    conversation_id = (await client.get("/api/conversations")).json()[0]["id"]

    fake_llm.push_action('{"action": "wait", "goal": null, "reason": "間が悪い"}')
    await client.post(f"/api/conversations/{conversation_id}/open")

    async with session_factory() as session:
        record = (
            await session.execute(select(ActionRecord).where(ActionRecord.is_proactive))
        ).scalar_one()
    assert record.provider == "fake"
    assert record.model == "fake-model"
    assert record.model_digest == "sha256:test"
    assert record.options == {"temperature": 0.2}
    # 正常な判断なので、読み取り失敗の印は付かない。
    assert record.is_fallback is False


async def test_a_fallback_wait_is_marked_apart_from_a_real_one(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """読み取り失敗による待機を、相手を見て待った判断と区別する。

    どちらも action は wait になる。区別できないと、適切に待てた回数を
    数えられない（第2回レビューの補足）。
    """
    from sqlalchemy import select

    from app.models import ActionRecord

    speaker_id = await _speaker(client)
    await _goal(
        session_factory,
        content="土曜に見た映画の感想を聞く",
        subject_speaker_id=speaker_id,
        visible_to_speaker_id=speaker_id,
    )

    fake_llm.push_action("行動を選べませんでした")
    fake_llm.push("こんばんは。")
    await client.post("/api/chat", json={"text": "ただいま"})

    async with session_factory() as session:
        record = (await session.execute(select(ActionRecord))).scalars().all()[-1]
    assert record.action == Action.WAIT.value
    assert record.is_fallback is True
    assert "読み取れません" in record.reason
