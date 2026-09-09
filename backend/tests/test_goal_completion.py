"""完了・繰り返し防止・予定変更（フェーズ4 PR9 / 完了条件3・4）。

PR8 から渡した条件を守る。

- 実行（届いた）と達成（答えが返った）を混ぜない。
- 実行済みだけで恒久的に再質問を止めない。止めるのは達成のときだけ。
- 中断のときは聞き直せる。
"""

from __future__ import annotations

from datetime import timedelta

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agent.goal import active_goals, create_goal, expire_overdue
from app.models import Goal, GoalRevision, GoalStatus, GoalTrigger, utcnow
from tests.conftest import FakeLLM


async def _setup(client: AsyncClient, session_factory: async_sessionmaker) -> tuple[int, int]:
    first = await client.post("/api/chat", json={"text": "こんにちは"})
    speaker_id = first.json()["user_message"]["speaker_id"]
    async with session_factory() as session:
        goal = await create_goal(
            session,
            content="土曜に見た映画の感想を聞く",
            subject_speaker_id=speaker_id,
            visible_to_speaker_id=speaker_id,
            status=GoalStatus.ACTIVE.value,
        )
        await session.commit()
        return first.json()["conversation_id"], goal.id


async def _ask_and_deliver(
    client: AsyncClient, fake_llm: FakeLLM, conversation_id: int
) -> None:
    """YUI から質問し、相手へ届いたことにする。"""
    fake_llm.push_action('{"action": "ask", "goal": 1, "reason": "聞ける"}')
    fake_llm.push("土曜の映画、どうでしたか？")
    opened = await client.post(f"/api/conversations/{conversation_id}/open")
    message_id = opened.json()["message"]["id"]
    await client.post(
        f"/api/conversations/messages/{message_id}/delivery", json={"state": "completed"}
    )


async def test_an_answer_completes_the_goal(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """相手の返答を受けて達成にする（完了条件3）。"""
    conversation_id, goal_id = await _setup(client, session_factory)
    await _ask_and_deliver(client, fake_llm, conversation_id)

    fake_llm.push_action(
        '{"action": "answer", "goal": null, "reason": "感想をもらった",'
        ' "answered": [1], "cancelled": []}'
    )
    fake_llm.push("それは良かったですね。")
    await client.post(
        "/api/chat",
        json={"text": "すごく良かったよ。音楽が印象的だった", "conversation_id": conversation_id},
    )

    async with session_factory() as session:
        goal = await session.get(Goal, goal_id)
        assert goal.status == GoalStatus.DONE.value
        assert goal.completed_at is not None
        # 実行と達成は別に残る。
        assert goal.last_executed_at is not None
        done = (
            await session.execute(
                select(GoalRevision).where(
                    GoalRevision.goal_id == goal_id, GoalRevision.action == "done"
                )
            )
        ).scalar_one()
        assert "相手が答えた" in done.reason


async def test_a_completed_goal_is_not_raised_again(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """達成した目標は、二度と行動選択に渡らない（完了条件3）。"""
    conversation_id, goal_id = await _setup(client, session_factory)
    await _ask_and_deliver(client, fake_llm, conversation_id)

    fake_llm.push_action(
        '{"action": "answer", "goal": null, "reason": "答えをもらった",'
        ' "answered": [1], "cancelled": []}'
    )
    fake_llm.push("よかったですね。")
    await client.post(
        "/api/chat", json={"text": "面白かったよ", "conversation_id": conversation_id}
    )

    async with session_factory() as session:
        goal = await session.get(Goal, goal_id)
        assert await active_goals(session, speaker_id=goal.subject_speaker_id) == []


async def test_an_executed_goal_can_be_asked_again_later(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """答えてもらえなかった質問は、時間が経てばまた聞ける。

    **実行済みだけで恒久的に止めない**（PR8 から渡した条件2）。間隔を置くだけで、
    禁止ではない。
    """
    conversation_id, goal_id = await _setup(client, session_factory)
    await _ask_and_deliver(client, fake_llm, conversation_id)

    async with session_factory() as session:
        goal = await session.get(Goal, goal_id)
        speaker_id = goal.subject_speaker_id
        # 聞いた直後は渡らない。同じことを続けて聞かないため。
        assert await active_goals(
            session, speaker_id=speaker_id, reask_interval_hours=12
        ) == []
        # 時間が経てばまた渡る。達成していないので終わっていない。
        later = utcnow() + timedelta(hours=13)
        again = await active_goals(
            session, speaker_id=speaker_id, now=later, reask_interval_hours=12
        )
        assert [g.id for g in again] == [goal_id]


async def test_a_cancelled_plan_cancels_the_goal(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """予定が無くなったら取り消す（完了条件4）。達成とは別の終わり方。"""
    conversation_id, goal_id = await _setup(client, session_factory)

    fake_llm.push_action(
        '{"action": "answer", "goal": null, "reason": "予定が消えた",'
        ' "answered": [], "cancelled": [1]}'
    )
    fake_llm.push("そうだったんですね。")
    await client.post(
        "/api/chat",
        json={"text": "土曜の映画、やめにしたんだ", "conversation_id": conversation_id},
    )

    async with session_factory() as session:
        goal = await session.get(Goal, goal_id)
        assert goal.status == GoalStatus.CANCELLED.value
        # 取り消しは達成ではない。完了の時刻は入れない。
        assert goal.completed_at is None
        cancelled = (
            await session.execute(
                select(GoalRevision).where(
                    GoalRevision.goal_id == goal_id, GoalRevision.action == "cancelled"
                )
            )
        ).scalar_one()
        assert "前提が無くなった" in cancelled.reason


async def test_an_answer_completes_even_without_a_delivery_notice(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """再生の通知が無くても、相手が答えたなら達成にする。

    **返答そのものが到達の証拠である。** 通知は欠けることがあり（ISSUE-013）、
    必須にすると、答えをもらったのに達成にならず、次の会話で同じことを聞く。
    """
    conversation_id, goal_id = await _setup(client, session_factory)

    # 質問はするが、再生の通知は送らない。
    fake_llm.push_action('{"action": "ask", "goal": 1, "reason": "聞ける"}')
    fake_llm.push("土曜の映画、どうでしたか？")
    await client.post(f"/api/conversations/{conversation_id}/open")

    fake_llm.push_action(
        '{"action": "answer", "goal": null, "reason": "答えをもらった",'
        ' "answered": [1], "cancelled": []}'
    )
    fake_llm.push("よかったですね。")
    await client.post(
        "/api/chat", json={"text": "すごく良かったよ", "conversation_id": conversation_id}
    )

    async with session_factory() as session:
        goal = await session.get(Goal, goal_id)
        assert goal.status == GoalStatus.DONE.value
        # 届いたことも記録する。返答が証拠になる。
        assert goal.last_executed_at is not None


async def test_an_answer_after_an_aborted_question_still_completes(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """音が鳴らずに止めた質問でも、相手が答えたなら達成にする。

    **再生の状態で達成を止めない**（PR10 レビューの指摘4）。一度は「aborted で
    開始時刻が無いものは届いていない」として `_raised_goal_ids` から除いたが、
    次の2つの理由で戻した。

    - `playing` の通知だけが落ちて `aborted` が届くと、鳴っていても開始時刻は
      NULL になる（ISSUE-013）。**開始時刻が無いことは未到達の証拠ではない。**
    - ローカルの会話では文字が画面に出ている。音が鳴らなくても相手は読める。

    除いた状態では、この具体的な返答を達成にできなかった。
    上の `test_an_answer_completes_even_without_a_delivery_notice` と対で読むこと。
    """
    conversation_id, goal_id = await _setup(client, session_factory)

    # 質問はしたが、鳴り始める前に止めた（playing を送らずに aborted）。
    fake_llm.push_action('{"action": "ask", "goal": 1, "reason": "聞ける"}')
    fake_llm.push("土曜の映画、どうでしたか？")
    opened = await client.post(f"/api/conversations/{conversation_id}/open")
    message_id = opened.json()["message"]["id"]
    await client.post(
        f"/api/conversations/messages/{message_id}/delivery", json={"state": "aborted"}
    )

    # 音は鳴らなかったが、相手は画面で読んで具体的に答えている。
    fake_llm.push_action(
        '{"action": "answer", "goal": null, "reason": "感想をもらった",'
        ' "answered": [1], "cancelled": []}'
    )
    fake_llm.push("それは良かったですね。")
    await client.post(
        "/api/chat",
        json={
            "text": "質問は画面で読んだよ。映画は面白かった。音楽が特に良かった",
            "conversation_id": conversation_id,
        },
    )

    async with session_factory() as session:
        goal = await session.get(Goal, goal_id)
        assert goal.status == GoalStatus.DONE.value
        assert goal.completed_at is not None


async def test_a_goal_that_was_never_asked_is_not_completed(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """聞いていない目標を、答えたことにしない。

    根拠のない完了は、繰り返し防止の裏返しで「聞いていないのに二度と聞かない」
    ことになる。
    """
    conversation_id, goal_id = await _setup(client, session_factory)

    # 一度も聞いていないのに「答えた」と返す出力。
    fake_llm.push_action(
        '{"action": "answer", "goal": null, "reason": "答えたように見えた",'
        ' "answered": [1], "cancelled": []}'
    )
    fake_llm.push("そうなんですね。")
    await client.post(
        "/api/chat", json={"text": "今日は天気がよかった", "conversation_id": conversation_id}
    )

    async with session_factory() as session:
        goal = await session.get(Goal, goal_id)
        assert goal.status == GoalStatus.ACTIVE.value
        assert goal.completed_at is None


async def test_an_overdue_goal_expires(session_factory: async_sessionmaker) -> None:
    """実行しないまま時期を過ぎた目標を終わらせる（完了条件4）。

    実行済みのものは対象にしない。一度聞いた目標は、期限ではなく達成したか
    どうかで終わらせる。
    """
    async with session_factory() as session:
        overdue = await create_goal(
            session,
            content="ずっと前の予定の感想を聞く",
            trigger=GoalTrigger.AFTER_DATE.value,
            due_at=utcnow() - timedelta(days=30),
            status=GoalStatus.ACTIVE.value,
        )
        asked = await create_goal(
            session,
            content="聞いたが答えてもらえなかった",
            trigger=GoalTrigger.AFTER_DATE.value,
            due_at=utcnow() - timedelta(days=30),
            status=GoalStatus.ACTIVE.value,
        )
        asked.last_executed_at = utcnow() - timedelta(days=29)
        recent = await create_goal(
            session,
            content="まだ期限内",
            trigger=GoalTrigger.AFTER_DATE.value,
            due_at=utcnow() - timedelta(days=2),
            status=GoalStatus.ACTIVE.value,
        )
        await session.commit()

        expired = await expire_overdue(session, after_days=14)
        await session.commit()

        assert [g.id for g in expired] == [overdue.id]
        assert (await session.get(Goal, overdue.id)).status == GoalStatus.EXPIRED.value
        # 聞いた目標は期限で終わらせない。達成したかどうかで決める。
        assert (await session.get(Goal, asked.id)).status == GoalStatus.ACTIVE.value
        assert (await session.get(Goal, recent.id)).status == GoalStatus.ACTIVE.value


# --- 第1回レビューへの対応 --------------------------------------------------


async def test_the_write_lock_is_released_before_generating(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """完了・取消を書いた後、生成を待つ間にロックを持たない（指摘1）。

    1回目のモデル呼び出しの前では閉じていたのに、2回目の前で同じ形を作って
    いた。**モデルを呼ぶ処理の前に、開いている書き込みが無いかを必ず確かめる。**
    """
    import asyncio

    conversation_id, goal_id = await _setup(client, session_factory)

    fake_llm.push_action(
        '{"action": "answer", "goal": null, "reason": "予定が消えた",'
        ' "answered": [], "cancelled": [1]}'
    )
    fake_llm.entered.clear()
    fake_llm.gate = asyncio.Event()
    # **2回目の呼び出し（返答の生成）で止める。** 1回目（行動選択）で止めると、
    # PR7 で直した箇所を測るだけで、完了・取消を書いた後のロックを測れない
    # （第2回レビューの指摘）。返答の生成は人格のプロンプトで呼ばれるため、
    # 指示文では選べない。
    fake_llm.gate_skip = 1

    task = asyncio.create_task(
        client.post(
            "/api/chat",
            json={"text": "土曜の映画やめた", "conversation_id": conversation_id},
        )
    )
    await asyncio.wait_for(fake_llm.held.wait(), timeout=5)

    # 生成を待っている間に、別の接続から書けること。取消もすでに見えている
    # （生成の前に確定している）。
    async with session_factory() as session:
        assert (await session.get(Goal, goal_id)).status == GoalStatus.CANCELLED.value
        written = await create_goal(
            session, content="別の接続から書いた目標", status=GoalStatus.ACTIVE.value
        )
        await session.commit()
        assert written.id is not None

    fake_llm.gate.set()
    fake_llm.gate = None
    assert (await asyncio.wait_for(task, timeout=10)).status_code == 200

    async with session_factory() as session:
        assert (await session.get(Goal, goal_id)).status == GoalStatus.CANCELLED.value


async def test_a_future_plan_can_be_cancelled_before_its_date(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """予定日より前でも、取消を反映する（指摘2）。

    予定の取消は、実行してよくなる日より前に起こる。判定の対象を実行条件で
    絞ると、中止したのに後日その話を持ち出す。
    """
    first = await client.post("/api/chat", json={"text": "こんにちは"})
    speaker_id = first.json()["user_message"]["speaker_id"]
    conversation_id = first.json()["conversation_id"]
    async with session_factory() as session:
        goal = await create_goal(
            session,
            content="土曜に見た映画の感想を聞く",
            subject_speaker_id=speaker_id,
            visible_to_speaker_id=speaker_id,
            trigger=GoalTrigger.AFTER_DATE.value,
            due_at=utcnow() + timedelta(days=7),
            status=GoalStatus.ACTIVE.value,
        )
        await session.commit()
        goal_id = goal.id

    fake_llm.push_action(
        '{"action": "answer", "goal": null, "reason": "予定が消えた",'
        ' "answered": [], "cancelled": [1]}'
    )
    fake_llm.push("そうだったんですね。")
    await client.post(
        "/api/chat",
        json={"text": "土曜の映画、やめにしたんだ", "conversation_id": conversation_id},
    )

    async with session_factory() as session:
        assert (await session.get(Goal, goal_id)).status == GoalStatus.CANCELLED.value
        # 予定日を過ぎても、もう持ち出さない。
        later = utcnow() + timedelta(days=8)
        assert await active_goals(session, speaker_id=speaker_id, now=later) == []


async def test_finishing_the_chosen_goal_does_not_fall_back_to_another(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """選んだ目標を同時に終わらせたら、別の目標へ振り替えない（指摘3）。

    選び直しは行動選択の仕事である。ここで補うと、渡す目標と実行の記録が
    食い違い、再質問の間隔も迂回する。
    """
    from sqlalchemy import select as sa_select

    from app.models import ActionRecord

    first = await client.post("/api/chat", json={"text": "こんにちは"})
    speaker_id = first.json()["user_message"]["speaker_id"]
    conversation_id = first.json()["conversation_id"]
    async with session_factory() as session:
        target = await create_goal(
            session,
            content="土曜に見た映画の感想を聞く",
            subject_speaker_id=speaker_id,
            visible_to_speaker_id=speaker_id,
            status=GoalStatus.ACTIVE.value,
        )
        other = await create_goal(
            session,
            content="買ったカメラを使ったか聞く",
            subject_speaker_id=speaker_id,
            visible_to_speaker_id=speaker_id,
            status=GoalStatus.ACTIVE.value,
        )
        await session.commit()
        target_id, other_id = target.id, other.id

    # 目標1を聞くと言いながら、同じ出力で目標1を取り消す。
    fake_llm.push_action(
        '{"action": "ask", "goal": 1, "reason": "聞く", "answered": [], "cancelled": [1]}'
    )
    fake_llm.push("そうなんですね。")
    reply = await client.post(
        "/api/chat",
        json={"text": "土曜の映画やめた", "conversation_id": conversation_id},
    )

    prompt = fake_llm.last_system_prompt
    # 別の目標を勝手に「聞く」対象へ振り替えない。残った目標は、持っている
    # ものとして渡すが、いまは持ち出さないと指示する。
    assert "いまは持ち出さないでください" in prompt
    assert "1つだけ" not in prompt
    assert reply.json()["run"]["selected_action"] == "answer"
    async with session_factory() as session:
        assert (await session.get(Goal, target_id)).status == GoalStatus.CANCELLED.value
        assert (await session.get(Goal, other_id)).status == GoalStatus.ACTIVE.value
        record = (
            await session.execute(sa_select(ActionRecord))
        ).scalars().all()[-1]
        assert record.selected_goal_ids is None


async def test_the_marks_come_from_separate_sources(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """「すでに聞いた」と「いまは聞き直さない」を、別の根拠で付ける（指摘4）。

    1つにまとめると、通知が無い質問が「聞いていない」ことになり、相手が答えても
    達成にできない。逆に、聞いた印が消えないと間隔を過ぎても聞き直せない。
    """
    conversation_id, goal_id = await _setup(client, session_factory)

    # 再生の通知は送らずに質問する。
    fake_llm.push_action('{"action": "ask", "goal": 1, "reason": "聞ける"}')
    fake_llm.push("土曜の映画、どうでしたか？")
    await client.post(f"/api/conversations/{conversation_id}/open")

    fake_llm.push_action(
        '{"action": "answer", "goal": null, "reason": "答えをもらった",'
        ' "answered": [1], "cancelled": []}'
    )
    fake_llm.push("よかったですね。")
    await client.post(
        "/api/chat", json={"text": "面白かったよ", "conversation_id": conversation_id}
    )

    # 行動選択へ渡した入力に、両方の印が出ている。
    selector_input = [
        call[1].content for call in fake_llm.calls if "持っている目標" in call[1].content
    ][-1]
    assert "（すでに聞いた）" in selector_input
    async with session_factory() as session:
        assert (await session.get(Goal, goal_id)).status == GoalStatus.DONE.value


async def test_the_decision_is_recorded_even_when_the_last_goal_ends(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """最後の目標を終わらせても、その判断を記録する（指摘5）。

    更新後の残りで決めると、状態を変えた判断そのものが記録から消える。
    完了条件5（行動と状態変化の根拠を追える）に対する穴になる。
    """
    from sqlalchemy import select as sa_select

    from app.models import ActionRecord

    conversation_id, goal_id = await _setup(client, session_factory)

    fake_llm.push_action(
        '{"action": "answer", "goal": null, "reason": "予定が消えた",'
        ' "answered": [], "cancelled": [1]}'
    )
    fake_llm.push("そうだったんですね。")
    await client.post(
        "/api/chat",
        json={"text": "土曜の映画やめた", "conversation_id": conversation_id},
    )

    async with session_factory() as session:
        records = list((await session.execute(sa_select(ActionRecord))).scalars())
    assert len(records) == 1
    assert records[0].candidate_goal_ids == [goal_id]
    assert records[0].model == "fake-model"


async def test_a_failed_generation_keeps_the_decision_record(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """返答の生成に失敗しても、状態を変えた判断の記録は残る（第2回レビュー）。

    目標の状態は確定しているのに、何がその状態にしたのかを追えないと、完了条件5
    （行動と状態変化の根拠を追える）を満たせない。
    """
    from sqlalchemy import select as sa_select

    from app.llm.base import LLMError
    from app.models import ActionRecord

    conversation_id, goal_id = await _setup(client, session_factory)

    original = fake_llm.chat
    calls = {"n": 0}

    async def fail_on_generation(messages, *, options=None):
        calls["n"] += 1
        if calls["n"] == 1:  # 行動選択は成功させる
            return await original(messages, options=options)
        raise LLMError("生成に失敗しました")

    fake_llm.push_action(
        '{"action": "answer", "goal": null, "reason": "予定が消えた",'
        ' "answered": [], "cancelled": [1]}'
    )
    fake_llm.chat = fail_on_generation  # type: ignore[method-assign]
    failed = await client.post(
        "/api/chat", json={"text": "土曜の映画やめた", "conversation_id": conversation_id}
    )
    assert failed.status_code == 503

    async with session_factory() as session:
        assert (await session.get(Goal, goal_id)).status == GoalStatus.CANCELLED.value
        records = list((await session.execute(sa_select(ActionRecord))).scalars())
    assert len(records) == 1
    assert records[0].candidate_goal_ids == [goal_id]
    # 話せていないので、発言とは結び付かない。
    assert records[0].message_id is None


async def test_a_question_that_was_never_spoken_is_not_a_basis_for_completion(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """生成に失敗した質問を「持ち出した」ことにしない（第2回レビュー）。

    判断は生成の前に保存するので、失敗した判断も記録に残る。それを根拠に
    達成を許すと、一度も話していない質問を聞いたことにする。
    """
    from app.llm.base import LLMError

    conversation_id, goal_id = await _setup(client, session_factory)

    original = fake_llm.chat
    calls = {"n": 0}

    async def fail_on_generation(messages, *, options=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return await original(messages, options=options)
        raise LLMError("生成に失敗しました")

    # 聞くと決めたが、生成に失敗する。
    fake_llm.push_action('{"action": "ask", "goal": 1, "reason": "聞ける"}')
    fake_llm.chat = fail_on_generation  # type: ignore[method-assign]
    assert (
        await client.post(
            "/api/chat", json={"text": "ただいま", "conversation_id": conversation_id}
        )
    ).status_code == 503

    # 次の発言で「答えた」と判定されても、聞いていないので達成にしない。
    fake_llm.chat = original  # type: ignore[method-assign]
    fake_llm.push_action(
        '{"action": "answer", "goal": null, "reason": "答えた", '
        '"answered": [1], "cancelled": []}'
    )
    fake_llm.push("そうなんですね。")
    await client.post(
        "/api/chat", json={"text": "面白かったよ", "conversation_id": conversation_id}
    )

    async with session_factory() as session:
        goal = await session.get(Goal, goal_id)
        assert goal.status == GoalStatus.ACTIVE.value
        assert goal.completed_at is None
