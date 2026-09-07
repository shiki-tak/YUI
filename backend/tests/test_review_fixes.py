"""レビュー指摘の再発を検知するテスト。

対象：docs/review/codex/phase1_review.md と phase1_review2.md
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models import Conversation, Memory, utcnow
from tests.conftest import FakeLLM


async def _say(
    client: AsyncClient,
    text: str,
    speaker: dict | None = None,
    conversation_id: int | None = None,
) -> dict:
    """発言を送る。conversation_id を渡さないと新しい会話になる点に注意。"""
    payload: dict = {"text": text}
    if speaker is not None:
        payload["speaker"] = speaker
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    response = await client.post("/api/chat", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


# --- #1 生成中に他の書き込みが止まらないこと -------------------------------


async def test_other_writes_succeed_while_generating(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
):
    """LLM の応答待ち中に記憶を訂正しても database is locked にならない。

    以前は発言を flush したトランザクションを保持したまま生成を待っていたため、
    別リクエストの書き込みが SQLite のロック待ちで失敗した。
    """
    first = await _say(client, "最初の発言")
    created = await client.post(
        "/api/memories", json={"kind": "experience", "content": "元の内容", "keywords": "元"}
    )
    memory_id = created.json()["id"]
    assert first["conversation_id"]

    fake_llm.entered.clear()
    fake_llm.gate = asyncio.Event()
    chat_task = asyncio.create_task(_say(client, "生成に時間がかかる発言"))

    # 生成に入るまで待つ（＝この時点でトランザクションが閉じているはず）。
    await asyncio.wait_for(fake_llm.entered.wait(), timeout=5)

    async with session_factory() as session:
        memory = await session.get(Memory, memory_id)
        memory.content = "生成中に訂正した内容"
        await asyncio.wait_for(session.commit(), timeout=5)

    fake_llm.gate.set()
    await asyncio.wait_for(chat_task, timeout=5)

    listed = (await client.get("/api/memories")).json()
    assert [m["content"] for m in listed if m["id"] == memory_id] == ["生成中に訂正した内容"]


# --- #2 非公開記憶が別の相手へ渡らないこと ---------------------------------

SPEAKER_A = {"source": "local", "external_id": "alice", "display_name": "アリス"}
SPEAKER_B = {"source": "local", "external_id": "bob", "display_name": "ボブ"}


async def test_private_memory_is_not_shared_with_another_speaker(
    client: AsyncClient, fake_llm: FakeLLM
):
    """A との会話に由来する非公開記憶を、B の会話へ渡さない。"""
    turn_a = await _say(client, "登山の話をした", speaker=SPEAKER_A)
    speaker_a_id = turn_a["user_message"]["speaker_id"]

    created = await client.post(
        "/api/memories",
        json={
            "kind": "experience",
            "content": "アリスと登山の計画を立てた",
            "keywords": "登山 計画",
            # 「誰について」ではなく「誰との会話で参照してよいか」を指定する。
            "subject_speaker_id": None,
            "visible_to_speaker_id": speaker_a_id,
            "visibility": "private",
        },
    )
    assert created.status_code == 201, created.text
    memory_id = created.json()["id"]

    again_a = await _say(client, "登山の計画はどうなった？", speaker=SPEAKER_A)
    assert memory_id in [m["memory"]["id"] for m in again_a["used_memories"]]

    turn_b = await _say(client, "登山の計画はどうなった？", speaker=SPEAKER_B)
    assert memory_id not in [m["memory"]["id"] for m in turn_b["used_memories"]]
    assert "アリスと登山" not in fake_llm.last_system_prompt


async def test_public_memory_is_shared_across_speakers(client: AsyncClient):
    """公開してよい記憶は相手を問わず使える。"""
    await _say(client, "はじめまして", speaker=SPEAKER_A)
    created = await client.post(
        "/api/memories",
        json={
            "kind": "fact",
            "content": "YUIはカメラの話が好き",
            "keywords": "カメラ 好き",
            "visibility": "public",
        },
    )
    memory_id = created.json()["id"]

    turn_b = await _say(client, "カメラの話をしよう", speaker=SPEAKER_B)
    assert memory_id in [m["memory"]["id"] for m in turn_b["used_memories"]]


async def test_stream_mode_excludes_private_memories(client: AsyncClient):
    turn = await _say(client, "配信の準備をする", speaker=SPEAKER_A)
    speaker_a_id = turn["user_message"]["speaker_id"]
    await client.post(
        "/api/memories",
        json={
            "kind": "experience",
            "content": "配信では話さない内緒の話",
            "keywords": "内緒 配信",
            "visible_to_speaker_id": speaker_a_id,
            "visibility": "private",
        },
    )
    result = (
        await client.get(
            "/api/memories/search",
            params={"q": "内緒の配信の話", "speaker_id": speaker_a_id, "mode": "stream"},
        )
    ).json()
    assert result["results"] == []


# --- #3 会話の終了が二重に走らないこと -------------------------------------


async def test_concurrent_end_produces_one_set_of_candidates(
    client: AsyncClient, fake_llm: FakeLLM
):
    first = await _say(client, "写真の話をした")
    conversation_id = first["conversation_id"]

    fake_llm.scripted = [
        json.dumps(
            [{"kind": "about_person", "content": "写真が好き", "certainty": "fact",
              "keywords": "写真", "about_partner": True,
              "source_message_id": first["user_message"]["id"]}],
            ensure_ascii=False,
        )
    ] * 2

    responses = await asyncio.gather(
        client.post(f"/api/conversations/{conversation_id}/end"),
        client.post(f"/api/conversations/{conversation_id}/end"),
    )
    codes = sorted(r.status_code for r in responses)
    assert codes == [200, 409], [r.text for r in responses]

    candidates = (
        await client.get(f"/api/conversations/{conversation_id}/candidates")
    ).json()
    assert len(candidates) == 1


async def test_failed_reflection_can_be_retried(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
):
    """抽出に失敗した会話を終了済みのまま残さず、やり直せるようにする。"""
    from app.llm.base import LLMError

    first = await _say(client, "振り返りに失敗する会話")
    conversation_id = first["conversation_id"]

    original_chat = fake_llm.chat

    async def failing_chat(*args, **kwargs):
        raise LLMError("接続できません")

    fake_llm.chat = failing_chat  # type: ignore[method-assign]
    failed = await client.post(f"/api/conversations/{conversation_id}/end")
    assert failed.status_code == 503

    async with session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        assert conversation.ended_at is None, "失敗した会話は終了済みにしない"

    fake_llm.chat = original_chat  # type: ignore[method-assign]
    fake_llm.scripted = ["[]"]
    retried = await client.post(f"/api/conversations/{conversation_id}/end")
    assert retried.status_code == 200


# --- #4 候補の根拠が発言ごとに正しいこと -----------------------------------


async def test_candidate_keeps_its_own_source_message(
    client: AsyncClient, fake_llm: FakeLLM
):
    """根拠が一律で最後の発言にならず、候補ごとの発言を指す。"""
    first = await _say(client, "写真を撮るのが趣味なんだ")
    conversation_id = first["conversation_id"]
    photo_message_id = first["user_message"]["id"]
    last = await _say(client, "今日はここまでにしよう", conversation_id=conversation_id)
    # 同じ会話の 2 ターン目であることを確かめる。別会話だと検証にならない。
    assert last["conversation_id"] == conversation_id
    last_message_id = last["user_message"]["id"]
    assert photo_message_id != last_message_id

    fake_llm.push(
        json.dumps(
            [
                {
                    "kind": "about_person",
                    "content": "開発者の趣味は写真",
                    "certainty": "fact",
                    "keywords": "写真 趣味",
                    "about_partner": True,
                    "source_message_id": photo_message_id,
                }
            ],
            ensure_ascii=False,
        )
    )
    candidates = (
        await client.post(f"/api/conversations/{first['conversation_id']}/end")
    ).json()
    assert candidates[0]["source_message_id"] == photo_message_id

    accepted = await client.post(
        f"/api/conversations/candidates/{candidates[0]['id']}/decide",
        json={"decision": "accept"},
    )
    memory_id = accepted.json()["accepted_memory_id"]
    memories = (await client.get("/api/memories")).json()
    target = next(m for m in memories if m["id"] == memory_id)
    assert target["source_message_id"] == photo_message_id

    # 根拠の発言の本文を画面から確認できる。
    message = (
        await client.get(f"/api/conversations/messages/{photo_message_id}")
    ).json()
    assert message["content"] == "写真を撮るのが趣味なんだ"


async def test_invented_source_message_is_left_unverified(
    client: AsyncClient, fake_llm: FakeLLM
):
    """存在しない発言番号を返されたら、無関係な発言へ付け替えず未確認にする。"""
    first = await _say(client, "写真を撮るのが趣味なんだ")
    conversation_id = first["conversation_id"]
    last = await _say(client, "今日はここまでにしよう", conversation_id=conversation_id)

    fake_llm.push(
        json.dumps(
            [{"kind": "about_person", "content": "開発者の趣味は写真", "certainty": "fact",
              "keywords": "写真", "about_partner": True, "source_message_id": 9999}],
            ensure_ascii=False,
        )
    )
    candidates = (await client.post(f"/api/conversations/{conversation_id}/end")).json()
    assert candidates[0]["source_message_id"] is None
    assert candidates[0]["source_message_id"] != last["user_message"]["id"]


# --- #7 画面と会話で同じ検索結果になること ---------------------------------


async def test_search_with_speaker_matches_conversation(client: AsyncClient):
    turn = await _say(client, "山の写真の話")
    speaker_id = turn["user_message"]["speaker_id"]
    await client.post(
        "/api/memories",
        json={
            "kind": "about_person",
            "content": "開発者は写真を撮るのが好き",
            "keywords": "写真 撮影",
            "subject_speaker_id": speaker_id,
            "visible_to_speaker_id": speaker_id,
        },
    )
    result = (
        await client.get("/api/memories/search", params={"q": "写真", "speaker_id": speaker_id})
    ).json()
    assert len(result["results"]) == 1

    speakers = (await client.get("/api/speakers")).json()
    assert any(s["external_id"] == "developer" and s["id"] == speaker_id for s in speakers)


@pytest.mark.parametrize("include_speaker", [True, False])
async def test_search_speaker_scope(client: AsyncClient, include_speaker: bool):
    """相手を指定しない検索では、相手に紐づく記憶は引かない（画面は指定する）。"""
    turn = await _say(client, "犬の話")
    speaker_id = turn["user_message"]["speaker_id"]
    await client.post(
        "/api/memories",
        json={
            "kind": "about_person",
            "content": "開発者は犬が好き",
            "keywords": "犬",
            "subject_speaker_id": speaker_id,
            "visible_to_speaker_id": speaker_id,
        },
    )
    params = {"q": "犬"}
    if include_speaker:
        params["speaker_id"] = speaker_id
    result = (await client.get("/api/memories/search", params=params)).json()
    assert len(result["results"]) == (1 if include_speaker else 0)


# --- 変更履歴に参照範囲が残ること -----------------------------------------


async def test_memory_revision_records_visibility_scope(client: AsyncClient):
    """変更履歴に参照範囲が残り、復元で戻せる。"""
    turn = await _say(client, "参照範囲の記録")
    speaker_id = turn["user_message"]["speaker_id"]
    created = await client.post(
        "/api/memories",
        json={
            "kind": "experience",
            "content": "範囲つき",
            "keywords": "範囲",
            "visible_to_speaker_id": speaker_id,
        },
    )
    memory_id = created.json()["id"]
    revisions = (await client.get(f"/api/memories/{memory_id}/revisions")).json()
    assert revisions[0]["after"]["visible_to_speaker_id"] == speaker_id


# --- 再評価（phase1_review2.md）で見つかった問題 ---------------------------


async def test_new_candidate_always_has_visibility_scope(
    client: AsyncClient, fake_llm: FakeLLM
):
    """振り返りで新しく作る候補は、必ず参照範囲を持つ。

    移行済みデータの検証は test_migration.py で行う（ここでは新規分だけ）。
    """
    first = await _say(client, "内緒の話をした", speaker=SPEAKER_A)
    speaker_a_id = first["user_message"]["speaker_id"]
    fake_llm.push(
        json.dumps(
            [{"kind": "experience", "content": "アリスと内緒の計画を立てた", "certainty": "fact",
              "keywords": "内緒 計画", "about_partner": False,
              "source_message_id": first["user_message"]["id"]}],
            ensure_ascii=False,
        )
    )
    candidates = (
        await client.post(f"/api/conversations/{first['conversation_id']}/end")
    ).json()
    assert candidates[0]["subject_speaker_id"] is None
    assert candidates[0]["visible_to_speaker_id"] == speaker_a_id

    accepted = await client.post(
        f"/api/conversations/candidates/{candidates[0]['id']}/decide",
        json={"decision": "accept"},
    )
    memory_id = accepted.json()["accepted_memory_id"]

    # 別の相手との会話には渡らない。
    turn_b = await _say(client, "内緒の計画はどうなった？", speaker=SPEAKER_B)
    assert memory_id not in [m["memory"]["id"] for m in turn_b["used_memories"]]


async def test_same_conversation_keeps_turn_order(client: AsyncClient, fake_llm: FakeLLM):
    """同じ会話へ同時に送っても、質問と返答の順序が入れ替わらない。

    生成前にトランザクションを閉じた結果、会話単位の直列化が無いと
    「質問A → 質問B → 返答B → 返答A」の順に記録されていた。
    """
    first = await _say(client, "最初")
    conversation_id = first["conversation_id"]

    fake_llm.entered.clear()
    fake_llm.gate = asyncio.Event()
    fake_llm.scripted = ["返答A", "返答B"]

    slow = asyncio.create_task(
        _say(client, "質問A", conversation_id=conversation_id)
    )
    await asyncio.wait_for(fake_llm.entered.wait(), timeout=5)

    second = asyncio.create_task(
        _say(client, "質問B", conversation_id=conversation_id)
    )
    await asyncio.sleep(0.05)  # 2 本目がロック待ちに入る時間を与える
    fake_llm.gate.set()
    await asyncio.wait_for(asyncio.gather(slow, second), timeout=5)

    detail = (await client.get(f"/api/conversations/{conversation_id}")).json()
    order = [m["content"] for m in detail["messages"]]
    assert order.index("質問A") < order.index("返答A") < order.index("質問B")
    assert order.index("質問B") < order.index("返答B")


async def test_interrupted_reflection_can_be_retried(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
):
    """振り返り中に中断されても、時間が経てばやり直せる。

    以前は生成前に ended_at を確定し、戻すのは LLMError のときだけだったため、
    キャンセルされると候補0件のまま再試行が409になっていた。
    """
    first = await _say(client, "中断される会話")
    conversation_id = first["conversation_id"]

    fake_llm.entered.clear()
    fake_llm.gate = asyncio.Event()
    task = asyncio.create_task(client.post(f"/api/conversations/{conversation_id}/end"))
    await asyncio.wait_for(fake_llm.entered.wait(), timeout=5)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    fake_llm.gate.set()

    candidates = (
        await client.get(f"/api/conversations/{conversation_id}/candidates")
    ).json()
    assert candidates == []

    # 処理中の間は再試行を弾く。
    busy = await client.post(f"/api/conversations/{conversation_id}/end")
    assert busy.status_code == 409

    # 一定時間が過ぎたら回収できる。
    async with session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        conversation.reflection_started_at = utcnow() - timedelta(hours=1)
        await session.commit()

    fake_llm.gate = None
    fake_llm.scripted = ["[]"]
    retried = await client.post(f"/api/conversations/{conversation_id}/end")
    assert retried.status_code == 200


async def test_completed_reflection_is_not_repeated(client: AsyncClient, fake_llm: FakeLLM):
    """完了した振り返りは、時間が経っても再実行しない。"""
    first = await _say(client, "一度だけ振り返る会話")
    conversation_id = first["conversation_id"]
    fake_llm.push("[]")
    assert (
        await client.post(f"/api/conversations/{conversation_id}/end")
    ).status_code == 200

    again = await client.post(f"/api/conversations/{conversation_id}/end")
    assert again.status_code == 409


# --- 3回目のレビューで見つかった問題 ---------------------------------------


async def test_message_is_rejected_while_reflecting(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
):
    """振り返り中に送った発言が、終了後の会話に追加されない。

    以前は終了判定がロック取得の前だけだったため、ロック待ちの間に振り返りが
    完了し、終了済みの会話へ発言と返答が入っていた。その分は振り返りに
    含まれず、完了済みのため再抽出もできなかった。
    """
    first = await _say(client, "最初の発言")
    conversation_id = first["conversation_id"]

    fake_llm.entered.clear()
    fake_llm.gate = asyncio.Event()
    fake_llm.scripted = ["[]"]
    end_task = asyncio.create_task(
        client.post(f"/api/conversations/{conversation_id}/end")
    )
    await asyncio.wait_for(fake_llm.entered.wait(), timeout=5)

    chat_task = asyncio.create_task(
        client.post(
            "/api/chat",
            json={"text": "振り返り中に割り込む発言", "conversation_id": conversation_id},
        )
    )
    await asyncio.sleep(0.05)

    gate = fake_llm.gate
    fake_llm.gate = None
    gate.set()

    end_response = await asyncio.wait_for(end_task, timeout=10)
    chat_response = await asyncio.wait_for(chat_task, timeout=10)
    assert end_response.status_code == 200
    assert chat_response.status_code == 409

    detail = (await client.get(f"/api/conversations/{conversation_id}")).json()
    assert "振り返り中に割り込む発言" not in [m["content"] for m in detail["messages"]]

    async with session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        assert conversation.reflection_completed_at is not None


async def test_running_reflection_is_not_reclaimed_after_stale_window(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
):
    """稼働中の振り返りが、回収期限を過ぎても二重に実行されない。

    以前は開始時刻をロック取得の前に設定していたため、ロック待ちの時間が
    回収期限に含まれ、稼働中の処理まで期限切れとみなされて候補が二重に
    保存された。
    """
    first = await _say(client, "二重振り返りの検証")
    conversation_id = first["conversation_id"]
    payload = json.dumps(
        [{"kind": "experience", "content": "検証用の経験", "certainty": "fact",
          "keywords": "検証", "about_partner": False,
          "source_message_id": first["user_message"]["id"]}],
        ensure_ascii=False,
    )

    fake_llm.entered.clear()
    fake_llm.gate = asyncio.Event()
    fake_llm.scripted = [payload, payload]

    running = asyncio.create_task(
        client.post(f"/api/conversations/{conversation_id}/end")
    )
    await asyncio.wait_for(fake_llm.entered.wait(), timeout=5)

    # 稼働中のまま、開始時刻だけ期限切れにする（ロック待ちが長引いた状況）。
    async with session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        conversation.reflection_started_at = utcnow() - timedelta(seconds=3600)
        await session.commit()

    second = asyncio.create_task(
        client.post(f"/api/conversations/{conversation_id}/end")
    )
    await asyncio.sleep(0.05)

    gate = fake_llm.gate
    fake_llm.gate = None
    gate.set()

    first_response = await asyncio.wait_for(running, timeout=10)
    second_response = await asyncio.wait_for(second, timeout=10)
    assert sorted([first_response.status_code, second_response.status_code]) == [200, 409]

    candidates = (
        await client.get(f"/api/conversations/{conversation_id}/candidates")
    ).json()
    assert len(candidates) == 1


# --- #6 振り返りの失敗を成功として扱わないこと -----------------------------


async def test_unparsable_reflection_is_reported_and_retryable(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
):
    """JSON として読めない出力を「候補なしの成功」として扱わない。"""
    first = await _say(client, "壊れた出力を返す会話")
    conversation_id = first["conversation_id"]

    fake_llm.push("すみません、うまくまとめられませんでした。")
    failed = await client.post(f"/api/conversations/{conversation_id}/end")
    assert failed.status_code == 503
    detail = failed.json()["detail"]
    assert "読み取れません" in detail or "見つかりません" in detail

    async with session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        assert conversation.ended_at is None
        assert conversation.reflection_completed_at is None

    # 会話は続けられ、振り返りもやり直せる。
    resumed = await client.post(
        "/api/chat", json={"text": "続けます", "conversation_id": conversation_id}
    )
    assert resumed.status_code == 200

    fake_llm.push("[]")
    retried = await client.post(f"/api/conversations/{conversation_id}/end")
    assert retried.status_code == 200
    assert retried.json() == []


@pytest.mark.parametrize(
    "output,expected_reason",
    [
        ("[{}]", "Field required"),
        ('[{"kind":"promise","content":42}]', "valid string"),
        ('[{"kind":"unknown","content":"何か"}]', "種別が不正"),
        ('[{"kind":"promise","content":"何か","certainty":"maybe"}]', "確かさが不正"),
        ('["文字列"]', "オブジェクトではありません"),
        # ISSUE-001：空白・改行・タブだけの本文。検証を通してから strip() すると
        # 本文が空の候補として保存され、抽出に成功したことになっていた。
        ('[{"kind":"promise","content":"   \\n\\t"}]', "at least 1 character"),
    ],
)
async def test_invalid_candidate_element_is_reported(
    client: AsyncClient,
    fake_llm: FakeLLM,
    session_factory: async_sessionmaker,
    output: str,
    expected_reason: str,
):
    """配列は読めても、候補として読み取れない要素があれば失敗として扱う。

    以前は要素を黙って捨てていたため、全要素が不正でも「候補なしの成功」に
    なり、会話が終了して再抽出できなくなっていた。
    """
    first = await _say(client, f"不正な要素を返す会話: {output}")
    conversation_id = first["conversation_id"]

    fake_llm.push(output)
    failed = await client.post(f"/api/conversations/{conversation_id}/end")
    assert failed.status_code == 503
    detail = failed.json()["detail"]
    assert "読み取れない候補" in detail
    assert expected_reason in detail

    async with session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        assert conversation.ended_at is None
        assert conversation.reflection_completed_at is None

    fake_llm.push("[]")
    assert (
        await client.post(f"/api/conversations/{conversation_id}/end")
    ).status_code == 200


async def test_partially_invalid_candidates_fail_as_a_whole(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
):
    """有効な候補が混ざっていても、1件でも読めなければ全体をやり直す。

    落とした候補は記憶になり損ねた経験であり、静かに捨てると失われたことに
    気づけない。振り返りの再実行は短時間で済む。
    """
    first = await _say(client, "一部だけ不正な出力の会話")
    conversation_id = first["conversation_id"]
    message_id = first["user_message"]["id"]

    fake_llm.push(
        json.dumps(
            [
                {"kind": "about_person", "content": "有効な候補", "certainty": "fact",
                 "keywords": "有効", "about_partner": True, "source_message_id": message_id},
                {"kind": "promise", "content": 42},
            ],
            ensure_ascii=False,
        )
    )
    failed = await client.post(f"/api/conversations/{conversation_id}/end")
    assert failed.status_code == 503
    assert "2件目" in failed.json()["detail"]

    # 有効だった候補も保存しない（部分的に採らない）。
    candidates = (
        await client.get(f"/api/conversations/{conversation_id}/candidates")
    ).json()
    assert candidates == []

    async with session_factory() as session:
        conversation = await session.get(Conversation, conversation_id)
        assert conversation.reflection_completed_at is None


async def test_valid_candidates_are_kept(client: AsyncClient, fake_llm: FakeLLM):
    """すべて読み取れる場合は、そのまま候補として保存する。"""
    first = await _say(client, "正常な出力の会話")
    message_id = first["user_message"]["id"]
    fake_llm.push(
        json.dumps(
            [
                {"kind": "about_person", "content": "ひとつめ", "certainty": "fact",
                 "keywords": "1", "about_partner": True, "source_message_id": message_id},
                {"kind": "experience", "content": "ふたつめ", "certainty": "inference",
                 "keywords": "2", "about_partner": False, "source_message_id": message_id},
            ],
            ensure_ascii=False,
        )
    )
    response = await client.post(f"/api/conversations/{first['conversation_id']}/end")
    assert response.status_code == 200
    assert [c["content"] for c in response.json()] == ["ひとつめ", "ふたつめ"]


async def test_empty_reflection_is_a_success(client: AsyncClient, fake_llm: FakeLLM):
    """残す価値が無い会話は、空の結果として正常に終了する。"""
    first = await _say(client, "とくに残すことのない雑談")
    fake_llm.push("[]")
    response = await client.post(
        f"/api/conversations/{first['conversation_id']}/end"
    )
    assert response.status_code == 200
    assert response.json() == []


# --- #5 未判断の候補を会話をまたいで回収できること ---------------------------


async def test_pending_candidates_survive_across_conversations(
    client: AsyncClient, fake_llm: FakeLLM
):
    """画面を再読み込みしても、未判断の候補を採用できる。"""
    first = await _say(client, "ひとつめの会話")
    fake_llm.push(
        json.dumps(
            [{"kind": "experience", "content": "ひとつめの候補", "certainty": "fact",
              "keywords": "ひとつめ", "about_partner": False,
              "source_message_id": first["user_message"]["id"]}],
            ensure_ascii=False,
        )
    )
    await client.post(f"/api/conversations/{first['conversation_id']}/end")

    second = await _say(client, "ふたつめの会話")
    fake_llm.push(
        json.dumps(
            [{"kind": "experience", "content": "ふたつめの候補", "certainty": "fact",
              "keywords": "ふたつめ", "about_partner": False,
              "source_message_id": second["user_message"]["id"]}],
            ensure_ascii=False,
        )
    )
    await client.post(f"/api/conversations/{second['conversation_id']}/end")

    pending = (await client.get("/api/conversations/candidates/pending")).json()
    contents = [c["content"] for c in pending]
    assert "ひとつめの候補" in contents
    assert "ふたつめの候補" in contents

    # 判断したものは一覧から外れる。
    await client.post(
        f"/api/conversations/candidates/{pending[0]['id']}/decide",
        json={"decision": "reject"},
    )
    remaining = (await client.get("/api/conversations/candidates/pending")).json()
    assert pending[0]["id"] not in [c["id"] for c in remaining]


# --- #8 復元で日時も戻ること -----------------------------------------------


async def test_restore_brings_back_occurred_at(client: AsyncClient):
    created = await client.post(
        "/api/memories",
        json={
            "kind": "experience",
            "content": "日時つきの記憶",
            "keywords": "日時",
            "occurred_at": "2026-08-01T10:00:00+00:00",
        },
    )
    memory_id = created.json()["id"]
    await client.patch(
        f"/api/memories/{memory_id}",
        json={"content": "訂正後", "occurred_at": "2026-09-01T10:00:00+00:00"},
    )
    restored = (await client.post(f"/api/memories/{memory_id}/restore")).json()
    assert restored["content"] == "日時つきの記憶"
    assert restored["occurred_at"].startswith("2026-08-01T10:00:00")


async def test_restore_brings_back_null_occurred_at(client: AsyncClient):
    created = await client.post(
        "/api/memories", json={"kind": "experience", "content": "日時なし", "keywords": "なし"}
    )
    memory_id = created.json()["id"]
    await client.patch(
        f"/api/memories/{memory_id}", json={"occurred_at": "2026-09-01T10:00:00+00:00"}
    )
    restored = (await client.post(f"/api/memories/{memory_id}/restore")).json()
    assert restored["occurred_at"] is None
