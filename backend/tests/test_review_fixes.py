"""レビュー指摘（docs/codex/phase1_review.md）の再発を検知するテスト。"""

from __future__ import annotations

import asyncio
import json

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models import Conversation, Memory
from tests.conftest import FakeLLM


async def _say(client: AsyncClient, text: str, speaker: dict | None = None) -> dict:
    payload: dict = {"text": text}
    if speaker is not None:
        payload["speaker"] = speaker
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
            "content": "ゆいはカメラの話が好き",
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
    photo_message_id = first["user_message"]["id"]
    last = await _say(client, "今日はここまでにしよう")
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


async def test_invented_source_message_falls_back(client: AsyncClient, fake_llm: FakeLLM):
    """存在しない発言番号を返されたら、その会話の発言に寄せる。"""
    first = await _say(client, "架空の番号を返す会話")
    fake_llm.push(
        json.dumps(
            [{"kind": "experience", "content": "何かの経験", "certainty": "fact",
              "keywords": "経験", "about_partner": False, "source_message_id": 9999}],
            ensure_ascii=False,
        )
    )
    candidates = (
        await client.post(f"/api/conversations/{first['conversation_id']}/end")
    ).json()
    assert candidates[0]["source_message_id"] == first["user_message"]["id"]


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
