"""フェーズ1の完了条件に対応するテスト。

- 再起動を越えて、以前の会話内容を返答に利用できる。
- 誤った記憶を訂正でき、次の返答に訂正内容が反映される。
- 参照した記憶と根拠の会話を開発者が確認できる。
"""

from __future__ import annotations

import json

from httpx import AsyncClient

from tests.conftest import FakeLLM


async def _say(client: AsyncClient, text: str, conversation_id: int | None = None) -> dict:
    payload: dict = {"text": text}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    response = await client.post("/api/chat", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


async def test_persona_is_applied_to_the_prompt(client: AsyncClient, fake_llm: FakeLLM):
    await _say(client, "こんにちは")
    system_prompt = fake_llm.last_system_prompt
    assert "ゆい" in system_prompt
    assert "知らないこと・覚えていないことは" in system_prompt


async def test_memory_survives_a_new_conversation(client: AsyncClient, fake_llm: FakeLLM):
    """別の会話（=再起動後に相当）でも、保存した記憶が返答に渡る。"""
    first = await _say(client, "写真を撮るのが好きなんだ")

    created = await client.post(
        "/api/memories",
        json={
            "kind": "promise",
            "content": "次は山で撮った写真の話をする約束をした",
            "certainty": "fact",
            "keywords": "写真 山 約束",
            "subject_speaker_id": first["user_message"]["speaker_id"],
            "source_message_id": first["user_message"]["id"],
            "source_conversation_id": first["conversation_id"],
        },
    )
    assert created.status_code == 201, created.text

    # conversation_id を渡さず、新しい会話として尋ねる。
    second = await _say(client, "前に何を話す約束をしたっけ？")
    assert second["conversation_id"] != first["conversation_id"]

    assert "山で撮った写真" in fake_llm.last_system_prompt
    assert [m["memory"]["id"] for m in second["used_memories"]] == [created.json()["id"]]


async def test_promise_is_retrieved_even_without_keyword_overlap(client: AsyncClient):
    first = await _say(client, "こんばんは")
    await client.post(
        "/api/memories",
        json={
            "kind": "promise",
            "content": "次回はカメラのレンズの話をすると決めた",
            "certainty": "fact",
            "keywords": "カメラ レンズ",
            "subject_speaker_id": first["user_message"]["speaker_id"],
        },
    )
    result = await _say(client, "今日はいい天気だね")
    reasons = [m["reason"] for m in result["used_memories"]]
    assert "直近の約束として常に参照" in reasons


async def test_correction_is_reflected_in_the_next_reply(
    client: AsyncClient, fake_llm: FakeLLM
):
    first = await _say(client, "好きな食べ物の話をしよう")
    created = await client.post(
        "/api/memories",
        json={
            "kind": "about_person",
            "content": "開発者はりんごが好き",
            "certainty": "fact",
            "keywords": "りんご 食べ物 好き",
            "subject_speaker_id": first["user_message"]["speaker_id"],
        },
    )
    memory_id = created.json()["id"]

    corrected = await client.patch(
        f"/api/memories/{memory_id}",
        json={
            "content": "開発者はみかんが好き（りんごは誤り）",
            "keywords": "みかん 食べ物 好き",
            "reason": "本人から訂正された",
        },
    )
    assert corrected.status_code == 200, corrected.text

    await _say(client, "好きな食べ物、覚えてる？")
    system_prompt = fake_llm.last_system_prompt
    assert "みかんが好き" in system_prompt
    assert "開発者はりんごが好き" not in system_prompt

    revisions = (await client.get(f"/api/memories/{memory_id}/revisions")).json()
    assert [r["action"] for r in revisions] == ["corrected", "created"]
    assert revisions[0]["before"]["content"] == "開発者はりんごが好き"
    assert revisions[0]["reason"] == "本人から訂正された"


async def test_correction_can_be_rolled_back(client: AsyncClient):
    await _say(client, "はじめまして")
    created = await client.post(
        "/api/memories",
        json={"kind": "experience", "content": "元の内容", "keywords": "元"},
    )
    memory_id = created.json()["id"]
    await client.patch(f"/api/memories/{memory_id}", json={"content": "変更後の内容"})

    restored = await client.post(f"/api/memories/{memory_id}/restore")
    assert restored.status_code == 200, restored.text
    assert restored.json()["content"] == "元の内容"


async def test_deleted_memory_is_not_used(client: AsyncClient, fake_llm: FakeLLM):
    first = await _say(client, "犬の話をしよう")
    created = await client.post(
        "/api/memories",
        json={
            "kind": "about_person",
            "content": "開発者は犬を飼っている",
            "keywords": "犬 ペット",
            "subject_speaker_id": first["user_message"]["speaker_id"],
        },
    )
    await client.delete(f"/api/memories/{created.json()['id']}?reason=事実誤認")

    result = await _say(client, "犬は元気？")
    assert result["used_memories"] == []
    assert "犬を飼っている" not in fake_llm.last_system_prompt


async def test_run_record_shows_the_basis_of_the_reply(client: AsyncClient):
    first = await _say(client, "山登りが趣味なんだ")
    created = await client.post(
        "/api/memories",
        json={
            "kind": "about_person",
            "content": "開発者の趣味は山登り",
            "keywords": "山登り 趣味",
            "subject_speaker_id": first["user_message"]["speaker_id"],
            "source_message_id": first["user_message"]["id"],
        },
    )
    memory_id = created.json()["id"]

    result = await _say(client, "山登りはどう？")
    run = (
        await client.get(f"/api/conversations/messages/{result['reply']['id']}/run")
    ).json()

    assert run["model"] == "fake-model"
    assert run["model_digest"] == "sha256:test"
    assert memory_id in run["referenced_memory_ids"]
    assert run["latency_ms"] is not None
    # 参照した記憶が、実際に渡したプロンプトに載っていることを確認できる。
    assert "開発者の趣味は山登り" in run["system_prompt"]

    # 記憶からは根拠の会話（どの発言から作られたか）を辿れる。
    memories = (await client.get("/api/memories")).json()
    target = next(m for m in memories if m["id"] == memory_id)
    assert target["source_message_id"] == first["user_message"]["id"]


async def test_reflection_extracts_and_accepts_candidates(
    client: AsyncClient, fake_llm: FakeLLM
):
    first = await _say(client, "写真を撮るのが好き。次は山で撮った写真の話をしよう")

    fake_llm.push(
        json.dumps(
            [
                {
                    "kind": "promise",
                    "content": "次は山で撮った写真の話をする",
                    "certainty": "fact",
                    "keywords": "写真 山 約束",
                    "about_partner": True,
                },
                {
                    "kind": "about_person",
                    "content": "開発者は写真を撮るのが好き",
                    "certainty": "fact",
                    "keywords": "写真 趣味",
                    "about_partner": True,
                },
            ],
            ensure_ascii=False,
        )
    )
    ended = await client.post(f"/api/conversations/{first['conversation_id']}/end")
    assert ended.status_code == 200, ended.text
    candidates = ended.json()
    assert len(candidates) == 2
    assert all(c["status"] == "pending" for c in candidates)

    accepted = await client.post(
        f"/api/conversations/candidates/{candidates[0]['id']}/decide",
        json={"decision": "accept"},
    )
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["status"] == "accepted"

    rejected = await client.post(
        f"/api/conversations/candidates/{candidates[1]['id']}/decide",
        json={"decision": "reject"},
    )
    assert rejected.json()["status"] == "rejected"

    memories = (await client.get("/api/memories")).json()
    assert [m["content"] for m in memories] == ["次は山で撮った写真の話をする"]


async def test_ended_conversation_rejects_new_messages(client: AsyncClient, fake_llm: FakeLLM):
    first = await _say(client, "そろそろ終わろうか")
    fake_llm.push("[]")
    await client.post(f"/api/conversations/{first['conversation_id']}/end")

    response = await client.post(
        "/api/chat", json={"text": "やっぱり続き", "conversation_id": first["conversation_id"]}
    )
    assert response.status_code == 409


async def test_ideal_response_is_recorded(client: AsyncClient):
    result = await _say(client, "ねえ、なにか話して")
    created = await client.post(
        f"/api/conversations/messages/{result['reply']['id']}/ideal",
        json={"ideal_text": "もっと短く、質問を添えて返してほしい", "note": "冗長だった"},
    )
    assert created.status_code == 200, created.text
    listed = (
        await client.get(f"/api/conversations/messages/{result['reply']['id']}/ideal")
    ).json()
    assert listed[0]["ideal_text"] == "もっと短く、質問を添えて返してほしい"
