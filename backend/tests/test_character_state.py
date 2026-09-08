"""変化する状態（ISSUE-015・016 / 設計書フェーズ3の3B）。

固定人格と分ける。人格は版として管理し、日常の更新で上書きしない。
関心は YUI 自身のもの、関係性は相手ごとのもので、どちらも根拠を持つ。
"""

from __future__ import annotations

from httpx import AsyncClient

from tests.conftest import FakeLLM


async def _speaker(client: AsyncClient, text: str = "こんにちは", **speaker) -> int:
    body = {"text": text}
    if speaker:
        body["speaker"] = speaker
    turn = await client.post("/api/chat", json=body)
    return turn.json()["user_message"]["speaker_id"]


async def _add(client: AsyncClient, **payload) -> dict:
    body = {
        "kind": "interest",
        "content": "雨の日の静かな時間が好き",
        "topic": "天気",
        # 参照範囲は必ず指定する（記憶と同じ。ISSUE-010 と同じ考え方）。
        "visible_to_all": True,
    }
    body.update(payload)
    response = await client.post("/api/states", json=body)
    assert response.status_code == 201, response.text
    return response.json()


async def test_new_state_is_a_candidate_until_accepted(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """採用するまで会話に使わない。開発者が確認してから使う。"""
    state = await _add(client)
    assert state["status"] == "pending"

    await client.post("/api/chat", json={"text": "雨の日はどう？"})
    assert "いまの自分" not in fake_llm.last_system_prompt

    await client.post(f"/api/states/{state['id']}/decide", json={"decision": "accept"})
    await client.post("/api/chat", json={"text": "雨の日はどう？"})
    assert "雨の日の静かな時間が好き" in fake_llm.last_system_prompt


async def test_state_is_separate_from_the_persona(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """人格とは別の見出しで渡す。いま思っていることを人格の一部にしない。"""
    state = await _add(client)
    await client.post(f"/api/states/{state['id']}/decide", json={"decision": "accept"})
    await client.post("/api/chat", json={"text": "こんにちは"})

    prompt = fake_llm.last_system_prompt
    assert prompt.index("# 性格") < prompt.index("# いまの自分")
    assert "# 関心" not in prompt  # 人格側の見出しに混ざっていない


async def test_relationship_needs_a_partner(client: AsyncClient) -> None:
    bad = await client.post(
        "/api/states", json={"kind": "relationship", "content": "距離が縮まった"}
    )
    assert bad.status_code == 422

    bad_interest = await client.post(
        "/api/states",
        json={"kind": "interest", "content": "写真に興味がある", "subject_speaker_id": 1},
    )
    assert bad_interest.status_code == 422


async def test_relationship_is_only_used_with_that_partner(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """別の相手との関係を持ち出さない（3C）。"""
    a_id = await _speaker(client, "こんにちは", source="local_text", external_id="st-a",
                          display_name="Aさん")
    state = await _add(
        client,
        kind="relationship",
        topic=None,
        content="Aさんとは写真の話でよく盛り上がる",
        subject_speaker_id=a_id,
        visible_to_all=False,
        visible_to_speaker_id=a_id,
    )
    await client.post(f"/api/states/{state['id']}/decide", json={"decision": "accept"})

    await client.post(
        "/api/chat",
        json={
            "text": "こんにちは",
            "speaker": {"source": "local_text", "external_id": "st-b", "display_name": "Bさん"},
        },
    )
    assert "Aさんとは写真の話" not in fake_llm.last_system_prompt

    await client.post(
        "/api/chat",
        json={
            "text": "こんにちは",
            "speaker": {"source": "local_text", "external_id": "st-a", "display_name": "Aさん"},
        },
    )
    assert "Aさんとは写真の話" in fake_llm.last_system_prompt


async def test_used_states_are_recorded(client: AsyncClient) -> None:
    """完了条件「参照した記憶・人格版・可変状態を追跡できる」。"""
    state = await _add(client)
    await client.post(f"/api/states/{state['id']}/decide", json={"decision": "accept"})
    reply = await client.post("/api/chat", json={"text": "こんにちは"})
    assert reply.json()["run"]["referenced_state_ids"] == [state["id"]]


async def test_correcting_the_basis_memory_marks_the_state(client: AsyncClient) -> None:
    """根拠の記憶を訂正すると、そこから作った状態に再評価の印が付く（ISSUE-016）。"""
    speaker_id = await _speaker(client)
    memory = (
        await client.post(
            "/api/memories",
            json={
                "kind": "experience",
                "content": "開発者と雨の日の話をした",
                "keywords": "雨 天気",
                "visible_to_speaker_id": speaker_id,
            },
        )
    ).json()
    state = await _add(
        client,
        basis_memory_ids=[memory["id"]],
        visible_to_all=False,
        visible_to_speaker_id=speaker_id,
    )
    await client.post(f"/api/states/{state['id']}/decide", json={"decision": "accept"})

    await client.patch(
        f"/api/memories/{memory['id']}",
        json={"content": "開発者と雪の日の話をした", "reason": "聞き違いだった"},
    )

    updated = (await client.get("/api/states?needs_review=true")).json()
    assert [s["id"] for s in updated] == [state["id"]]
    assert "訂正された" in updated[0]["review_reason"]


async def test_state_under_review_is_not_used(client: AsyncClient, fake_llm: FakeLLM) -> None:
    """印が付いている間は会話に渡さない。訂正が反映されないまま話さないため。"""
    speaker_id = await _speaker(client)
    memory = (
        await client.post(
            "/api/memories",
            json={
                "kind": "experience",
                "content": "開発者と雨の日の話をした",
                "keywords": "雨",
                "visible_to_speaker_id": speaker_id,
            },
        )
    ).json()
    state = await _add(
        client,
        basis_memory_ids=[memory["id"]],
        visible_to_all=False,
        visible_to_speaker_id=speaker_id,
    )
    await client.post(f"/api/states/{state['id']}/decide", json={"decision": "accept"})
    await client.delete(f"/api/memories/{memory['id']}?reason=誤りのため")

    await client.post("/api/chat", json={"text": "雨の日はどう？"})
    assert "雨の日の静かな時間が好き" not in fake_llm.last_system_prompt

    # 確認して印を下ろすと、また使われる。
    await client.patch(
        f"/api/states/{state['id']}", json={"reviewed": True, "reason": "根拠を確認した"}
    )
    await client.post("/api/chat", json={"text": "雨の日はどう？"})
    assert "雨の日の静かな時間が好き" in fake_llm.last_system_prompt


async def test_changes_are_kept_in_the_history(client: AsyncClient) -> None:
    state = await _add(client)
    await client.post(
        f"/api/states/{state['id']}/decide",
        json={"decision": "accept", "reason": "会話で確かめた"},
    )
    await client.patch(
        f"/api/states/{state['id']}",
        json={"content": "雨の日の静かな時間が好き（音も含めて）", "reason": "言い方を直した"},
    )
    revisions = (await client.get(f"/api/states/{state['id']}/revisions")).json()
    assert [r["action"] for r in revisions] == ["corrected", "accepted", "created"]
    assert revisions[1]["reason"] == "会話で確かめた"
    assert revisions[0]["before"]["content"] == "雨の日の静かな時間が好き"


async def test_reflection_creates_state_candidates(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """会話の振り返りから、関心・関係性の候補が作られる（ISSUE-015）。"""
    first = await client.post("/api/chat", json={"text": "山で撮った写真を見せたいな"})
    conversation_id = first.json()["conversation_id"]

    fake_llm.push("[]")  # 記憶の候補は無し
    fake_llm.push_state(
        '[{"kind":"interest","topic":"写真","content":"人が撮った写真から、'
        'その人が何を見ていたかを知りたい","reason":"写真の話で強く反応した"},'
        '{"kind":"relationship","content":"開発者とは写真の話でよく盛り上がる",'
        '"reason":"同じ話題が続いている"}]'
    )
    ended = await client.post(f"/api/conversations/{conversation_id}/end")
    assert ended.status_code == 200

    states = (await client.get("/api/states?state_status=pending")).json()
    kinds = {s["kind"] for s in states}
    assert kinds == {"interest", "relationship"}
    interest = next(s for s in states if s["kind"] == "interest")
    relationship = next(s for s in states if s["kind"] == "relationship")
    # 関心は YUI 自身のもの。相手には紐づけない。
    assert interest["subject_speaker_id"] is None
    assert interest["topic"] == "写真"
    # 関係性は、その相手のもの。
    assert relationship["subject_speaker_id"] is not None
    # どちらも、その会話から作られたことが分かる。
    assert relationship["source_conversation_id"] == conversation_id


async def test_state_extraction_failure_can_be_retried(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """読み取れない出力で、静かに「更新なし」にしない。やり直せる。"""
    first = await client.post("/api/chat", json={"text": "写真の話をしよう"})
    conversation_id = first.json()["conversation_id"]

    fake_llm.push("[]")
    fake_llm.push_state("関心はありません")
    failed = await client.post(f"/api/conversations/{conversation_id}/end")
    assert failed.status_code == 503
    assert "状態の更新候補" in failed.json()["detail"]

    fake_llm.push("[]")
    fake_llm.push_state("[]")
    assert (await client.post(f"/api/conversations/{conversation_id}/end")).status_code == 200


async def test_no_state_candidates_when_the_partner_is_unclear(
    fake_llm: FakeLLM
) -> None:
    """相手を決められない会話では、状態の候補を作らない。

    非公開の会話から作った状態にも参照範囲を引き継ぐ必要があり、誰に限るかを
    決められないまま作ると、別の相手へ渡る（全体レビューの指摘3）。
    """
    from app.agent.state_reflection import propose_state_candidates  # noqa: PLC0415

    fake_llm.push_state('[{"kind":"relationship","content":"距離が縮まった"}]')
    created = await propose_state_candidates(
        llm=fake_llm,
        transcript="[#1] Aさん: こんにちは\n[#2] Bさん: こんにちは",
        partner_speaker_id=None,
        current_states=[],
    )
    assert created == []
    # モデルも呼ばない。決められないと分かっている場合に問い合わせない。
    assert not any("いまの状態" in call[-1].content for call in fake_llm.calls)
