"""記憶の参照範囲（ISSUE-010 / 設計書フェーズ3の3C）。

複数の相手が入る前に、「誰との会話で参照してよいか」を必ず決めた状態にする。
既定で「誰にでも渡す」にしていると、ある相手との会話から作られた記憶が
別の相手へ黙って渡る。
"""

from __future__ import annotations

from httpx import AsyncClient

from app.models import Visibility
from tests.conftest import FakeLLM


async def _add(client: AsyncClient, **payload) -> dict:
    body = {"kind": "experience", "content": "何かの記憶", "keywords": "何か"}
    body.update(payload)
    return (await client.post("/api/memories", json=body)).json()


async def test_scope_must_be_stated_when_adding(client: AsyncClient) -> None:
    """参照範囲を書かない追加は受け付けない。既定で全員に渡さない。"""
    response = await client.post(
        "/api/memories", json={"kind": "experience", "content": "範囲の無い記憶"}
    )
    assert response.status_code == 422
    assert "参照範囲" in response.text


async def test_scope_cannot_be_both(client: AsyncClient) -> None:
    speakers = (await client.get("/api/speakers")).json()
    speaker_id = speakers[0]["id"] if speakers else 1
    response = await client.post(
        "/api/memories",
        json={
            "kind": "experience",
            "content": "両方指定した記憶",
            "visible_to_speaker_id": speaker_id,
            "visible_to_all": True,
        },
    )
    assert response.status_code == 422


async def test_scope_is_kept_as_chosen(client: AsyncClient) -> None:
    """限定しないことも選べる。選んだ結果としてそうなる。"""
    turn = await client.post("/api/chat", json={"text": "こんにちは"})
    speaker_id = turn.json()["user_message"]["speaker_id"]

    limited = await _add(client, content="この相手に限る", visible_to_speaker_id=speaker_id)
    assert limited["visible_to_speaker_id"] == speaker_id

    shared = await _add(client, content="限定しない", visible_to_all=True)
    assert shared["visible_to_speaker_id"] is None


async def test_unscoped_memories_can_be_listed(client: AsyncClient) -> None:
    """振り分けのために、範囲を限定していない非公開の記憶を洗い出せる。"""
    turn = await client.post("/api/chat", json={"text": "こんにちは"})
    speaker_id = turn.json()["user_message"]["speaker_id"]

    unscoped = await _add(client, content="限定していない記憶", visible_to_all=True)
    scoped = await _add(client, content="限定した記憶", visible_to_speaker_id=speaker_id)
    public = await _add(
        client,
        content="配信で使ってよい記憶",
        visible_to_all=True,
        visibility=Visibility.PUBLIC.value,
    )

    listed = (await client.get("/api/memories?unscoped=true")).json()
    ids = [m["id"] for m in listed]
    assert unscoped["id"] in ids
    assert scoped["id"] not in ids
    # 公開可能な記憶は、限定しないことが前提なので洗い出しに含めない。
    assert public["id"] not in ids


async def test_scope_can_be_assigned_afterwards(client: AsyncClient) -> None:
    """洗い出した記憶を、後から相手へ振り分けられる。履歴にも残る。"""
    turn = await client.post("/api/chat", json={"text": "こんにちは"})
    speaker_id = turn.json()["user_message"]["speaker_id"]
    memory = await _add(client, content="あとで振り分ける記憶", visible_to_all=True)

    assigned = await client.patch(
        f"/api/memories/{memory['id']}",
        json={"visible_to_speaker_id": speaker_id, "reason": "相手を限定した"},
    )
    assert assigned.status_code == 200
    assert assigned.json()["visible_to_speaker_id"] == speaker_id
    assert memory["id"] not in [
        m["id"] for m in (await client.get("/api/memories?unscoped=true")).json()
    ]

    revisions = (await client.get(f"/api/memories/{memory['id']}/revisions")).json()
    assert revisions[0]["before"]["visible_to_speaker_id"] is None
    assert revisions[0]["after"]["visible_to_speaker_id"] == speaker_id

    # 限定しない状態へ戻すこともできる。
    back = await client.patch(
        f"/api/memories/{memory['id']}", json={"visible_to_all": True, "reason": "戻した"}
    )
    assert back.json()["visible_to_speaker_id"] is None


async def test_speaker_labels_appear_for_multiple_partners(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """相手が複数いる会話でだけ、発言に誰のものかを付ける（ISSUE-020）。

    付けないと履歴の発言がすべて同じ「user」に見え、直前の相手の発言を
    目の前の相手のものとして扱う。ひとりとの会話では付けない。
    """
    a = {"source": "local_text", "external_id": "label-a", "display_name": "Aさん"}
    b = {"source": "local_text", "external_id": "label-b", "display_name": "Bさん"}

    first = await client.post("/api/chat", json={"text": "私は登山が好き", "speaker": a})
    conversation_id = first.json()["conversation_id"]
    # ひとりとの会話では名前を付けない。
    assert fake_llm.calls[-1][-1].content == "私は登山が好き"

    await client.post(
        "/api/chat",
        json={"text": "私は登山が苦手", "conversation_id": conversation_id, "speaker": b},
    )
    sent = [m.content for m in fake_llm.calls[-1]]
    assert "Aさん：私は登山が好き" in sent
    assert sent[-1] == "Bさん：私は登山が苦手"
