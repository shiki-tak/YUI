"""同じ出来事を二重に覚えない（ISSUE-018 / 設計書フェーズ3の3C）。

自動で捨てない。似ている記憶を控えて、採用を判断するときに見せる。
反復を独立した経験として数えないための手がかりにする。
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.agent.memory_store import _SIMILAR_THRESHOLD, similarity, tokenize
from tests.conftest import FakeLLM


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ("開発者は写真を撮るのが好き 写真 趣味", "開発者は写真を撮るのが好き 写真 趣味", True),
        (
            "開発者が高尾山に登った 高尾山 登山",
            "開発者は先週 高尾山に登ったと話した 高尾山 登山",
            True,
        ),
        ("開発者は写真を撮るのが好き 写真 趣味", "開発者の趣味は写真撮影 写真 趣味 撮影", True),
        # 訂正も「近い」として出す。重複ではないが、開発者に見せる価値がある。
        ("開発者はコーヒーが好き コーヒー 飲み物", "開発者は紅茶が好き 紅茶 飲み物", True),
        # 実モデルが2回目に出した言い方。長くなっても同じ話だと分かる。
        (
            "開発者は写真を撮ることが好きで、休日はカメラを持って歩く 写真 カメラ",
            "開発者は写真を撮るのが好き 写真 趣味 撮影",
            True,
        ),
        ("開発者は写真を撮るのが好き 写真 趣味", "開発者は写真展に行った 写真展 外出", False),
        ("開発者は写真を撮るのが好き 写真 趣味", "開発者の弟は辛いものが苦手 弟 辛い", False),
    ],
)
def test_similarity_separates_duplicates_from_other_topics(
    left: str, right: str, expected: bool
) -> None:
    assert (similarity(tokenize(left), tokenize(right)) >= _SIMILAR_THRESHOLD) is expected


async def _reflect(client: AsyncClient, fake_llm: FakeLLM, text: str, output: str) -> list[dict]:
    first = await client.post("/api/chat", json={"text": text})
    fake_llm.push(output)
    ended = await client.post(f"/api/conversations/{first.json()['conversation_id']}/end")
    assert ended.status_code == 200, ended.text
    return ended.json()


async def test_candidate_points_at_the_memory_it_repeats(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """同じ内容を2回目に話しても、捨てずに「近い記憶」として控える。"""
    output = (
        '[{"kind":"about_person","content":"開発者は写真を撮るのが好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"写真 趣味","about_partner":true}]'
    )
    first = await _reflect(client, fake_llm, "写真を撮るのが好きなんだ", output)
    assert first[0]["similar_memory_ids"] is None  # まだ記憶が無い

    accepted = await client.post(
        f"/api/conversations/candidates/{first[0]['id']}/decide", json={"decision": "accept"}
    )
    memory_id = accepted.json()["accepted_memory_id"]

    second = await _reflect(client, fake_llm, "写真を撮るのが好きでね", output)
    # 捨てずに候補として出す。近い記憶を控えている。
    assert second[0]["similar_memory_ids"] == [memory_id]


async def test_unrelated_candidate_is_not_marked(client: AsyncClient, fake_llm: FakeLLM) -> None:
    await _reflect(
        client,
        fake_llm,
        "写真を撮るのが好きなんだ",
        '[{"kind":"about_person","content":"開発者は写真を撮るのが好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"写真 趣味","about_partner":true}]',
    )
    candidates = (await client.get("/api/conversations/candidates/pending")).json()
    await client.post(
        f"/api/conversations/candidates/{candidates[0]['id']}/decide", json={"decision": "accept"}
    )

    other = await _reflect(
        client,
        fake_llm,
        "弟は辛いものが苦手なんだ",
        '[{"kind":"about_person","content":"開発者の弟は辛いものが苦手","certainty":"fact",'
        '"provenance":"hearsay","keywords":"弟 辛い","about_partner":false}]',
    )
    assert other[0]["similar_memory_ids"] is None


async def test_similar_memories_respect_the_reference_scope(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """別の相手にだけ見せる記憶は、近い記憶としても出さない。

    候補の画面に出すだけでも、別の相手の記憶を持ち出すことになる。
    """
    a = {"source": "local_text", "external_id": "dup-a", "display_name": "Aさん"}
    b = {"source": "local_text", "external_id": "dup-b", "display_name": "Bさん"}

    turn = await client.post("/api/chat", json={"text": "こんにちは", "speaker": a})
    a_id = turn.json()["user_message"]["speaker_id"]
    await client.post(
        "/api/memories",
        json={
            "kind": "about_person",
            "content": "Aさんは写真を撮るのが好き",
            "keywords": "写真 趣味",
            "visible_to_speaker_id": a_id,
        },
    )

    first = await client.post("/api/chat", json={"text": "写真が好きでね", "speaker": b})
    fake_llm.push(
        '[{"kind":"about_person","content":"Bさんは写真を撮るのが好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"写真 趣味","about_partner":true}]'
    )
    ended = await client.post(f"/api/conversations/{first.json()['conversation_id']}/end")
    assert ended.json()[0]["similar_memory_ids"] is None
