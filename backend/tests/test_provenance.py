"""発言の種別（ISSUE-017 / 設計書フェーズ3の3C）。

「AさんがBさんの好みを話した」を、Bさん本人の発言と区別する。分類そのものは
モデルに任せるが、**分類を信じて相手に紐づけることはしない**。伝聞を相手の
情報として保存すると、次の会話でその相手の好みとして使われる。
"""

from __future__ import annotations

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models import MemoryCandidate, Provenance
from tests.conftest import FakeLLM, end_and_wait


async def _talk_and_reflect(client: AsyncClient, fake_llm: FakeLLM, text: str, output: str):
    first = await client.post("/api/chat", json={"text": text})
    conversation_id = first.json()["conversation_id"]
    fake_llm.push(output)
    ended = await end_and_wait(client, conversation_id)
    assert ended.status_code == 202, ended.text
    return (await client.get(f"/api/conversations/{conversation_id}/candidates")).json()


async def test_hearsay_is_not_attached_to_the_partner(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """伝聞は、話している相手についての情報として保存しない。

    モデルが about_partner を true にしても、伝聞なら採らない。相手の好みとして
    残ると、次の会話でその相手の好みとして使われる。
    """
    candidates = await _talk_and_reflect(
        client,
        fake_llm,
        "弟のBは辛いものが苦手なんだ",
        '[{"kind":"about_person","content":"開発者の弟は辛いものが苦手","certainty":"fact",'
        '"provenance":"hearsay","keywords":"弟 辛い 苦手","about_partner":true}]',
    )
    assert candidates[0]["provenance"] == Provenance.HEARSAY.value
    # 相手についての情報にしない。
    assert candidates[0]["subject_speaker_id"] is None


async def test_firsthand_is_attached_to_the_partner(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    candidates = await _talk_and_reflect(
        client,
        fake_llm,
        "私は辛いものが好きなんだ",
        '[{"kind":"about_person","content":"開発者は辛いものが好き","certainty":"fact",'
        '"provenance":"firsthand","keywords":"辛い 好き","about_partner":true}]',
    )
    assert candidates[0]["provenance"] == Provenance.FIRSTHAND.value
    assert candidates[0]["subject_speaker_id"] is not None


async def test_unknown_provenance_is_allowed_and_not_guessed(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """決められないものは unknown のまま。どちらかへ寄せない。"""
    candidates = await _talk_and_reflect(
        client,
        fake_llm,
        "その話、誰かから聞いた気がする",
        '[{"kind":"experience","content":"どこかで聞いた話がある","certainty":"inference",'
        '"provenance":"unknown","keywords":"話 記憶","about_partner":false}]',
    )
    assert candidates[0]["provenance"] == Provenance.UNKNOWN.value


async def test_invalid_provenance_fails_the_whole_reflection(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """読み取れない入手経路は、既定へ寄せずに失敗させる。"""
    first = await client.post("/api/chat", json={"text": "入手経路が不正な出力"})
    conversation_id = first.json()["conversation_id"]
    fake_llm.push(
        '[{"kind":"experience","content":"何か","provenance":"聞いた話"}]'
    )
    # 抽出はジョブの中で失敗する。終了そのものは受け付けたうえで、失敗の理由を
    # 進行状態から見る（フェーズ4 PR5）。
    accepted = await end_and_wait(client, conversation_id)
    assert accepted.status_code == 202
    progress = (await client.get(f"/api/conversations/{conversation_id}/reflection")).json()
    assert progress["state"] == "failed"
    assert "入手経路が不正" in progress["error"]


async def test_accepted_candidate_keeps_its_provenance(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    candidates = await _talk_and_reflect(
        client,
        fake_llm,
        "友人のCは猫を飼っているらしい",
        '[{"kind":"about_person","content":"開発者の友人Cは猫を飼っている","certainty":"fact",'
        '"provenance":"hearsay","keywords":"友人 猫","about_partner":false}]',
    )
    accepted = await client.post(
        f"/api/conversations/candidates/{candidates[0]['id']}/decide",
        json={"decision": "accept"},
    )
    memory_id = accepted.json()["accepted_memory_id"]
    memory = (await client.get(f"/api/memories/{memory_id}")).json()
    assert memory["provenance"] == Provenance.HEARSAY.value

    async with session_factory() as session:
        stored = (await session.execute(select(MemoryCandidate))).scalars().all()
        assert {c.provenance for c in stored} == {Provenance.HEARSAY.value}


async def test_hearsay_is_marked_when_given_to_the_model(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """伝聞として渡す。本人から聞いたことと区別できないと、相手の発言として扱う。"""
    turn = await client.post("/api/chat", json={"text": "こんにちは"})
    speaker_id = turn.json()["user_message"]["speaker_id"]
    await client.post(
        "/api/memories",
        json={
            "kind": "about_person",
            "content": "開発者の弟は辛いものが苦手",
            "keywords": "弟 辛い 苦手",
            "provenance": "hearsay",
            "visible_to_speaker_id": speaker_id,
        },
    )
    await client.post("/api/chat", json={"text": "弟の苦手なものって何だっけ？"})
    assert "人づてに聞いた" in fake_llm.last_system_prompt


async def test_attribution_can_be_corrected_when_accepting(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """モデルが誤って相手に紐づけても、採用の時点で人が直せる。

    実モデルは「弟は辛いものが苦手」を、指示を精密にする前は5回中5回
    firsthand・about_partner=true と分類した。分類を信じきらず、直せる経路を残す。
    """
    candidates = await _talk_and_reflect(
        client,
        fake_llm,
        "弟は辛いものが苦手なんだ",
        '[{"kind":"about_person","content":"開発者の弟は辛いものが苦手","certainty":"fact",'
        '"provenance":"firsthand","keywords":"弟 辛い","about_partner":true}]',
    )
    # モデルの分類のまま候補になっている。
    assert candidates[0]["subject_speaker_id"] is not None
    assert candidates[0]["provenance"] == Provenance.FIRSTHAND.value

    accepted = await client.post(
        f"/api/conversations/candidates/{candidates[0]['id']}/decide",
        json={
            "decision": "accept",
            "subject_to_none": True,
            "provenance": "hearsay",
            "reason": "弟についての伝聞なので、開発者本人の情報にしない",
        },
    )
    memory = (await client.get(f"/api/memories/{accepted.json()['accepted_memory_id']}")).json()
    assert memory["subject_speaker_id"] is None
    assert memory["provenance"] == Provenance.HEARSAY.value
