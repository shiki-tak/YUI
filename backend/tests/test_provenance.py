"""発言の種別（ISSUE-017 / v0.1）。

「AさんがBさんの好みを話した」を、Bさん本人の発言と区別する。分類そのものは
モデルに任せるが、**分類を信じて相手に紐づけることはしない**。伝聞を相手の
情報として保存すると、次の会話でその相手の好みとして使われる。
"""

from __future__ import annotations

from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models import MemoryCandidate, Provenance
from tests.conftest import FakeLLM


async def _talk_and_reflect(client: AsyncClient, fake_llm: FakeLLM, text: str, output: str):
    first = await client.post("/api/chat", json={"text": text})
    conversation_id = first.json()["conversation_id"]
    fake_llm.push(output)
    ended = await client.post(f"/api/conversations/{conversation_id}/end")
    assert ended.status_code == 200, ended.text
    return ended.json()


async def test_hearsay_is_not_attached_to_the_partner(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """第三者についての内容は、話している相手に紐づけない。

    相手の好みとして残ると、次の会話でその相手の好みとして使われる。

    **入手経路は about_partner から導出する**（ISSUE-023、2026-09-09）。以前は
    provenance と about_partner を別々に訊いて、食い違ったら伝聞側へ倒していた。
    どちらの指示文も「その内容が相手自身についてのものか」を訊いており、同じ軸を
    2回訊いていた。実測で食い違い、**本人の予定が伝聞になった**
    （`plan-is-kept` が 0/4）。訊くのは1つにした。
    """
    candidates = await _talk_and_reflect(
        client,
        fake_llm,
        "弟のBは辛いものが苦手なんだ",
        '[{"kind":"about_person","content":"開発者の弟は辛いものが苦手","certainty":"fact",'
        '"keywords":"弟 辛い 苦手","about_partner":false}]',
    )
    assert candidates[0]["provenance"] == Provenance.HEARSAY.value
    # 相手についての情報にしない。
    assert candidates[0]["subject_speaker_id"] is None


async def test_firsthand_is_attached_to_the_partner(client: AsyncClient, fake_llm: FakeLLM) -> None:
    candidates = await _talk_and_reflect(
        client,
        fake_llm,
        "私は辛いものが好きなんだ",
        '[{"kind":"about_person","content":"開発者は辛いものが好き","certainty":"fact",'
        '"keywords":"辛い 好き","about_partner":true}]',
    )
    assert candidates[0]["provenance"] == Provenance.FIRSTHAND.value
    assert candidates[0]["subject_speaker_id"] is not None


async def test_provenance_is_derived_not_asked(client: AsyncClient, fake_llm: FakeLLM) -> None:
    """入手経路はモデルへ訊かない。about_partner から決める（ISSUE-023）。

    モデルが provenance を書いてきても無視する。**同じ軸を2回訊くと食い違い、
    食い違えば必ずどちらかが誤りになる。** 導出できるものは訊かない
    導出できるものを訊かない。

    以前は「読み取れない入手経路は既定へ寄せずに失敗させる」としていたが、
    読み取る対象そのものが無くなったので、その検査も無くなった。
    """
    candidates = await _talk_and_reflect(
        client,
        fake_llm,
        "私は毎朝走っているんだ",
        '[{"kind":"about_person","content":"開発者は毎朝走っている","certainty":"fact",'
        # モデルが誤った入手経路を書いてきても、about_partner が優先される。
        '"provenance":"聞いた話","keywords":"走る 毎朝","about_partner":true}]',
    )
    assert candidates[0]["provenance"] == Provenance.FIRSTHAND.value
    assert candidates[0]["subject_speaker_id"] is not None


async def test_accepted_candidate_keeps_its_provenance(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    candidates = await _talk_and_reflect(
        client,
        fake_llm,
        "友人のCは猫を飼っているらしい",
        '[{"kind":"about_person","content":"開発者の友人Cは猫を飼っている","certainty":"fact",'
        '"keywords":"友人 猫","about_partner":false}]',
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


async def test_an_impression_is_firsthand_even_though_it_is_not_about_the_partner(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """キャラクター自身の受け止め方は、本人の発言として残す（ISSUE-023）。

    impression は YUI 自身の受け止め方なので、相手についての内容ではない
    （about_partner は false）。入手経路を about_partner だけから導出すると
    伝聞へ倒れるが、**自分が感じたことを人づてに聞いたことにはならない。**

    実測（2026-09-09）では、自動採用された impression の 4/8 がこれで誤って
    いた（「YUI は開発者の睡眠不足を心配している」が伝聞）。
    """
    candidates = await _talk_and_reflect(
        client,
        fake_llm,
        "今週は仕事が立て込んでいて、あまり眠れていないんだ",
        '[{"kind":"impression","content":"YUI は開発者の睡眠不足を心配している",'
        '"certainty":"inference","keywords":"心配 睡眠","about_partner":false}]',
    )
    assert candidates[0]["provenance"] == Provenance.FIRSTHAND.value
    # 相手についての情報ではないので、相手には紐づけない。
    assert candidates[0]["subject_speaker_id"] is None


async def test_a_missing_about_partner_is_unknown_not_hearsay(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """判断材料が返っていないなら、伝聞と確定しない。

    入手経路を about_partner から導出するようにしたとき、**省略を既定の false
    として扱っていた**。「第三者についてと判断した false」と区別が付かず、
    モデルが何も答えていないのに伝聞として長期記憶へ入る。

    以前 provenance を訊いていた頃は、省略を unknown にできていた。導出元を
    変えたときに、その表現力を落としていた。
    """
    candidates = await _talk_and_reflect(
        client,
        fake_llm,
        "次は写真の話をしよう",
        # about_partner が無い出力。
        '[{"kind":"promise","content":"次は写真の話をする","certainty":"fact"}]',
    )
    assert candidates[0]["provenance"] == Provenance.UNKNOWN.value
    assert candidates[0]["subject_speaker_id"] is None
