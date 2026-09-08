"""出来事の日時（ISSUE-004 / 設計書フェーズ3の3B）。

いつの話かを残す。訂正した好みと、過去の経験を区別するために要る。
読み取れない日付は「日付なし」として扱う。間違った日付は、日付が無いことより悪い。
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from httpx import AsyncClient

from app.agent.reflection import _parse_occurred_on
from app.config import LOCAL_TZ
from tests.conftest import FakeLLM

TODAY = date(2026, 9, 7)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2026-09-01", date(2026, 9, 1)),
        ("  2026-08-15  ", date(2026, 8, 15)),
        (TODAY.isoformat(), TODAY),
        # 読み取れないものは日付なし。それらしい日付へ寄せない。
        ("先週", None),
        ("2026-13-01", None),
        ("", None),
        (None, None),
        # 未来の出来事は書かない。
        ("2026-09-08", None),
    ],
)
def test_only_readable_past_dates_are_kept(value: str | None, expected: date | None) -> None:
    parsed = _parse_occurred_on(value, today=TODAY)
    # 保存は UTC。日付として合っているかは、地域時刻に直して見る。
    assert (parsed.astimezone(LOCAL_TZ).date() if parsed else None) == expected


def test_date_is_stored_as_the_start_of_the_day_here() -> None:
    """地域時刻のその日の始まりとして保存する。表示で日付がずれないため。"""
    parsed = _parse_occurred_on("2026-09-01", today=TODAY)
    assert parsed is not None
    assert parsed.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M") == "2026-09-01 00:00"
    # 保存は UTC。前日の15時になる。SQLite が timezone を落とすため、
    # UTC で保存しないと読み戻したときに9時間ずれる（ISSUE-003 と同じ）。
    assert parsed.strftime("%Y-%m-%d %H:%M") == "2026-08-31 15:00"
    assert parsed.tzinfo == UTC


async def _reflect(client: AsyncClient, fake_llm: FakeLLM, text: str, output: str) -> list[dict]:
    first = await client.post("/api/chat", json={"text": text})
    fake_llm.push(output)
    ended = await client.post(f"/api/conversations/{first.json()['conversation_id']}/end")
    assert ended.status_code == 200, ended.text
    return ended.json()


async def test_today_is_given_to_the_model(client: AsyncClient, fake_llm: FakeLLM) -> None:
    """「先週」を日付へ直すには、今日が何日かが要る。"""
    await _reflect(client, fake_llm, "先週の話なんだけど", "[]")
    # 振り返りは2回呼ぶ（記憶の抽出と、関心・関係性の抽出）。日付を渡すのは
    # 記憶の抽出のほう。
    memory_call = next(
        call for call in fake_llm.calls if "occurred_on" in call[0].content
    )
    sent = memory_call[-1].content
    assert "振り返りを行っている日:" in sent
    assert datetime.now(LOCAL_TZ).date().isoformat() in sent
    # 相対的な日付の基準は、発言の日付。会話ログにも日付が入っている。
    assert "／" in sent and "] 開発者:" in sent


async def test_candidate_keeps_the_event_date(client: AsyncClient, fake_llm: FakeLLM) -> None:
    candidates = await _reflect(
        client,
        fake_llm,
        "先週、実家の犬が18歳になったんだ",
        '[{"kind":"about_person","content":"開発者の実家の犬が18歳になった",'
        '"certainty":"fact","provenance":"firsthand","occurred_on":"2026-09-01",'
        '"keywords":"犬 実家 誕生日","about_partner":true}]',
    )
    assert candidates[0]["occurred_at"].startswith("2026-08-31T15:00")


async def test_accepted_memory_keeps_the_event_date(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    candidates = await _reflect(
        client,
        fake_llm,
        "先週、実家の犬が18歳になったんだ",
        '[{"kind":"about_person","content":"開発者の実家の犬が18歳になった",'
        '"certainty":"fact","provenance":"firsthand","occurred_on":"2026-09-01",'
        '"keywords":"犬 実家 誕生日","about_partner":true}]',
    )
    accepted = await client.post(
        f"/api/conversations/candidates/{candidates[0]['id']}/decide",
        json={"decision": "accept"},
    )
    memory = (await client.get(f"/api/memories/{accepted.json()['accepted_memory_id']}")).json()
    assert memory["occurred_at"].startswith("2026-08-31T15:00")


async def test_undated_content_stays_undated(client: AsyncClient, fake_llm: FakeLLM) -> None:
    """好みのように、いつのことか決まらない内容は日付を作らない。"""
    candidates = await _reflect(
        client,
        fake_llm,
        "私、写真を撮るのが好きなんだ",
        '[{"kind":"about_person","content":"開発者は写真を撮るのが好き","certainty":"fact",'
        '"provenance":"firsthand","occurred_on":null,"keywords":"写真 趣味",'
        '"about_partner":true}]',
    )
    assert candidates[0]["occurred_at"] is None


async def test_unreadable_date_does_not_fail_the_reflection(
    client: AsyncClient, fake_llm: FakeLLM
) -> None:
    """読み取れない日付で、振り返り全体を落とさない。日付なしとして残す。"""
    candidates = await _reflect(
        client,
        fake_llm,
        "この前の話なんだけど",
        '[{"kind":"experience","content":"開発者が何かを話した","certainty":"inference",'
        '"provenance":"firsthand","occurred_on":"先週のどこか","keywords":"話",'
        '"about_partner":false}]',
    )
    assert candidates[0]["occurred_at"] is None
