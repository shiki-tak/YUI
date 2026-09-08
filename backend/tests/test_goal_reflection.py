"""会話から目標を抽出する（フェーズ4 PR6）。

振り返りの4回目の呼び出し。記憶・関心とは別の呼び出しにしてある。同じ指示文へ
項目を足すと元の抽出が落ちることを、フェーズ3で4回測った（ISSUE-017・021）。

ここで確かめるのは器の側で、抽出の質ではない。質は実モデルで
`python -m app.evaluation.run --aspect proactive` を流して見る。
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from httpx import AsyncClient
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agent.goal_reflection import (
    GoalPayload,
    GoalReflectionError,
    format_current_goals,
    propose_goal_candidates,
)
from app.models import Goal, GoalStatus, GoalTrigger
from tests.conftest import FakeLLM, end_and_wait, reflection_progress


class _Stub:
    """指示文と入力を記録するだけのクライアント。"""

    def __init__(self, text: str) -> None:
        self.text = text
        self.messages: list = []

    async def chat(self, messages, *, options=None):
        self.messages = messages
        from app.llm.base import LLMResponse

        return LLMResponse(
            text=self.text,
            provider="stub",
            model="stub",
            model_digest=None,
            options=options or {},
            latency_ms=1,
            prompt_tokens=1,
            completion_tokens=1,
        )


async def test_a_plan_becomes_a_goal_with_a_date() -> None:
    """予定を話したら、その日以降に聞く目標として読み取る。"""
    llm = _Stub(
        '[{"content": "土曜に見た映画の感想を聞く", "after_date": "2026-09-14",'
        ' "reason": "土曜に映画を見に行くと話した"}]'
    )
    payloads = await propose_goal_candidates(
        llm=llm,  # type: ignore[arg-type]
        transcript="開発者: 今週の土曜に映画を見に行くんだ",
        partner_speaker_id=1,
        current_goals=[],
        today=date(2026, 9, 8),
    )
    assert len(payloads) == 1
    trigger, due_at = payloads[0].schedule()
    assert trigger == GoalTrigger.AFTER_DATE.value
    # 日本時間 9/14 00:00 は UTC では 9/13 15:00。保存は UTC に揃える。
    assert due_at == datetime(2026, 9, 13, 15, 0, tzinfo=UTC)


async def test_a_goal_without_a_date_waits_for_the_next_conversation() -> None:
    """日付が無い目標は「次の会話」で実行してよい。"""
    for value in ('""', "null"):
        llm = _Stub(f'[{{"content": "カメラの設定の話をする", "after_date": {value}}}]')
        payloads = await propose_goal_candidates(
            llm=llm,  # type: ignore[arg-type]
            transcript="開発者: 次はカメラの設定の話をしよう",
            partner_speaker_id=1,
            current_goals=[],
            today=date(2026, 9, 8),
        )
        assert payloads[0].schedule() == (GoalTrigger.NEXT_CONVERSATION.value, None)


async def test_an_unreadable_date_is_a_failure_not_a_looser_condition() -> None:
    """書いてあるのに読めない日付は、失敗として扱う（第1回レビューの指摘3）。

    黙って「次の会話」へ落とすと、予定より前に聞いてよい目標になる。記憶の
    occurred_at は欠けても意味が変わらないが、**実行条件は欠けると意味が
    変わる**。制限が緩む方向へ黙ってずらさない。
    """
    for value in ("来週くらい", "2026-13-45", "9/14"):
        llm = _Stub(
            f'[{{"content": "映画の感想を聞く", "after_date": "{value}"}}]'
        )
        with pytest.raises(GoalReflectionError) as exc:
            await propose_goal_candidates(
                llm=llm,  # type: ignore[arg-type]
                transcript="開発者: 今週の土曜に映画を見に行く",
                partner_speaker_id=1,
                current_goals=[],
                today=date(2026, 9, 8),
            )
        assert "実行の基準日" in str(exc.value)


async def test_unreadable_output_is_an_error_not_an_empty_result() -> None:
    """読み取れなかったことを、「目標なしの成功」にしない。

    静かに空を返すと、失敗した会話が終了済みになり、やり直せなくなる
    （記憶・状態の抽出と同じ扱い）。
    """
    llm = _Stub("すみません、うまくまとめられませんでした。")
    with pytest.raises(GoalReflectionError):
        await propose_goal_candidates(
            llm=llm,  # type: ignore[arg-type]
            transcript="開発者: こんばんは",
            partner_speaker_id=1,
            current_goals=[],
            today=date(2026, 9, 8),
        )

    # 配列は読めても、要素が読み取れなければ失敗として扱う。
    llm = _Stub('[{"after_date": "2026-09-14"}]')
    with pytest.raises(GoalReflectionError):
        await propose_goal_candidates(
            llm=llm,  # type: ignore[arg-type]
            transcript="開発者: こんばんは",
            partner_speaker_id=1,
            current_goals=[],
            today=date(2026, 9, 8),
        )


async def test_no_goals_when_the_partner_is_unclear() -> None:
    """相手を決められない会話では、目標を作らない。

    目標は「誰に聞くか」を持つ。相手が決まらないまま作ると、別の相手との会話で
    持ち出される（関心・関係性と同じ扱い、設計書 3C）。
    """
    llm = _Stub('[{"content": "作られてはいけない目標"}]')
    assert (
        await propose_goal_candidates(
            llm=llm,  # type: ignore[arg-type]
            transcript="Aさん: こんばんは\nBさん: どうも",
            partner_speaker_id=None,
            current_goals=[],
            today=date(2026, 9, 8),
        )
        == []
    )
    # モデルを呼ぶ前に決める。呼んでから捨てるのは無駄で、失敗の種にもなる。
    assert llm.messages == []


async def test_existing_goals_are_shown_so_they_are_not_repeated() -> None:
    """いまある目標を渡す。同じ内容を重ねて出させないため。"""
    llm = _Stub("[]")
    goal = Goal(id=1, content="土曜に見た映画の感想を聞く")
    await propose_goal_candidates(
        llm=llm,  # type: ignore[arg-type]
        transcript="開発者: こんばんは",
        partner_speaker_id=1,
        current_goals=[goal],
        today=date(2026, 9, 8),
    )
    assert "土曜に見た映画の感想を聞く" in llm.messages[1].content
    assert "振り返りを行っている日: 2026-09-08" in llm.messages[1].content
    assert format_current_goals([]) == "いまの目標: まだありません。"


async def test_extraction_is_a_separate_call_from_memories_and_states(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """振り返りが目標を候補として保存する。記憶・関心とは別の呼び出しで作る。

    同じ指示文へ項目を足すと、元の抽出が落ちる（ISSUE-017 で実測）。役割ごとに
    分ければ、片方の調整がもう片方を壊さない。
    """
    first = await client.post("/api/chat", json={"text": "今週の土曜に映画を見に行くんだ"})
    conversation_id = first.json()["conversation_id"]

    fake_llm.push(
        '[{"kind":"experience","content":"開発者は土曜に映画を見に行く","certainty":"fact",'
        '"provenance":"firsthand","keywords":"映画 土曜","about_partner":true}]'
    )
    fake_llm.push_state('[{"kind":"interest","topic":"映画","content":"映画の話に興味がある"}]')
    fake_llm.push_goal(
        '[{"content": "土曜に見た映画の感想を聞く", "after_date": "2026-09-14",'
        ' "reason": "土曜に映画を見に行くと話した"}]'
    )

    await end_and_wait(client, conversation_id)
    assert (await reflection_progress(client, conversation_id))["state"] == "completed"

    goals = (await client.get("/api/goals")).json()
    assert [g["content"] for g in goals] == ["土曜に見た映画の感想を聞く"]
    # 候補のまま。採用するまで行動選択には渡さない。
    assert goals[0]["status"] == GoalStatus.PENDING.value
    assert goals[0]["trigger"] == GoalTrigger.AFTER_DATE.value
    assert goals[0]["source_conversation_id"] == conversation_id
    # 相手が決まっている。別の相手との会話には渡さない。
    assert goals[0]["subject_speaker_id"] is not None
    assert goals[0]["visible_to_speaker_id"] == goals[0]["subject_speaker_id"]


async def test_a_failed_goal_extraction_does_not_save_the_others(
    client: AsyncClient, fake_llm: FakeLLM, session_factory: async_sessionmaker
) -> None:
    """目標の抽出だけが失敗しても、記憶・関心を部分的に保存しない。

    保存はすべての抽出が成功してからまとめて行う（フェーズ3全体レビューの
    指摘4・5）。4回目が増えても同じ扱いにする。
    """
    first = await client.post("/api/chat", json={"text": "今週の土曜に映画を見に行くんだ"})
    conversation_id = first.json()["conversation_id"]

    fake_llm.push(
        '[{"kind":"experience","content":"開発者は土曜に映画を見に行く","certainty":"fact",'
        '"provenance":"firsthand","keywords":"映画","about_partner":true}]'
    )
    fake_llm.push_state("[]")
    fake_llm.push_goal("読み取れない出力")

    await end_and_wait(client, conversation_id)

    progress = await reflection_progress(client, conversation_id)
    assert progress["state"] == "failed"
    assert "目標の候補" in progress["error"]

    candidates = (
        await client.get(f"/api/conversations/{conversation_id}/candidates")
    ).json()
    assert candidates == []
    async with session_factory() as session:
        assert list((await session.execute(select(Goal))).scalars()) == []


def test_payload_rejects_blank_content() -> None:
    """空白だけの本文を、目標として受け取らない（ISSUE-001 と同じ）。"""
    with pytest.raises(ValidationError):
        GoalPayload.model_validate({"content": "   \n\t"})


async def test_a_failed_goal_extraction_is_recorded_per_attempt(fake_llm: FakeLLM) -> None:
    """評価は、目標の抽出が失敗しても試行の失敗として記録する（指摘1）。

    例外が評価全体へ漏れると、残りの試行とシナリオが走らず、集計そのものが
    取れない。記憶・関心の抽出と同じ扱いにする。
    """
    from app.config import get_settings
    from app.evaluation.runner import run_scenarios
    from app.evaluation.scenario import Scenario
    from app.persona import load_persona

    scenario = Scenario.model_validate(
        {
            "id": "goal-extraction-fails",
            "aspect": "proactive",
            "steps": [
                {"kind": "say", "text": "今週の土曜に映画を見に行くんだ"},
                {"kind": "reflect", "expect_goal_any": ["感想"]},
            ],
        }
    )
    # 1つ目は say の返答、2つ目が記憶の「選ぶ」側。
    fake_llm.push("楽しみですね。")
    fake_llm.push("[]")
    fake_llm.push_state("[]")
    fake_llm.push_goal("読み取れない出力")

    results = await run_scenarios(
        [scenario], llm=fake_llm, persona=load_persona(), settings=get_settings()
    )
    attempt = results[0].attempts[0]
    assert not attempt.ok
    assert attempt.reflections and attempt.reflections[0].error
    assert "目標の候補" in attempt.reflections[0].error


async def test_relative_dates_are_counted_from_the_utterance() -> None:
    """相対的な日付は、発言の日から数えるよう指示する（指摘2）。

    振り返る日を基準にすると、同じ会話を別の日に振り返っただけで予定の日付が
    変わる。記憶の抽出は発言日を基準にしており、そちらと食い違う。
    """
    llm = _Stub("[]")
    await propose_goal_candidates(
        llm=llm,  # type: ignore[arg-type]
        transcript="[#1／2026-09-08] 開発者: 明日、面接があるんだ",
        partner_speaker_id=1,
        current_goals=[],
        today=date(2026, 9, 10),
    )
    instruction = llm.messages[0].content
    hint = llm.messages[1].content
    assert "その言い方が出てきた発言の日付から数える" in instruction + hint
    # 振り返る日は、予定が過ぎたかどうかの判断にだけ使うと書いてある。
    assert "予定がもう過ぎたかどうか" in hint
