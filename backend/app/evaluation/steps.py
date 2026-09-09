"""評価シナリオの手順を実行する（ISSUE-006 の通し評価）。

完了条件1・2は、記憶が**作られてから**使われるまでを見る必要がある。記憶を
事前に入れて1会話を流すだけでは、抽出と採用の経路を通らない。ここでは、
振り返り・採用・訂正・再起動を会話の途中に挟めるようにする。

採用と訂正は、API と同じ処理（`create_memory`・`record_revision`・
`mark_derived_for_review`）を呼ぶ。評価のためだけの近道を作ると、実際の経路と
違うものを測ることになる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent import auto_adopt
from app.agent.character_state import active_states, create_state
from app.agent.character_state import record_revision as state_revision
from app.agent.character_state import snapshot as state_snapshot
from app.agent.derived import mark_derived_for_review
from app.agent.goal import create_goal, list_goals
from app.agent.goal import record_revision as goal_revision
from app.agent.goal import snapshot as goal_snapshot
from app.agent.goal_reflection import propose_goal_candidates
from app.agent.memory_store import create_memory, record_revision, snapshot
from app.agent.reflection import extract_candidates, format_transcript
from app.agent.state_reflection import propose_state_candidates
from app.config import LOCAL_TZ
from app.evaluation.scenario import StepSpec
from app.llm.base import LLMClient
from app.models import (
    CandidateStatus,
    CharacterState,
    Conversation,
    Goal,
    GoalStatus,
    Memory,
    MemoryCandidate,
    MemoryStatus,
    Message,
    SpeakerKind,
    StateKind,
    StateStatus,
    utcnow,
)


@dataclass
class MemoryChange:
    """記憶を1件変えた結果と、その波及。

    「印が付いた」ことを記録で確かめられるようにする。渡っていないことだけを
    見ると、目標を一律に渡さない実装でも通る（第1回レビューの指摘2）。
    """

    memory: Memory
    marked_goals: list[Goal] = field(default_factory=list)
    marked_states: list[CharacterState] = field(default_factory=list)

    @property
    def id(self) -> int:
        return self.memory.id

    @property
    def content(self) -> str:
        return self.memory.content


@dataclass
class ReflectOutcome:
    """振り返りを1回流した結果。"""

    candidates: list[MemoryCandidate] = field(default_factory=list)
    accepted: list[Memory] = field(default_factory=list)
    states: list[CharacterState] = field(default_factory=list)
    accepted_states: list[CharacterState] = field(default_factory=list)
    # 振り返りが出した目標の候補。抽出は後続の PR で入るため、いまは常に空。
    goals: list[Goal] = field(default_factory=list)
    accepted_goals: list[Goal] = field(default_factory=list)
    error: str | None = None


async def run_reflection(
    session: AsyncSession,
    *,
    llm: LLMClient,
    conversation: Conversation,
    character_name: str,
    step: StepSpec,
    now: datetime | None = None,
    auto_adopt_kinds: set[str] | None = None,
) -> ReflectOutcome:
    """会話を振り返り、必要なら候補を採用する。

    会話終了のAPIと同じ順序で、記憶の抽出（拾う→選ぶ）と、関心・関係性の抽出を
    行う。片方だけを流すと、状態への訂正の波及を測れない。

    採用も API と同じ流れで、候補から記憶を作り、候補に採用済みの印を付ける。
    """
    partner_id = await _sole_partner_id(session, conversation.id)
    messages = await _conversation_messages(session, conversation.id)
    current_states = await active_states(
        session, speaker_id=partner_id, mode=conversation.mode
    )

    candidates = await extract_candidates(
        session,
        llm=llm,
        conversation=conversation,
        character_name=character_name,
        # 時間を進めた評価では、進めた側の日付で「昨日」「先週」を解釈させる。
        now=now,
    )
    transcript = format_transcript(messages, character_name)
    state_payloads = await propose_state_candidates(
        llm=llm,
        transcript=transcript,
        partner_speaker_id=partner_id,
        current_states=current_states,
    )
    # 目標の抽出も、API と同じく別の呼び出しで行う。評価だけ経路が欠けると、
    # 完了条件1（自分から質問できる）を抽出から測れない。
    goal_payloads = await propose_goal_candidates(
        llm=llm,
        transcript=transcript,
        partner_speaker_id=partner_id,
        current_goals=[
            goal
            for goal in await list_goals(session, subject_speaker_id=partner_id)
            if goal.status in {GoalStatus.PENDING.value, GoalStatus.ACTIVE.value}
        ],
        # 時間を進めた評価では、進めた側の日付で予定を数えさせる。
        today=(now or utcnow()).astimezone(LOCAL_TZ).date(),
    )

    session.add_all(candidates)
    states: list[CharacterState] = []
    for payload in state_payloads:
        is_relationship = payload.kind == StateKind.RELATIONSHIP.value
        states.append(
            await create_state(
                session,
                kind=payload.kind,
                content=payload.content,
                topic=None if is_relationship else payload.topic,
                subject_speaker_id=partner_id if is_relationship else None,
                visible_to_speaker_id=partner_id,
                source_conversation_id=conversation.id,
                status=StateStatus.PENDING.value,
                reason=payload.reason or "会話の振り返りから",
            )
        )
    goals: list[Goal] = []
    for goal_payload in goal_payloads:
        trigger, due_at = goal_payload.schedule()
        goals.append(
            await create_goal(
                session,
                content=goal_payload.content,
                subject_speaker_id=partner_id,
                trigger=trigger,
                due_at=due_at,
                visible_to_speaker_id=partner_id,
                source_conversation_id=conversation.id,
                status=GoalStatus.PENDING.value,
                reason=goal_payload.reason or "会話の振り返りから",
            )
        )
    await session.flush()

    # 自動採用（フェーズ4 PR11）。**本番の振り返りと同じ関数を通す。**
    # 評価が別経路を持つと、自動採用を有効にして測ったつもりが、一度も通って
    # いない測定になる。PR12 は有無を比べる測定なので、そこが狂うと結論が変わる。
    auto_accepted = await auto_adopt.apply(
        session,
        kinds=auto_adopt_kinds or set(),
        candidates=candidates,
        states=states,
        goals=goals,
        # 時間を進めた評価では、進めた側の時刻で記憶を作る。自動採用だけ実時計
        # だと、検索の時間減衰が採用の有無で変わる（レビューの指摘5）。
        now=now,
    )

    outcome = ReflectOutcome(candidates=candidates, states=states, goals=goals)
    # **自動で採用したものも「採用した記憶」として扱う**（レビューの指摘2）。
    # 入れないと、後続の手順が使う鍵とレポートの表示が揃わない。
    outcome.accepted.extend(auto_accepted)
    if not step.accept:
        await session.commit()
        return outcome

    for candidate in candidates:
        if step.accept_contains and not any(
            word in candidate.content for word in step.accept_contains
        ):
            continue
        if candidate.status != CandidateStatus.PENDING.value:
            # すでに自動採用されている。もう一度作ると、**本番には無い重複の
            # 記憶が評価DBに入る**（レビューの指摘2）。採用済みという結果は
            # 同じなので、この手順は満たされている。
            continue
        memory = await create_memory(
            session,
            kind=candidate.kind,
            content=candidate.content,
            subject_speaker_id=candidate.subject_speaker_id,
            visible_to_speaker_id=candidate.visible_to_speaker_id,
            certainty=candidate.certainty,
            provenance=candidate.provenance,
            visibility=candidate.visibility,
            keywords=candidate.keywords,
            occurred_at=candidate.occurred_at,
            source_message_id=candidate.source_message_id,
            source_conversation_id=candidate.conversation_id,
            reason="評価用会話で採用",
            # 時間を進めた後に採用した記憶は、進めた側の時刻で作る。実時計だと、
            # 検索の減衰が「まだ作られていない記憶」を新しいものとして扱う。
            created_at=now,
        )
        candidate.status = CandidateStatus.ACCEPTED.value
        candidate.accepted_memory_id = memory.id
        outcome.accepted.append(memory)

    if step.accept_states:
        # 状態も採用する。API と同じく、根拠が空ならその会話から採用された
        # 記憶を暫定の根拠として結び付ける。
        await session.flush()
        for state in states:
            before = state_snapshot(state)
            if not state.basis_memory_ids and state.source_conversation_id is not None:
                basis = [memory.id for memory in outcome.accepted]
                state.basis_memory_ids = basis or None
                state.basis_is_provisional = True
            state.status = StateStatus.ACTIVE.value
            await session.flush()
            state_revision(
                session, state, action="accepted", before=before, reason="評価用会話で採用"
            )
            outcome.accepted_states.append(state)

    await session.commit()
    return outcome


async def _sole_partner_id(session: AsyncSession, conversation_id: int) -> int | None:
    """その会話にひとりだけいる相手。複数いれば None（会話APIと同じ扱い）。"""
    stmt = (
        select(Message.speaker_id)
        .where(
            Message.conversation_id == conversation_id,
            Message.speaker_kind == SpeakerKind.USER.value,
            Message.speaker_id.isnot(None),
        )
        .distinct()
        .limit(2)
    )
    found = list((await session.execute(stmt)).scalars())
    return found[0] if len(found) == 1 else None


async def _conversation_messages(session: AsyncSession, conversation_id: int) -> list[Message]:
    stmt = (
        select(Message)
        .options(selectinload(Message.speaker))
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id)
    )
    return list((await session.execute(stmt)).scalars())


async def _find_memory(session: AsyncSession, match: str) -> Memory | None:
    stmt = select(Memory).where(
        Memory.status == MemoryStatus.ACTIVE.value, Memory.content.contains(match)
    )
    return (await session.execute(stmt)).scalars().first()


async def correct_memory(
    session: AsyncSession, *, match: str, content: str
) -> MemoryChange | None:
    """開発者が記憶を訂正する。API と同じく、履歴と波及も起こす。"""
    memory = await _find_memory(session, match)
    if memory is None:
        return None
    before = snapshot(memory)
    memory.content = content
    await session.flush()
    record_revision(session, memory, action="corrected", before=before, reason="評価用会話で訂正")
    marks = await mark_derived_for_review(
        session,
        memory_id=memory.id,
        reason=f"根拠にした記憶 #{memory.id} が訂正された",
        source_conversation_id=memory.source_conversation_id,
    )
    await session.commit()
    return MemoryChange(memory=memory, marked_goals=marks.goals, marked_states=marks.states)


async def delete_memory(session: AsyncSession, *, match: str) -> MemoryChange | None:
    """開発者が記憶を削除する。"""
    memory = await _find_memory(session, match)
    if memory is None:
        return None
    before = snapshot(memory)
    memory.status = MemoryStatus.DELETED.value
    await session.flush()
    record_revision(session, memory, action="deleted", before=before, reason="評価用会話で削除")
    marks = await mark_derived_for_review(
        session,
        memory_id=memory.id,
        reason=f"根拠にした記憶 #{memory.id} が削除された",
        source_conversation_id=memory.source_conversation_id,
    )
    await session.commit()
    return MemoryChange(memory=memory, marked_goals=marks.goals, marked_states=marks.states)


async def accept_goal(session: AsyncSession, *, match: str) -> list[Goal]:
    """開発者が目標を採用する。API と同じく、状態を変えて履歴を残す。

    **これは「使う対象が採用済みになったか」の確認である**（PR11 第2回レビュー）。
    自動採用に限らず、すでに active の目標も通す。合格を「開発者が採用した証拠」
    と読んではいけない。手動採用の経路そのものは、自動採用なしの設定で別に測る。

    候補のまま置いた目標を、行動選択へ渡る状態にする。採用の経路を通さずに
    最初から採用済みで置くと、「候補 → 採用 → 参照」の経路を測れない。
    フェーズ3で、記憶を事前に入れる評価が抽出と採用を通っていなかったのと
    同じ穴になる。
    """
    stmt = select(Goal).where(
        Goal.status.in_([GoalStatus.PENDING.value, GoalStatus.ACTIVE.value]),
        Goal.content.contains(match),
    )
    goals = list((await session.execute(stmt)).scalars())
    for goal in goals:
        if goal.status == GoalStatus.ACTIVE.value:
            # **すでに自動採用されている**（PR11 レビューの指摘2）。採用済み
            # という結果は同じなので、この手順は満たされている。候補だけを
            # 探すと「『感想』に当たる目標が無かった」と不合格になり、
            # **自動採用に成功しただけで失敗が増える。**
            continue
        before = goal_snapshot(goal)
        goal.status = GoalStatus.ACTIVE.value
        await session.flush()
        goal_revision(session, goal, action="accepted", before=before, reason="評価用会話で採用")
    await session.commit()
    return goals
