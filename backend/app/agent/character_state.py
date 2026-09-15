"""変化する状態：YUI の関心と、相手との関係（ISSUE-015・016）。

設計書 4.1 が分けている4層のうち、経験で変わっていくものを扱う。固定人格
（口調・価値観・自己設定）は personas/<版>.toml にあり、ここでは触らない。
日常の振り返りで人格を上書きしないための境目である。

保存の形は記憶と揃える。候補を出し、開発者が採用し、根拠と変更履歴を残す。
単一の好感度の数値にはしない。何を根拠にそう思っているかを本文で残す。
"""

from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    CharacterState,
    CharacterStateRevision,
    ConversationMode,
    StateKind,
    StateStatus,
    Visibility,
)

KIND_LABEL = {
    StateKind.INTEREST.value: "関心",
    StateKind.RELATIONSHIP.value: "相手との関係",
}


def snapshot(state: CharacterState) -> dict:
    """変更履歴に残す内容。"""
    return {
        "kind": state.kind,
        "subject_speaker_id": state.subject_speaker_id,
        "topic": state.topic,
        "content": state.content,
        "basis_memory_ids": list(state.basis_memory_ids or []),
        "basis_is_provisional": state.basis_is_provisional,
        "needs_review": state.needs_review,
        "review_reason": state.review_reason,
        "status": state.status,
        "visibility": state.visibility,
        "visible_to_speaker_id": state.visible_to_speaker_id,
        "superseded_by_id": state.superseded_by_id,
    }


def record_revision(
    session: AsyncSession,
    state: CharacterState,
    *,
    action: str,
    before: dict | None,
    reason: str | None = None,
) -> CharacterStateRevision:
    revision = CharacterStateRevision(
        state_id=state.id, action=action, before=before, after=snapshot(state), reason=reason
    )
    session.add(revision)
    return revision


async def create_state(
    session: AsyncSession,
    *,
    kind: str,
    content: str,
    topic: str | None = None,
    subject_speaker_id: int | None = None,
    basis_memory_ids: list[int] | None = None,
    visibility: str = Visibility.PRIVATE.value,
    visible_to_speaker_id: int | None = None,
    source_conversation_id: int | None = None,
    status: str = StateStatus.PENDING.value,
    reason: str | None = None,
) -> CharacterState:
    """状態を作る。既定は候補（pending）で、会話にはまだ使わない。"""
    state = CharacterState(
        kind=kind,
        content=content,
        topic=topic,
        subject_speaker_id=subject_speaker_id,
        basis_memory_ids=basis_memory_ids or None,
        visibility=visibility,
        visible_to_speaker_id=visible_to_speaker_id,
        source_conversation_id=source_conversation_id,
        status=status,
    )
    session.add(state)
    await session.flush()
    record_revision(session, state, action="created", before=None, reason=reason)
    return state


async def list_states(
    session: AsyncSession,
    *,
    kind: str | None = None,
    status: str | None = None,
    needs_review: bool | None = None,
) -> list[CharacterState]:
    stmt = select(CharacterState).order_by(CharacterState.id.desc())
    if kind is not None:
        stmt = stmt.where(CharacterState.kind == kind)
    if status is not None:
        stmt = stmt.where(CharacterState.status == status)
    if needs_review is not None:
        stmt = stmt.where(CharacterState.needs_review.is_(needs_review))
    return list((await session.execute(stmt)).scalars())


async def active_states(
    session: AsyncSession, *, speaker_id: int | None, mode: str = ConversationMode.LOCAL
) -> list[CharacterState]:
    """会話に渡す状態。

    関心は YUI 自身のものなので、相手が誰でも渡す。関係性は、その相手のものだけ
    渡す。別の相手との関係を持ち出さないため（設計書 3C）。

    再評価が必要な印が付いたものは渡さない。根拠が変わったまま使い続けると、
    訂正が反映されていない状態で話すことになる（ISSUE-016）。
    """
    stmt = select(CharacterState).where(
        CharacterState.status == StateStatus.ACTIVE.value,
        CharacterState.needs_review.is_(False),
    )
    if mode == ConversationMode.STREAM:
        stmt = stmt.where(CharacterState.visibility == Visibility.PUBLIC.value)
    else:
        scope = CharacterState.visible_to_speaker_id.is_(None)
        if speaker_id is not None:
            scope = or_(scope, CharacterState.visible_to_speaker_id == speaker_id)
        stmt = stmt.where(scope)

    if speaker_id is not None:
        stmt = stmt.where(
            or_(
                CharacterState.kind == StateKind.INTEREST.value,
                CharacterState.subject_speaker_id == speaker_id,
            )
        )
    else:
        stmt = stmt.where(CharacterState.kind == StateKind.INTEREST.value)
    return list((await session.execute(stmt.order_by(CharacterState.id))).scalars())


async def mark_for_review(
    session: AsyncSession,
    *,
    memory_id: int,
    reason: str,
    source_conversation_id: int | None = None,
) -> list[CharacterState]:
    """根拠にした記憶が変わった状態へ、再評価の印を付ける（ISSUE-016）。

    自動では消さない。訂正が正しいのか、そこから作った状態も直すべきなのかは
    開発者が判断する。印が付いている間は会話に渡さない。

    根拠を記憶IDで持つ状態だけでなく、**同じ会話から作られた状態**にも印を
    付ける。振り返りが作った状態は、候補の時点では記憶がまだ採用されておらず、
    記憶IDを持てない。会話単位は粗いが、印は会話に渡さなくするだけで、
    開発者が確認すれば戻せる。届かないまま古い内容を話すほうが重い
    （v0.1 レビューの指摘）。
    """
    stmt = select(CharacterState).where(
        # 撤回中のものも対象にする。撤回している間に根拠が変わり、そのまま
        # 有効へ戻すと、古い内容が印なしで会話へ渡る（再々レビューの指摘2）。
        CharacterState.status.in_(
            [
                StateStatus.ACTIVE.value,
                StateStatus.PENDING.value,
                StateStatus.WITHDRAWN.value,
            ]
        ),
    )
    affected: list[CharacterState] = []
    for state in (await session.execute(stmt)).scalars():
        by_memory = memory_id in (state.basis_memory_ids or [])
        # 根拠を持たない状態と、自動で並べた暫定の根拠しか持たない状態は、
        # 会話単位で拾う。暫定の根拠は、後から採用された記憶が抜けているため、
        # 完全に特定した根拠として扱えない（再々レビューの指摘1）。
        by_conversation = (
            (not state.basis_memory_ids or state.basis_is_provisional)
            and source_conversation_id is not None
            and state.source_conversation_id == source_conversation_id
        )
        if not (by_memory or by_conversation):
            continue
        before = snapshot(state)
        state.needs_review = True
        state.review_reason = reason
        await session.flush()
        record_revision(session, state, action="needs_review", before=before, reason=reason)
        affected.append(state)
    return affected
