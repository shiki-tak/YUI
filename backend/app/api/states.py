"""可変状態の API（ISSUE-015・016 / 設計書フェーズ3の3B）。

関心と関係性を、候補から採用まで開発者が確認できるようにする。固定人格は
ここでは触れない。人格は版として管理し、日常の更新で上書きしない。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.character_state import (
    create_state,
    list_states,
    record_revision,
    snapshot,
)
from app.db import get_session
from app.models import CharacterState, CharacterStateRevision, StateStatus
from app.schemas import (
    CharacterStateCreate,
    CharacterStateDecision,
    CharacterStateOut,
    CharacterStateRevisionOut,
    CharacterStateUpdate,
)

router = APIRouter(tags=["states"])


async def _get(session: AsyncSession, state_id: int) -> CharacterState:
    state = await session.get(CharacterState, state_id)
    if state is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "状態が見つかりません。")
    return state


@router.get("/states", response_model=list[CharacterStateOut])
async def read_states(
    kind: str | None = None,
    state_status: str | None = None,
    needs_review: bool | None = None,
    session: AsyncSession = Depends(get_session),
) -> list[CharacterState]:
    """関心・関係性の一覧。

    needs_review=true で、根拠が変わって再評価が要るものだけを見られる。
    """
    return await list_states(session, kind=kind, status=state_status, needs_review=needs_review)


@router.post("/states", response_model=CharacterStateOut, status_code=status.HTTP_201_CREATED)
async def add_state(
    payload: CharacterStateCreate, session: AsyncSession = Depends(get_session)
) -> CharacterState:
    """状態を候補として追加する。採用するまで会話には使わない。"""
    return await create_state(
        session,
        kind=payload.kind,
        content=payload.content,
        topic=payload.topic,
        subject_speaker_id=payload.subject_speaker_id,
        basis_memory_ids=payload.basis_memory_ids,
        visibility=payload.visibility.value,
        visible_to_speaker_id=payload.visible_to_speaker_id,
        reason=payload.reason or "手動で追加",
    )


@router.post("/states/{state_id}/decide", response_model=CharacterStateOut)
async def decide_state(
    state_id: int, payload: CharacterStateDecision, session: AsyncSession = Depends(get_session)
) -> CharacterState:
    """候補を採用または却下する。"""
    state = await _get(session, state_id)
    if state.status != StateStatus.PENDING.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "この状態はすでに判断済みです。")

    before = snapshot(state)
    if payload.decision == "accept":
        if payload.content:
            state.content = payload.content
        state.status = StateStatus.ACTIVE.value
        action = "accepted"
    else:
        state.status = StateStatus.REJECTED.value
        action = "rejected"
    await session.flush()
    record_revision(session, state, action=action, before=before, reason=payload.reason)
    return state


@router.patch("/states/{state_id}", response_model=CharacterStateOut)
async def update_state(
    state_id: int, payload: CharacterStateUpdate, session: AsyncSession = Depends(get_session)
) -> CharacterState:
    """採用済みの状態を直す。再評価の印を下ろすのもここで行う。"""
    state = await _get(session, state_id)
    changes = payload.model_dump(exclude_unset=True, exclude={"reason"})
    if not changes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "変更する項目がありません。")

    before = snapshot(state)
    if payload.content:
        state.content = payload.content
    if payload.status is not None:
        state.status = payload.status
    if payload.reviewed:
        # 確認したので印を下ろす。何を根拠に下ろしたかは履歴に残る。
        state.needs_review = False
        state.review_reason = None
    await session.flush()
    record_revision(session, state, action="corrected", before=before, reason=payload.reason)
    return state


@router.get("/states/{state_id}/revisions", response_model=list[CharacterStateRevisionOut])
async def read_revisions(
    state_id: int, session: AsyncSession = Depends(get_session)
) -> list[CharacterStateRevision]:
    await _get(session, state_id)
    stmt = (
        select(CharacterStateRevision)
        .where(CharacterStateRevision.state_id == state_id)
        .order_by(CharacterStateRevision.id.desc())
    )
    return list((await session.execute(stmt)).scalars())
