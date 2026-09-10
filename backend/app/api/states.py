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
from app.models import (
    CharacterState,
    CharacterStateRevision,
    Memory,
    MemoryStatus,
    StateStatus,
    Visibility,
)
from app.schemas import (
    CharacterStateCreate,
    CharacterStateDecision,
    CharacterStateOut,
    CharacterStateRevisionOut,
    CharacterStateUpdate,
)

router = APIRouter(tags=["states"])


async def _check_scope_against_basis(session: AsyncSession, payload: CharacterStateCreate) -> None:
    """根拠の記憶より広い参照範囲を許さない。"""
    stmt = select(Memory).where(Memory.id.in_(payload.basis_memory_ids))
    memories = list((await session.execute(stmt)).scalars())
    found = {memory.id for memory in memories}
    missing = [mid for mid in payload.basis_memory_ids if mid not in found]
    if missing:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"根拠にした記憶が見つかりません: {'、'.join(str(m) for m in missing)}",
        )

    dead = [memory.id for memory in memories if memory.status != MemoryStatus.ACTIVE.value]
    if dead:
        # 削除・訂正された記憶を根拠にした状態は、作られた時点で古い前提を
        # 持っている。訂正の波及（ISSUE-016）は「後から変わったもの」を拾う
        # 仕組みなので、作る前から無効だった根拠は拾えない（ISSUE-026）。
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"根拠にした記憶が有効ではありません: "
            f"{'、'.join(f'#{mid}' for mid in dead)}。"
            "訂正後の記憶を根拠にしてください。",
        )

    for memory in memories:
        if memory.visibility == Visibility.PRIVATE.value and (
            payload.visibility == Visibility.PUBLIC
        ):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"記憶 #{memory.id} は非公開です。そこから作る状態を公開にはできません。",
            )
        limited_to = memory.visible_to_speaker_id
        if limited_to is not None and payload.visible_to_speaker_id != limited_to:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"記憶 #{memory.id} は相手を限定しています。"
                f"そこから作る状態も、その相手に限定してください"
                f"（visible_to_speaker_id={limited_to}）。",
            )


async def _dead_basis(session: AsyncSession, state: CharacterState) -> list[int]:
    """根拠のうち、もう有効でない記憶。"""
    if not state.basis_memory_ids:
        return []
    stmt = select(Memory).where(
        Memory.id.in_(state.basis_memory_ids),
        Memory.status == MemoryStatus.ACTIVE.value,
    )
    alive = {memory.id for memory in (await session.execute(stmt)).scalars()}
    return [mid for mid in state.basis_memory_ids if mid not in alive]


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
    """状態を候補として追加する。採用するまで会話には使わない。

    根拠にした記憶より広い範囲では作れない。記憶側で相手を限定していても、
    そこから作った状態が別の相手へ渡ると、限定した意味が無くなる。
    """
    if payload.basis_memory_ids:
        await _check_scope_against_basis(session, payload)
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
        # 根拠が会話単位しかない状態に、その会話から採用された記憶を結び付ける。
        # 候補の時点では記憶がまだ採用されておらず、記憶IDを持てないため
        # （フェーズ3再レビューの指摘1）。
        if not state.basis_memory_ids and state.source_conversation_id is not None:
            stmt = select(Memory).where(
                Memory.source_conversation_id == state.source_conversation_id,
                Memory.status == MemoryStatus.ACTIVE.value,
            )
            basis = [memory.id for memory in (await session.execute(stmt)).scalars()]
            state.basis_memory_ids = basis or None
            # 自動で並べた根拠は暫定。後から採用される記憶が入らないため、
            # 訂正の波及では会話単位でも拾う（再々レビューの指摘1）。
            state.basis_is_provisional = True
        state.status = StateStatus.ACTIVE.value
        action = "accepted"
        # 候補として置かれてから採用までの間に、根拠が消えていることがある。
        # 拒否はせず印を付ける。状態そのものを捨てるかは開発者が決める
        # （印が付いている間は会話へ渡らない）。撤回から有効へ戻すときの
        # 検査と揃える（ISSUE-026）。
        gone = await _dead_basis(session, state)
        if gone:
            state.needs_review = True
            state.review_reason = (
                f"根拠にした記憶 {'、'.join(f'#{mid}' for mid in gone)} が有効でなくなっている"
            )
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
    if payload.status == StateStatus.ACTIVE.value and not payload.reviewed:
        # 有効へ戻すときは、根拠がまだ生きているかを見る。撤回している間に
        # 根拠が消えていることがある（再々レビューの指摘2）。
        gone = await _dead_basis(session, state)
        if gone:
            state.needs_review = True
            state.review_reason = (
                f"根拠にした記憶 {'、'.join(f'#{mid}' for mid in gone)} が有効でなくなっている"
            )
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
