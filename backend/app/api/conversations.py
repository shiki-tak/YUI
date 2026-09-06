"""会話履歴・振り返り・記憶候補・理想の返答の API。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent.memory_store import create_memory
from app.agent.reflection import extract_candidates
from app.db import get_session
from app.llm import get_llm_client
from app.llm.base import LLMClient, LLMError
from app.models import (
    CandidateStatus,
    Conversation,
    IdealResponse,
    MemoryCandidate,
    Message,
    RunRecord,
    Speaker,
    SpeakerKind,
    utcnow,
)
from app.persona import BASE_PERSONA
from app.schemas import (
    CandidateDecision,
    ConversationDetail,
    ConversationOut,
    IdealResponseCreate,
    IdealResponseOut,
    MemoryCandidateOut,
    RunRecordDetail,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationOut])
async def list_conversations(
    limit: int = 50, session: AsyncSession = Depends(get_session)
) -> list[Conversation]:
    stmt = select(Conversation).order_by(Conversation.id.desc()).limit(limit)
    return list((await session.execute(stmt)).scalars())


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: int, session: AsyncSession = Depends(get_session)
) -> Conversation:
    stmt = (
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )
    conversation = (await session.execute(stmt)).scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "会話が見つかりません。")
    return conversation


async def _last_partner(session: AsyncSession, conversation_id: int) -> Speaker | None:
    stmt = (
        select(Speaker)
        .join(Message, Message.speaker_id == Speaker.id)
        .where(
            Message.conversation_id == conversation_id,
            Message.speaker_kind == SpeakerKind.USER.value,
        )
        .order_by(Message.id.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


@router.post("/{conversation_id}/end", response_model=list[MemoryCandidateOut])
async def end_conversation(
    conversation_id: int,
    session: AsyncSession = Depends(get_session),
    llm: LLMClient = Depends(get_llm_client),
) -> list[MemoryCandidate]:
    """会話を終了し、長期記憶の候補を抽出する。採用は別途 /candidates で行う。"""
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "会話が見つかりません。")

    existing = (
        await session.execute(
            select(MemoryCandidate).where(MemoryCandidate.conversation_id == conversation_id)
        )
    ).scalars()
    if any(True for _ in existing):
        raise HTTPException(
            status.HTTP_409_CONFLICT, "この会話の記憶候補はすでに抽出されています。"
        )

    partner = await _last_partner(session, conversation_id)
    try:
        candidates = await extract_candidates(
            session,
            llm=llm,
            conversation=conversation,
            partner_speaker_id=partner.id if partner else None,
            partner_name=partner.display_name if partner else "相手",
            character_name=BASE_PERSONA.name,
        )
    except LLMError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    conversation.ended_at = utcnow()
    return candidates


@router.get("/{conversation_id}/candidates", response_model=list[MemoryCandidateOut])
async def list_candidates(
    conversation_id: int, session: AsyncSession = Depends(get_session)
) -> list[MemoryCandidate]:
    stmt = (
        select(MemoryCandidate)
        .where(MemoryCandidate.conversation_id == conversation_id)
        .order_by(MemoryCandidate.id)
    )
    return list((await session.execute(stmt)).scalars())


@router.post("/candidates/{candidate_id}/decide", response_model=MemoryCandidateOut)
async def decide_candidate(
    candidate_id: int,
    payload: CandidateDecision,
    session: AsyncSession = Depends(get_session),
) -> MemoryCandidate:
    """候補を採用して長期記憶にするか、却下する。"""
    candidate = await session.get(MemoryCandidate, candidate_id)
    if candidate is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "候補が見つかりません。")
    if candidate.status != CandidateStatus.PENDING.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "この候補はすでに判断済みです。")

    if payload.decision == "reject":
        candidate.status = CandidateStatus.REJECTED.value
        return candidate

    memory = await create_memory(
        session,
        kind=(payload.kind.value if payload.kind else candidate.kind),
        content=(payload.content or candidate.content),
        subject_speaker_id=candidate.subject_speaker_id,
        certainty=(payload.certainty.value if payload.certainty else candidate.certainty),
        visibility=(payload.visibility.value if payload.visibility else candidate.visibility),
        keywords=(payload.keywords if payload.keywords is not None else candidate.keywords),
        source_message_id=candidate.source_message_id,
        source_conversation_id=candidate.conversation_id,
        reason=payload.reason or "会話の振り返りから採用",
    )
    candidate.status = CandidateStatus.ACCEPTED.value
    candidate.accepted_memory_id = memory.id
    return candidate


@router.get("/messages/{message_id}/run", response_model=RunRecordDetail)
async def get_run_record(
    message_id: int, session: AsyncSession = Depends(get_session)
) -> RunRecord:
    """返答の実行記録。参照した記憶・モデル・設定・応答時間を確認する。"""
    stmt = select(RunRecord).where(RunRecord.message_id == message_id)
    run = (await session.execute(stmt)).scalar_one_or_none()
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "実行記録が見つかりません。")
    return run


@router.post("/messages/{message_id}/ideal", response_model=IdealResponseOut)
async def create_ideal_response(
    message_id: int,
    payload: IdealResponseCreate,
    session: AsyncSession = Depends(get_session),
) -> IdealResponse:
    """理想の返答を記録する。フェーズ7の教師データと比較評価に使う。"""
    message = await session.get(Message, message_id)
    if message is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "発言が見つかりません。")
    if message.speaker_kind != SpeakerKind.CHARACTER.value:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "理想の返答はキャラクターの発言に対して記録します。"
        )
    ideal = IdealResponse(
        message_id=message_id, ideal_text=payload.ideal_text, note=payload.note
    )
    session.add(ideal)
    await session.flush()
    return ideal


@router.get("/messages/{message_id}/ideal", response_model=list[IdealResponseOut])
async def list_ideal_responses(
    message_id: int, session: AsyncSession = Depends(get_session)
) -> list[IdealResponse]:
    stmt = (
        select(IdealResponse)
        .where(IdealResponse.message_id == message_id)
        .order_by(IdealResponse.id.desc())
    )
    return list((await session.execute(stmt)).scalars())
