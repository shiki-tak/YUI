"""長期記憶の一覧・検索・追加・訂正・削除。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.character_state import mark_for_review
from app.agent.memory_store import (
    create_memory,
    record_revision,
    search_memories,
    snapshot,
)
from app.config import Settings, get_settings
from app.db import get_session
from app.models import (
    ConversationMode,
    Memory,
    MemoryRevision,
    MemoryStatus,
    Speaker,
    Visibility,
)
from app.schemas import (
    MemoryCreate,
    MemoryOut,
    MemoryRevisionOut,
    MemorySearchResult,
    MemoryUpdate,
    RetrievedMemoryOut,
    SpeakerOut,
)

router = APIRouter(tags=["memories"])


@router.get("/speakers", response_model=list[SpeakerOut])
async def list_speakers(session: AsyncSession = Depends(get_session)) -> list[Speaker]:
    return list((await session.execute(select(Speaker).order_by(Speaker.id))).scalars())


@router.get("/memories", response_model=list[MemoryOut])
async def list_memories(
    include_inactive: bool = False,
    subject_speaker_id: int | None = None,
    unscoped: bool = False,
    limit: int = Query(default=200, le=1000),
    session: AsyncSession = Depends(get_session),
) -> list[Memory]:
    """記憶の一覧。

    unscoped=true で、参照範囲を限定していない非公開の記憶だけを返す。
    複数の相手が入る前に、どれを誰との会話に限るかを振り分けるために使う
    （ISSUE-010）。公開可能な記憶は、限定しないことが前提なので含めない。
    """
    stmt = select(Memory).order_by(Memory.id.desc()).limit(limit)
    if not include_inactive:
        stmt = stmt.where(Memory.status == MemoryStatus.ACTIVE.value)
    if subject_speaker_id is not None:
        stmt = stmt.where(Memory.subject_speaker_id == subject_speaker_id)
    if unscoped:
        stmt = stmt.where(
            Memory.visible_to_speaker_id.is_(None),
            Memory.visibility == Visibility.PRIVATE.value,
        )
    return list((await session.execute(stmt)).scalars())


@router.get("/memories/search", response_model=MemorySearchResult)
async def search(
    q: str = Query(min_length=1),
    speaker_id: int | None = None,
    mode: str = ConversationMode.LOCAL.value,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> MemorySearchResult:
    """会話で使われる検索と同じ経路。何が引けるかを開発画面で確認する。"""
    results = await search_memories(
        session,
        query=q,
        speaker_id=speaker_id,
        mode=mode,
        limit=settings.memory_retrieval_limit,
    )
    return MemorySearchResult(
        query=q,
        results=[
            RetrievedMemoryOut(
                memory=MemoryOut.model_validate(item.memory),
                score=item.score,
                reason=item.reason,
            )
            for item in results
        ],
    )


@router.get("/memories/{memory_id}", response_model=MemoryOut)
async def get_memory(
    memory_id: int, session: AsyncSession = Depends(get_session)
) -> Memory:
    """記憶を1件取得する。過去の返答が参照した記憶を、IDから辿るために使う。

    訂正・削除済みでも返す。当時どの記憶を渡したかを確認するためで、
    「今は無い記憶を根拠にしていた」ことも分かる必要がある。
    """
    memory = await session.get(Memory, memory_id)
    if memory is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "記憶が見つかりません。")
    return memory


@router.post("/memories", response_model=MemoryOut, status_code=status.HTTP_201_CREATED)
async def add_memory(
    payload: MemoryCreate, session: AsyncSession = Depends(get_session)
) -> Memory:
    return await create_memory(
        session,
        kind=payload.kind.value,
        content=payload.content,
        subject_speaker_id=payload.subject_speaker_id,
        # visible_to_all のときは限定しない（NULL）。既定ではなく、選んだ結果。
        visible_to_speaker_id=payload.visible_to_speaker_id,
        certainty=payload.certainty.value,
        provenance=payload.provenance.value,
        visibility=payload.visibility.value,
        keywords=payload.keywords,
        occurred_at=payload.occurred_at,
        source_message_id=payload.source_message_id,
        source_conversation_id=payload.source_conversation_id,
        reason=payload.reason or "手動で追加",
    )


@router.patch("/memories/{memory_id}", response_model=MemoryOut)
async def correct_memory(
    memory_id: int, payload: MemoryUpdate, session: AsyncSession = Depends(get_session)
) -> Memory:
    """記憶を訂正する。更新前の内容を履歴に残し、元に戻せるようにする。"""
    memory = await session.get(Memory, memory_id)
    if memory is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "記憶が見つかりません。")

    before = snapshot(memory)
    changes = payload.model_dump(exclude_unset=True, exclude={"reason"})
    if not changes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "変更する項目がありません。")

    # 参照範囲は「限定しない」へ戻す指定があるため、他の項目と分けて扱う。
    scope_to_all = changes.pop("visible_to_all", None)
    if scope_to_all and changes.get("visible_to_speaker_id") is not None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "visible_to_speaker_id と visible_to_all は同時に指定できません。",
        )
    if scope_to_all:
        memory.visible_to_speaker_id = None
        changes.pop("visible_to_speaker_id", None)

    for field, value in changes.items():
        # 空にできるのは日時だけ。他の項目を None にする指定は無視する。
        if value is None and field != "occurred_at":
            continue
        # kind・certainty・visibility は Enum で返るため、保存する値を取り出す。
        setattr(memory, field, value.value if isinstance(value, Enum) else value)

    await session.flush()
    record_revision(
        session, memory, action="corrected", before=before, reason=payload.reason or "訂正"
    )
    # この記憶を根拠にした関心・関係性へ、再評価の印を付ける（ISSUE-016）。
    # 自動では直さない。訂正が派生先にも及ぶかは開発者が判断する。
    await mark_for_review(
        session, memory_id=memory.id, reason=f"根拠にした記憶 #{memory.id} が訂正された"
    )
    return memory


@router.delete("/memories/{memory_id}", response_model=MemoryOut)
async def delete_memory(
    memory_id: int,
    reason: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> Memory:
    """記憶を無効にする。行は残し、履歴から元に戻せるようにする。"""
    memory = await session.get(Memory, memory_id)
    if memory is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "記憶が見つかりません。")
    if memory.status == MemoryStatus.DELETED.value:
        return memory

    before = snapshot(memory)
    memory.status = MemoryStatus.DELETED.value
    await session.flush()
    record_revision(
        session, memory, action="deleted", before=before, reason=reason or "誤りのため削除"
    )
    await mark_for_review(
        session, memory_id=memory.id, reason=f"根拠にした記憶 #{memory.id} が削除された"
    )
    return memory


@router.post("/memories/{memory_id}/restore", response_model=MemoryOut)
async def restore_memory(
    memory_id: int,
    revision_id: int | None = None,
    session: AsyncSession = Depends(get_session),
) -> Memory:
    """指定した変更履歴の状態へ戻す。省略時は直前の状態へ戻す。"""
    memory = await session.get(Memory, memory_id)
    if memory is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "記憶が見つかりません。")

    stmt = select(MemoryRevision).where(MemoryRevision.memory_id == memory_id)
    if revision_id is not None:
        stmt = stmt.where(MemoryRevision.id == revision_id)
    revision = (
        await session.execute(stmt.order_by(MemoryRevision.id.desc()).limit(1))
    ).scalar_one_or_none()
    if revision is None or revision.before is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "戻せる変更履歴がありません。")

    before = snapshot(memory)
    for field, value in revision.before.items():
        if field == "occurred_at":
            # 履歴には ISO 文字列で入っているため datetime へ戻す。NULL も戻す。
            memory.occurred_at = datetime.fromisoformat(value) if value else None
            continue
        setattr(memory, field, value)
    await session.flush()
    record_revision(
        session,
        memory,
        action="restored",
        before=before,
        reason=f"変更履歴 {revision.id} の状態へ復元",
    )
    return memory


@router.get("/memories/{memory_id}/revisions", response_model=list[MemoryRevisionOut])
async def list_revisions(
    memory_id: int, session: AsyncSession = Depends(get_session)
) -> list[MemoryRevision]:
    stmt = (
        select(MemoryRevision)
        .where(MemoryRevision.memory_id == memory_id)
        .order_by(MemoryRevision.id.desc())
    )
    return list((await session.execute(stmt)).scalars())
