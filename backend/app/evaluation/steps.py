"""評価シナリオの手順を実行する（ISSUE-006 の通し評価）。

完了条件1・2は、記憶が**作られてから**使われるまでを見る必要がある。記憶を
事前に入れて1会話を流すだけでは、抽出と採用の経路を通らない。ここでは、
振り返り・採用・訂正・再起動を会話の途中に挟めるようにする。

採用と訂正は、API と同じ処理（`create_memory`・`record_revision`・
`mark_for_review`）を呼ぶ。評価のためだけの近道を作ると、実際の経路と
違うものを測ることになる。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.character_state import mark_for_review
from app.agent.memory_store import create_memory, record_revision, snapshot
from app.agent.reflection import extract_candidates
from app.evaluation.scenario import StepSpec
from app.llm.base import LLMClient
from app.models import (
    CandidateStatus,
    Conversation,
    Memory,
    MemoryCandidate,
    MemoryStatus,
)


@dataclass
class ReflectOutcome:
    """振り返りを1回流した結果。"""

    candidates: list[MemoryCandidate] = field(default_factory=list)
    accepted: list[Memory] = field(default_factory=list)
    error: str | None = None


async def run_reflection(
    session: AsyncSession,
    *,
    llm: LLMClient,
    conversation: Conversation,
    character_name: str,
    step: StepSpec,
) -> ReflectOutcome:
    """会話を振り返り、必要なら候補を採用する。

    採用は API と同じ流れで、候補から記憶を作り、候補に採用済みの印を付ける。
    """
    candidates = await extract_candidates(
        session, llm=llm, conversation=conversation, character_name=character_name
    )
    session.add_all(candidates)
    await session.flush()

    outcome = ReflectOutcome(candidates=candidates)
    if not step.accept:
        await session.commit()
        return outcome

    for candidate in candidates:
        if step.accept_contains and not any(
            word in candidate.content for word in step.accept_contains
        ):
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
        )
        candidate.status = CandidateStatus.ACCEPTED.value
        candidate.accepted_memory_id = memory.id
        outcome.accepted.append(memory)

    await session.commit()
    return outcome


async def _find_memory(session: AsyncSession, match: str) -> Memory | None:
    stmt = select(Memory).where(
        Memory.status == MemoryStatus.ACTIVE.value, Memory.content.contains(match)
    )
    return (await session.execute(stmt)).scalars().first()


async def correct_memory(session: AsyncSession, *, match: str, content: str) -> Memory | None:
    """開発者が記憶を訂正する。API と同じく、履歴と波及も起こす。"""
    memory = await _find_memory(session, match)
    if memory is None:
        return None
    before = snapshot(memory)
    memory.content = content
    await session.flush()
    record_revision(session, memory, action="corrected", before=before, reason="評価用会話で訂正")
    await mark_for_review(
        session,
        memory_id=memory.id,
        reason=f"根拠にした記憶 #{memory.id} が訂正された",
        source_conversation_id=memory.source_conversation_id,
    )
    await session.commit()
    return memory


async def delete_memory(session: AsyncSession, *, match: str) -> Memory | None:
    """開発者が記憶を削除する。"""
    memory = await _find_memory(session, match)
    if memory is None:
        return None
    before = snapshot(memory)
    memory.status = MemoryStatus.DELETED.value
    await session.flush()
    record_revision(session, memory, action="deleted", before=before, reason="評価用会話で削除")
    await mark_for_review(
        session,
        memory_id=memory.id,
        reason=f"根拠にした記憶 #{memory.id} が削除された",
        source_conversation_id=memory.source_conversation_id,
    )
    await session.commit()
    return memory
