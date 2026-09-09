"""自動採用したものを見て、まとめて戻す（フェーズ4 PR11）。

設計書は「更新前後を比較し、評価できた種類から自動採用へ移す」としている。
移した結果が悪ければ**戻せること**が前提なので、戻す経路をここに置く。

戻すのは自動採用したものだけで、手動で採用したものには触れない。区別は
`auto_adopted` の印で行う。

**すでに効果が出たものは戻さない。** 達成した目標や、訂正・削除された記憶を
戻すと、履歴のほうを書き換えることになる。何を戻せなかったかは応答に残し、
黙って取りこぼさない。
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.character_state import record_revision as record_state_revision
from app.agent.character_state import snapshot as state_snapshot
from app.agent.derived import mark_derived_for_review
from app.agent.goal import as_utc
from app.agent.goal import record_revision as record_goal_revision
from app.agent.goal import snapshot as goal_snapshot
from app.agent.memory_store import record_revision as record_memory_revision
from app.agent.memory_store import snapshot as memory_snapshot
from app.db import get_session
from app.models import (
    CharacterState,
    Goal,
    GoalStatus,
    Memory,
    MemoryRevision,
    MemoryStatus,
    StateStatus,
)
from app.schemas import AutoAdoptedOut, RollbackResult, RollbackSkipped

router = APIRouter(tags=["auto-adopt"])

_ROLLBACK_REASON = "自動採用の取り消し"

# 採用そのものの履歴。これ以外の履歴があれば、開発者が後から手を入れている。
_ADOPTION_ACTIONS = frozenset({"created", "accepted_with_edits"})


async def _was_edited(session: AsyncSession, memory: Memory) -> bool:
    """採用の後で、開発者が手を入れたか（レビューの指摘3）。

    訂正しても status は active のままなので、状態では見分けられない。履歴に
    採用以外の記録があるかで見る。**直して残したものを取り消すと、開発者の
    判断まで巻き込む。**
    """
    stmt = select(MemoryRevision.action).where(MemoryRevision.memory_id == memory.id)
    actions = set((await session.execute(stmt)).scalars())
    return bool(actions - _ADOPTION_ACTIONS)


def _row(kind: str, item: Memory | CharacterState | Goal) -> AutoAdoptedOut:
    return AutoAdoptedOut(
        table=kind,
        id=item.id,
        kind=getattr(item, "kind", "goal"),
        content=item.content,
        status=item.status,
        created_at=item.created_at,
    )


async def _auto_adopted(
    session: AsyncSession, since: datetime | None
) -> tuple[list[Memory], list[CharacterState], list[Goal]]:
    """自動採用したもの。`since` があればそれ以降に作られたものだけ。

    **`since` は UTC へそろえてから比べる**（レビューの指摘4）。SQLite は
    timezone を落として比較するため、地域時刻のまま渡すと、その壁時計の値が
    UTC として扱われる。同じ瞬間を JST で渡すと対象が変わっていた。
    保存側と同じ `as_utc` を通す。
    """
    since_utc = as_utc(since).replace(tzinfo=None) if since is not None else None
    result = []
    for model in (Memory, CharacterState, Goal):
        stmt = select(model).where(model.auto_adopted.is_(True))
        if since_utc is not None:
            stmt = stmt.where(model.created_at >= since_utc)
        result.append(list((await session.execute(stmt.order_by(model.id))).scalars()))
    return result[0], result[1], result[2]


@router.get("/auto-adopt", response_model=list[AutoAdoptedOut])
async def list_auto_adopted(
    since: datetime | None = None,
    session: AsyncSession = Depends(get_session),
) -> list[AutoAdoptedOut]:
    """自動採用したものの一覧。戻す前に何が入ったかを見るために使う。"""
    memories, states, goals = await _auto_adopted(session, since)
    return (
        [_row("memory", item) for item in memories]
        + [_row("state", item) for item in states]
        + [_row("goal", item) for item in goals]
    )


@router.post("/auto-adopt/rollback", response_model=RollbackResult)
async def rollback(
    since: datetime | None = None,
    session: AsyncSession = Depends(get_session),
) -> RollbackResult:
    """自動採用したものをまとめて戻す。

    **戻すのは、まだ効果が出ていないものだけ。** 達成した目標や、訂正・削除
    された記憶に触れると、そちらの履歴を書き換えることになる。戻せなかった
    ものは `skipped` に理由とともに残す。**黙って取りこぼさない。**

    記憶は削除の印を付けるだけで、行は消さない（既存の復元の経路で戻せる）。
    """
    memories, states, goals = await _auto_adopted(session, since)
    reverted: list[AutoAdoptedOut] = []
    skipped: list[RollbackSkipped] = []

    for memory in memories:
        if memory.status != MemoryStatus.ACTIVE.value:
            skipped.append(
                RollbackSkipped(
                    table="memory",
                    id=memory.id,
                    reason=f"すでに {memory.status} になっています。",
                )
            )
            continue
        if await _was_edited(session, memory):
            # **開発者が直して残したものは取り消さない**（レビューの指摘3）。
            # 訂正しても status は active のままなので、状態では見分けられない。
            # 自動採用の出自と、いま取り消してよいかは別の判断である。
            skipped.append(
                RollbackSkipped(
                    table="memory",
                    id=memory.id,
                    reason="採用後に開発者が手を入れています。",
                )
            )
            continue
        before = memory_snapshot(memory)
        memory.status = MemoryStatus.DELETED.value
        await session.flush()
        record_memory_revision(
            session, memory, action="deleted", before=before, reason=_ROLLBACK_REASON
        )
        # **削除は派生物へ波及させる**（レビューの指摘1）。既存の削除APIが
        # 守っている経路を、新しい削除経路が迂回していた。取り消した記憶が
        # 目標や関心を介して使われ続ける。
        await mark_derived_for_review(
            session,
            memory_id=memory.id,
            reason=f"根拠にした記憶 #{memory.id} の自動採用を取り消した",
            source_conversation_id=memory.source_conversation_id,
        )
        reverted.append(_row("memory", memory))

    for state in states:
        if state.status != StateStatus.ACTIVE.value:
            skipped.append(
                RollbackSkipped(
                    table="state", id=state.id, reason=f"すでに {state.status} になっています。"
                )
            )
            continue
        before = state_snapshot(state)
        state.status = StateStatus.WITHDRAWN.value
        record_state_revision(
            session, state, action="withdrawn", before=before, reason=_ROLLBACK_REASON
        )
        reverted.append(_row("state", state))

    for goal in goals:
        if goal.status != GoalStatus.ACTIVE.value:
            # 達成・取消・期限切れは、戻すと履歴のほうを書き換えることになる。
            skipped.append(
                RollbackSkipped(
                    table="goal", id=goal.id, reason=f"すでに {goal.status} になっています。"
                )
            )
            continue
        if goal.last_executed_at is not None:
            # 一度でも相手へ持ち出した目標。取り下げても、聞いた事実は消えない。
            skipped.append(
                RollbackSkipped(
                    table="goal", id=goal.id, reason="すでに相手へ持ち出しています。"
                )
            )
            continue
        before = goal_snapshot(goal)
        goal.status = GoalStatus.WITHDRAWN.value
        record_goal_revision(
            session, goal, action="withdrawn", before=before, reason=_ROLLBACK_REASON
        )
        reverted.append(_row("goal", goal))

    await session.flush()
    return RollbackResult(reverted=reverted, skipped=skipped)
