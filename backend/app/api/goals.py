"""目標の API（設計書 5・6 / フェーズ4）。

設計書の実装内容4「採用・保留・却下・取消・完了を管理画面で扱う」にあたる。
記憶・可変状態と同じく、候補を開発者が確認してから使う。

api/states.py を写している。参照範囲の検査と、根拠が死んでいないかの確認は
同じ規則で行う。目標だけに要るのは、実行条件と期限の検査（agent 側の
normalize_due_at に集める）と、終わり方の区別である。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.goal import (
    create_goal,
    list_goals,
    normalize_due_at,
    record_revision,
    snapshot,
)
from app.db import get_session
from app.models import (
    Goal,
    GoalRevision,
    GoalStatus,
    GoalTrigger,
    Memory,
    MemoryStatus,
    Visibility,
    utcnow,
)
from app.schemas import (
    GoalCreate,
    GoalDecision,
    GoalOut,
    GoalRevisionOut,
    GoalUpdate,
)

router = APIRouter(tags=["goals"])

# 状態の遷移。ここに無い変更は受け付けない（第1回レビューの指摘2・3）。
#
# - 候補（pending）の採用・却下は decide だけで行う。PATCH から active へ
#   動かせると、根拠の補完と accepted の履歴を通らない採用ができてしまう。
# - 終わった目標は状態を変えられない。達成した目標を有効へ戻せると、
#   完了条件「一度完了した質問・目標を繰り返さない」が状態の上で崩れる。
#   前提が戻ったのなら、新しい目標として作り直す。
# - 撤回（withdrawn）だけは戻せる。開発者が一時的に外すための状態で、
#   戻すときに根拠が生きているかを確かめる。
_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    GoalStatus.PENDING.value: set(),
    GoalStatus.ACTIVE.value: {
        GoalStatus.ACTIVE.value,
        GoalStatus.DONE.value,
        GoalStatus.WITHDRAWN.value,
        GoalStatus.CANCELLED.value,
        GoalStatus.EXPIRED.value,
    },
    GoalStatus.WITHDRAWN.value: {
        GoalStatus.WITHDRAWN.value,
        GoalStatus.ACTIVE.value,
        GoalStatus.CANCELLED.value,
        GoalStatus.EXPIRED.value,
    },
    GoalStatus.DONE.value: set(),
    GoalStatus.REJECTED.value: set(),
    GoalStatus.CANCELLED.value: set(),
    GoalStatus.EXPIRED.value: set(),
}


async def _check_scope_against_basis(session: AsyncSession, payload: GoalCreate) -> None:
    """根拠の記憶より広い参照範囲を許さない。

    記憶側で相手を限定していても、そこから作った目標が別の相手へ渡ると、
    限定した意味が無くなる。「この前の映画どうだった？」を別人に聞く形。
    """
    stmt = select(Memory).where(Memory.id.in_(payload.basis_memory_ids))
    memories = list((await session.execute(stmt)).scalars())
    found = {memory.id for memory in memories}
    missing = [mid for mid in payload.basis_memory_ids if mid not in found]
    if missing:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"根拠にした記憶が見つかりません: {'、'.join(str(m) for m in missing)}",
        )

    dead = [
        memory.id for memory in memories if memory.status != MemoryStatus.ACTIVE.value
    ]
    if dead:
        # 削除・訂正された記憶を根拠にした目標は、作られた時点で古い前提を
        # 持っている。訂正の波及は「後から変わったもの」を拾う仕組みなので、
        # 作る前に無効だった根拠は拾えない（第1回レビューの指摘1）。
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
                f"記憶 #{memory.id} は非公開です。そこから作る目標を公開にはできません。",
            )
        limited_to = memory.visible_to_speaker_id
        if limited_to is not None and payload.visible_to_speaker_id != limited_to:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"記憶 #{memory.id} は相手を限定しています。"
                f"そこから作る目標も、その相手に限定してください"
                f"（visible_to_speaker_id={limited_to}）。",
            )


async def _dead_basis(session: AsyncSession, goal: Goal) -> list[int]:
    """根拠のうち、もう有効でない記憶。"""
    if not goal.basis_memory_ids:
        return []
    stmt = select(Memory).where(
        Memory.id.in_(goal.basis_memory_ids),
        Memory.status == MemoryStatus.ACTIVE.value,
    )
    alive = {memory.id for memory in (await session.execute(stmt)).scalars()}
    return [mid for mid in goal.basis_memory_ids if mid not in alive]


async def _get(session: AsyncSession, goal_id: int) -> Goal:
    goal = await session.get(Goal, goal_id)
    if goal is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "目標が見つかりません。")
    return goal


def _resolve_schedule(
    goal: Goal, *, trigger: str | None, due_at: object, changed: set[str]
) -> None:
    """実行条件と期限を、組み合わせごと検査してから入れる。

    片方だけ変えられると、after_date のまま基準日時を消す、条件を
    next_conversation へ戻したのに期限が残る、といった状態が作れる。判定は
    agent 側の normalize_due_at に集めてあり、ここは同じ関数へ渡すだけにする
    （経路ごとに検査を書くと、後から足した経路が通らない）。
    """
    if not ({"trigger", "due_at"} & changed):
        return
    next_trigger = trigger if trigger is not None else goal.trigger
    next_due = due_at if "due_at" in changed else goal.due_at
    # 条件を「次の会話」へ戻したら、期限は消す。効かない期限が残ったままだと、
    # 画面には日付が出ているのに実行条件は日付を見ない、という状態になる。
    if next_trigger == GoalTrigger.NEXT_CONVERSATION.value and "due_at" not in changed:
        next_due = None
    try:
        goal.due_at = normalize_due_at(next_trigger, next_due)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    goal.trigger = next_trigger


@router.get("/goals", response_model=list[GoalOut])
async def read_goals(
    goal_status: str | None = None,
    needs_review: bool | None = None,
    subject_speaker_id: int | None = None,
    session: AsyncSession = Depends(get_session),
) -> list[Goal]:
    """目標の一覧。

    needs_review=true で、根拠が変わって再評価が要るものだけを見られる。
    """
    return await list_goals(
        session,
        status=goal_status,
        needs_review=needs_review,
        subject_speaker_id=subject_speaker_id,
    )


@router.post("/goals", response_model=GoalOut, status_code=status.HTTP_201_CREATED)
async def add_goal(
    payload: GoalCreate, session: AsyncSession = Depends(get_session)
) -> Goal:
    """目標を候補として追加する。採用するまで行動選択には渡さない。"""
    if payload.basis_memory_ids:
        await _check_scope_against_basis(session, payload)
    try:
        return await create_goal(
            session,
            content=payload.content,
            subject_speaker_id=payload.subject_speaker_id,
            trigger=payload.trigger,
            due_at=payload.due_at,
            basis_memory_ids=payload.basis_memory_ids,
            visibility=payload.visibility.value,
            visible_to_speaker_id=payload.visible_to_speaker_id,
            reason=payload.reason or "手動で追加",
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


@router.post("/goals/{goal_id}/decide", response_model=GoalOut)
async def decide_goal(
    goal_id: int, payload: GoalDecision, session: AsyncSession = Depends(get_session)
) -> Goal:
    """候補を採用または却下する。採用のときに内容と実行条件を直せる。"""
    goal = await _get(session, goal_id)
    if goal.status != GoalStatus.PENDING.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "この目標はすでに判断済みです。")

    before = snapshot(goal)
    if payload.decision == "accept":
        if payload.content:
            goal.content = payload.content
        changed = payload.model_fields_set & {"trigger", "due_at"}
        _resolve_schedule(
            goal, trigger=payload.trigger, due_at=payload.due_at, changed=changed
        )
        # 根拠が会話単位しかない目標に、その会話から採用された記憶を結び付ける。
        # 候補の時点では記憶がまだ採用されておらず、記憶IDを持てないため。
        if not goal.basis_memory_ids and goal.source_conversation_id is not None:
            stmt = select(Memory).where(
                Memory.source_conversation_id == goal.source_conversation_id,
                Memory.status == MemoryStatus.ACTIVE.value,
            )
            basis = [memory.id for memory in (await session.execute(stmt)).scalars()]
            goal.basis_memory_ids = basis or None
            # 自動で並べた根拠は暫定。後から採用される記憶が入らないため、
            # 訂正の波及では会話単位でも拾う。
            goal.basis_is_provisional = True
        goal.status = GoalStatus.ACTIVE.value
        action = "accepted"
        # 候補として置かれてから採用までの間に、根拠が消えていることがある。
        # 拒否はせず印を付ける。目標そのものを捨てるかは開発者が決める
        # （印が付いている間は行動選択へ渡らない）。
        gone = await _dead_basis(session, goal)
        if gone:
            goal.needs_review = True
            goal.review_reason = (
                f"根拠にした記憶 {'、'.join(f'#{mid}' for mid in gone)} が"
                "有効でなくなっている"
            )
    else:
        goal.status = GoalStatus.REJECTED.value
        action = "rejected"
    await session.flush()
    record_revision(session, goal, action=action, before=before, reason=payload.reason)
    return goal


@router.patch("/goals/{goal_id}", response_model=GoalOut)
async def update_goal(
    goal_id: int, payload: GoalUpdate, session: AsyncSession = Depends(get_session)
) -> Goal:
    """採用済みの目標を直す。終わり方の記録と、再評価の印を下ろすのもここ。"""
    goal = await _get(session, goal_id)
    changes = payload.model_dump(exclude_unset=True, exclude={"reason"})
    if not changes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "変更する項目がありません。")

    if payload.status is not None and payload.status not in _ALLOWED_TRANSITIONS[goal.status]:
        if goal.status == GoalStatus.PENDING.value:
            detail = (
                "候補の採用・却下は /goals/{id}/decide で行ってください。"
                "採用のときだけ、根拠の補完と採用の履歴が残ります。"
            )
        else:
            detail = (
                f"終わった目標（{goal.status}）の状態は変えられません。"
                "前提が戻ったのなら、新しい目標として作り直してください。"
            )
        raise HTTPException(status.HTTP_409_CONFLICT, detail)

    before = snapshot(goal)
    if payload.content:
        goal.content = payload.content
    _resolve_schedule(
        goal,
        trigger=payload.trigger,
        due_at=payload.due_at,
        changed=payload.model_fields_set & {"trigger", "due_at"},
    )
    if payload.status is not None:
        goal.status = payload.status
        # 達成した時刻は、実行した時刻とは別に残す（設計書 6）。質問を投げた
        # ことと、相手が答えたことを混ぜない。
        if payload.status == GoalStatus.DONE.value and goal.completed_at is None:
            goal.completed_at = utcnow()
    if payload.reviewed:
        # 確認したので印を下ろす。何を根拠に下ろしたかは履歴に残る。
        goal.needs_review = False
        goal.review_reason = None
    if payload.status == GoalStatus.ACTIVE.value and not payload.reviewed:
        # 有効へ戻すときは、根拠がまだ生きているかを見る。撤回している間に
        # 根拠が消えていることがある。
        gone = await _dead_basis(session, goal)
        if gone:
            goal.needs_review = True
            goal.review_reason = (
                f"根拠にした記憶 {'、'.join(f'#{mid}' for mid in gone)} が"
                "有効でなくなっている"
            )
    await session.flush()
    record_revision(session, goal, action="corrected", before=before, reason=payload.reason)
    return goal


@router.get("/goals/{goal_id}/revisions", response_model=list[GoalRevisionOut])
async def read_revisions(
    goal_id: int, session: AsyncSession = Depends(get_session)
) -> list[GoalRevision]:
    await _get(session, goal_id)
    stmt = (
        select(GoalRevision)
        .where(GoalRevision.goal_id == goal_id)
        .order_by(GoalRevision.id.desc())
    )
    return list((await session.execute(stmt)).scalars())
