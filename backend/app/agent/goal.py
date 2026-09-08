"""目標：次に何を話したいか（設計書 5「目標」・6、フェーズ4）。

記憶（あったこと）とも、関心・関係性（いまどう思っているか）とも分ける。
目標だけが実行条件と期限を持ち、**実行したことと達成したことを別に記録する**。
質問を投げても相手が答えなければ、実行済みで未達成である。

保存の形は character_state.py を写す。候補を出し、開発者が採用し、根拠と変更
履歴を残す。根拠にした記憶が変わったら再評価の印を付け、印が付いている間は
行動選択へ渡さない（ISSUE-016）。

このモジュールは目標を「保存し、渡してよいものを選ぶ」までを持つ。実際に
どの行動を選ぶか、いつ完了にするかはフェーズ4の後続で足す。
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    ConversationMode,
    Goal,
    GoalRevision,
    GoalStatus,
    GoalTrigger,
    Visibility,
    utcnow,
)

TRIGGER_LABEL = {
    GoalTrigger.NEXT_CONVERSATION.value: "次に話すとき",
    GoalTrigger.AFTER_DATE.value: "指定した日以降",
}


def as_utc(value: datetime) -> datetime:
    """日時を UTC へそろえる。

    SQLite は timezone を落として保存・比較する。地域時刻のまま渡すと、その
    地域の壁時計の値が UTC として扱われ、比べる相手とずれる。保存する側だけ
    そろえても足りない。**比較に使う値も同じ基準へ直す**（第1回レビューの
    指摘1：同じ瞬間を JST で渡すと実行してよい判定が変わった）。

    timezone を持たない値は UTC として解釈する。保存が UTC なので、読み戻した
    値をそのまま渡しても同じ基準になる（config.to_local と同じ約束）。
    """
    return value.astimezone(UTC) if value.tzinfo is not None else value.replace(tzinfo=UTC)


def normalize_due_at(trigger: str, due_at: datetime | None) -> datetime | None:
    """実行条件と期限の組み合わせを確かめ、保存する形へ直す。

    after_date は基準の日時が無いと、いつ実行してよいのかが決まらない。逆に
    next_conversation へ日時を渡されても使い道が無く、渡した側は効くつもりで
    いる。どちらも黙って受け取らない。

    判定をここ1か所に置くのは、目標を作る経路が複数あるため（開発者が画面から
    作る／振り返りが候補として作る）。経路ごとに書くと、後から足した経路が
    検査を通らない。

    保存は UTC に揃える。SQLite は timezone を落として保存するため、地域時刻の
    まま渡すと、読み戻したときに UTC として解釈されて9時間ずれる。
    """
    if trigger not in {GoalTrigger.NEXT_CONVERSATION.value, GoalTrigger.AFTER_DATE.value}:
        raise ValueError(f"実行条件が不正です：{trigger}")
    if trigger == GoalTrigger.AFTER_DATE.value:
        if due_at is None:
            raise ValueError("実行条件が after_date のときは due_at が要ります。")
    elif due_at is not None:
        raise ValueError("実行条件が next_conversation のときは due_at を指定できません。")

    if due_at is None:
        return None
    return as_utc(due_at)


def snapshot(goal: Goal) -> dict:
    """変更履歴に残す内容。

    実行と達成の両方を含める。「いつ質問して、いつ答えを得たか」を後から
    追えるようにするため（完了条件5）。
    """
    return {
        "content": goal.content,
        "subject_speaker_id": goal.subject_speaker_id,
        "trigger": goal.trigger,
        "due_at": goal.due_at.isoformat() if goal.due_at else None,
        "basis_memory_ids": list(goal.basis_memory_ids or []),
        "basis_is_provisional": goal.basis_is_provisional,
        "needs_review": goal.needs_review,
        "review_reason": goal.review_reason,
        "status": goal.status,
        "visibility": goal.visibility,
        "visible_to_speaker_id": goal.visible_to_speaker_id,
        "last_executed_at": goal.last_executed_at.isoformat() if goal.last_executed_at else None,
        "completed_at": goal.completed_at.isoformat() if goal.completed_at else None,
    }


def record_revision(
    session: AsyncSession,
    goal: Goal,
    *,
    action: str,
    before: dict | None,
    reason: str | None = None,
) -> GoalRevision:
    revision = GoalRevision(
        goal_id=goal.id, action=action, before=before, after=snapshot(goal), reason=reason
    )
    session.add(revision)
    return revision


async def create_goal(
    session: AsyncSession,
    *,
    content: str,
    subject_speaker_id: int | None = None,
    trigger: str = GoalTrigger.NEXT_CONVERSATION.value,
    due_at: datetime | None = None,
    basis_memory_ids: list[int] | None = None,
    basis_is_provisional: bool = False,
    visibility: str = Visibility.PRIVATE.value,
    visible_to_speaker_id: int | None = None,
    source_conversation_id: int | None = None,
    status: str = GoalStatus.PENDING.value,
    reason: str | None = None,
) -> Goal:
    """目標を作る。既定は候補（pending）で、まだ行動選択には渡さない。"""
    goal = Goal(
        content=content,
        subject_speaker_id=subject_speaker_id,
        trigger=trigger,
        due_at=normalize_due_at(trigger, due_at),
        basis_memory_ids=basis_memory_ids or None,
        basis_is_provisional=basis_is_provisional,
        visibility=visibility,
        visible_to_speaker_id=visible_to_speaker_id,
        source_conversation_id=source_conversation_id,
        status=status,
    )
    session.add(goal)
    await session.flush()
    record_revision(session, goal, action="created", before=None, reason=reason)
    return goal


async def list_goals(
    session: AsyncSession,
    *,
    status: str | None = None,
    needs_review: bool | None = None,
    subject_speaker_id: int | None = None,
) -> list[Goal]:
    stmt = select(Goal).order_by(Goal.id.desc())
    if status is not None:
        stmt = stmt.where(Goal.status == status)
    if needs_review is not None:
        stmt = stmt.where(Goal.needs_review.is_(needs_review))
    if subject_speaker_id is not None:
        stmt = stmt.where(Goal.subject_speaker_id == subject_speaker_id)
    return list((await session.execute(stmt)).scalars())


async def active_goals(
    session: AsyncSession,
    *,
    speaker_id: int | None,
    mode: str = ConversationMode.LOCAL,
    now: datetime | None = None,
) -> list[Goal]:
    """いま実行してよい目標。

    採用済み（active）で、再評価の印が付いておらず、実行条件を満たすものだけ。
    候補のまま行動に使わないのは、記憶・状態と同じ理由による。

    相手が決まっている目標は、その相手との会話でだけ渡す。別の相手に
    「この前の映画どうだった？」と聞けば、話していない人の予定を持ち出す
    ことになる（設計書 3C）。相手が決まらない目標は誰との会話でも渡す。

    再評価の印が付いたものは渡さない。根拠が変わったまま実行すると、訂正が
    反映されていない前提で質問することになる（ISSUE-016）。
    """
    # 比較に使う時刻も UTC へそろえる。保存側だけそろえても、渡された時刻が
    # 地域時刻のままだと、同じ瞬間でも実行してよいかの判定が変わる。
    now = as_utc(now) if now is not None else utcnow()
    stmt = select(Goal).where(
        Goal.status == GoalStatus.ACTIVE.value,
        Goal.needs_review.is_(False),
    )
    if mode == ConversationMode.STREAM:
        stmt = stmt.where(Goal.visibility == Visibility.PUBLIC.value)
    else:
        scope = Goal.visible_to_speaker_id.is_(None)
        if speaker_id is not None:
            scope = or_(scope, Goal.visible_to_speaker_id == speaker_id)
        stmt = stmt.where(scope)

    if speaker_id is not None:
        stmt = stmt.where(
            or_(Goal.subject_speaker_id.is_(None), Goal.subject_speaker_id == speaker_id)
        )
    else:
        stmt = stmt.where(Goal.subject_speaker_id.is_(None))

    # 実行条件。判定は SQL で行う。SQLite は timezone を落として返すため、
    # 読み戻した値を Python 側で比べると、実行環境の時刻として解釈される。
    stmt = stmt.where(
        or_(
            Goal.trigger == GoalTrigger.NEXT_CONVERSATION.value,
            and_(Goal.trigger == GoalTrigger.AFTER_DATE.value, Goal.due_at <= now),
        )
    )
    return list((await session.execute(stmt.order_by(Goal.id))).scalars())


async def mark_for_review(
    session: AsyncSession,
    *,
    memory_id: int,
    reason: str,
    source_conversation_id: int | None = None,
) -> list[Goal]:
    """根拠にした記憶が変わった目標へ、再評価の印を付ける（ISSUE-016）。

    character_state.mark_for_review と同じ規則で拾う。フェーズ3で3回続けて
    指摘された穴が、目標でもそのまま当てはまるため。

    - 記憶IDを根拠に持つ目標。
    - 根拠を持たない目標と、暫定の根拠しか持たない目標。振り返りが作った候補は
      記憶がまだ採用されておらず、記憶IDを持てない。会話単位は粗いが、印は
      行動選択へ渡さなくするだけで、開発者が確認すれば戻せる。
    - 撤回中（withdrawn）の目標。撤回している間に根拠が変わり、そのまま有効へ
      戻すと、古い内容が印なしで実行される。

    終わった目標（done・rejected・cancelled・expired）は対象にしない。もう
    実行されないため、印を付けても行動は変わらない。達成済みの目標の根拠が
    訂正された場合にどう扱うかは、完了・取消の扱いと一緒に決める。

    自動では消さない。訂正が正しいのか、目標そのものを取り消すべきなのかは
    開発者が判断する。
    """
    stmt = select(Goal).where(
        Goal.status.in_(
            [
                GoalStatus.ACTIVE.value,
                GoalStatus.PENDING.value,
                GoalStatus.WITHDRAWN.value,
            ]
        ),
    )
    affected: list[Goal] = []
    for goal in (await session.execute(stmt)).scalars():
        by_memory = memory_id in (goal.basis_memory_ids or [])
        by_conversation = (
            (not goal.basis_memory_ids or goal.basis_is_provisional)
            and source_conversation_id is not None
            and goal.source_conversation_id == source_conversation_id
        )
        if not (by_memory or by_conversation):
            continue
        before = snapshot(goal)
        goal.needs_review = True
        goal.review_reason = reason
        await session.flush()
        record_revision(session, goal, action="needs_review", before=before, reason=reason)
        affected.append(goal)
    return affected


async def mark_executed(
    session: AsyncSession,
    *,
    goal_ids: list[int],
    delivered: str,
    at: datetime | None = None,
) -> list[Goal]:
    """質問が相手へ届いたので、実行済みにする（[ISSUE-011](docs/issues/issues.md)）。

    **生成しただけでは実行済みにしない。** 質問文を作っても音声が鳴らなければ、
    相手は一度も聞いていない。それを実行済みにすると、繰り返し防止の裏返しで
    「相手は聞いていないのに、二度と聞かれない」ことになる。設計書 6 も
    「質問文を生成しただけで、相手から感想を聞けたとは記録しない」と書いている。

    中断（aborted）も実行済みにする。途中までは届いているためで、単純に
    除外すると逆に不正確になる。ただし**どこまで届いたかは履歴に残す**。
    完了と中断を混ぜない。

    実行と達成は別である。ここで入れるのは last_executed_at だけで、
    completed_at は相手の返答を受けてから入れる（PR9）。
    """
    if not goal_ids:
        return []
    at = at or utcnow()
    stmt = select(Goal).where(
        Goal.id.in_(goal_ids),
        # 終わった目標は実行済みにしない。届いた通知が遅れて来ることがある。
        Goal.status.in_([GoalStatus.ACTIVE.value, GoalStatus.WITHDRAWN.value]),
    )
    executed: list[Goal] = []
    for goal in (await session.execute(stmt)).scalars():
        before = snapshot(goal)
        goal.last_executed_at = at
        await session.flush()
        record_revision(
            session,
            goal,
            action="executed",
            before=before,
            reason=f"相手へ届いた（再生: {delivered}）",
        )
        executed.append(goal)
    return executed
