"""発話の再生状態。

設計書「6. 会話・自発的行動の流れ」の 7 にあたる。生成しただけの文章と、実際に
話し終えた内容を混同しないため、再生の開始・完了・中断を記録する。

状態は generated → playing → completed／aborted の一方向に進む。すでに
終わった発言への通知は、開発者が聞き直しているだけとみなして記録を動かさない。
1 回目に相手へ届いたときの記録を残すため。

遷移は「いまの状態」を条件に含めた UPDATE で行う。読み込んだオブジェクト上で
判定すると、古い状態を読んだ並行リクエストが終わった記録を上書きできる。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.goal import mark_executed
from app.models import Action, ActionRecord, DeliveryState, Message

# 目標を実際に持ち出した行動。待機・回答・調査では話していない。
_SPOKEN_ACTIONS = (Action.ASK.value, Action.SUGGEST.value)

# その通知を受け付けられる、いまの状態。ここに無い状態からは動かさない。
_ALLOWED_FROM: dict[DeliveryState, tuple[DeliveryState, ...]] = {
    DeliveryState.PLAYING: (DeliveryState.GENERATED,),
    DeliveryState.COMPLETED: (DeliveryState.GENERATED, DeliveryState.PLAYING),
    DeliveryState.ABORTED: (DeliveryState.GENERATED, DeliveryState.PLAYING),
}


async def apply_delivery_state(
    session: AsyncSession,
    message: Message,
    state: DeliveryState,
    *,
    now: datetime,
) -> Message:
    """再生の通知を反映し、確定した発言を返す。

    反映できるかの判定は DB 上で行う。呼び出し側が読み込んだ状態は、
    判定に使わない。
    """
    allowed = _ALLOWED_FROM.get(state)
    if allowed is None:
        raise ValueError(f"再生の通知として受け付けない状態です: {state}")

    values: dict[str, Any] = {"delivery_state": state.value}
    if state is DeliveryState.PLAYING:
        values["delivery_started_at"] = now
    else:
        # 再生を始める前に止めた場合、開始時刻は入れない。鳴っていない音声を
        # 「再生した」ことにしないため。
        values["delivery_finished_at"] = now

    result = await session.execute(
        update(Message)
        .where(
            Message.id == message.id,
            Message.delivery_state.in_([s.value for s in allowed]),
        )
        .values(**values)
        # 条件の評価を Python 側で行わせない。判定は SQL に任せる。
        .execution_options(synchronize_session=False)
    )
    # 実際に状態が動いたか。**動いた通知だけを「届いた」として数える。**
    # 現在の状態と比べると、聞き直し（すでに completed への completed）でも
    # 一致してしまい、実行の記録が増える。
    advanced = (result.rowcount or 0) > 0

    # **状態の更新と目標の更新を、同じトランザクションで確定する。**
    # 先に状態だけ commit すると、目標の更新に失敗したときに状態が終端に
    # なってしまい、同じ通知を送り直しても advanced が偽になって永久に
    # 反映されない（第1回レビューの指摘1）。
    if advanced:
        # **判定の前に、DB の確定値を読み直す。** 条件付き UPDATE は DB の最新
        # 状態を見て成功するが、手元のオブジェクトは読み込んだときのままである
        # （synchronize_session=False）。開始の通知と中断の通知が重なると、
        # 開始時刻が入っているのに「鳴り始めていない」と判定して実行の記録を
        # 落とす（第2回レビューの指摘1）。commit はまだしない。原子性を保つ。
        await session.refresh(message)
        await _mark_goals_executed(session, message, state=state, now=now)
    await session.commit()

    # 反映できたかに関わらず、DB 上の確定した状態を返す。
    await session.refresh(message)
    return message


async def _mark_goals_executed(
    session: AsyncSession, message: Message, *, state: DeliveryState, now: datetime
) -> None:
    """その発言が持ち出した目標を、実行済みにする（ISSUE-011）。

    **生成しただけでは実行済みにしない。** 届いたと言えるのは次の2つ。

    - completed：最後まで届いた。
    - aborted で、**再生が始まっていた**もの。途中までは届いている。
      鳴り始める前に止めた場合（generated → aborted）は届いていないので
      数えない。開始時刻の有無で見分ける（第1回レビューの指摘2）。

    どの目標を持ち出したかは action_records から引く。run_records の
    referenced_goal_ids は「渡した目標」で、待機のときも入るため使わない。
    さらに、**目標を持ち出す行動（ask / suggest）の記録だけ**を見る。待機や
    回答の記録に目標の番号が付いていても、その目標は話していない
    （第1回レビューの指摘4）。
    """
    if state is DeliveryState.COMPLETED:
        delivered = state.value
    elif state is DeliveryState.ABORTED and message.delivery_started_at is not None:
        # 鳴り始めてから止めた。途中までは届いている。
        delivered = state.value
    else:
        # 鳴り始める前の中断は、届いていない。
        return

    stmt = select(ActionRecord).where(
        ActionRecord.message_id == message.id,
        ActionRecord.action.in_(_SPOKEN_ACTIONS),
    )
    for record in (await session.execute(stmt)).scalars():
        await mark_executed(
            session,
            goal_ids=list(record.selected_goal_ids or []),
            delivered=delivered,
            at=now,
        )
