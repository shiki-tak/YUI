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

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DeliveryState, Message

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

    await session.execute(
        update(Message)
        .where(
            Message.id == message.id,
            Message.delivery_state.in_([s.value for s in allowed]),
        )
        .values(**values)
        # 条件の評価を Python 側で行わせない。判定は SQL に任せる。
        .execution_options(synchronize_session=False)
    )
    await session.commit()

    # 反映できたかに関わらず、DB 上の確定した状態を返す。
    await session.refresh(message)
    return message
