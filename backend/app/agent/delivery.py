"""発話の再生状態。

設計書「5. 1回の会話の流れ」の 8 にあたる。生成しただけの文章と、実際に
話し終えた内容を混同しないため、再生の開始・完了・中断を記録する。

状態は generated → playing → completed／aborted の一方向に進む。すでに
終わった発言への通知は、開発者が聞き直しているだけとみなして記録を動かさない。
1 回目に相手へ届いたときの記録を残すため。
"""

from __future__ import annotations

from datetime import datetime

from app.models import DeliveryState, Message

_FINISHED = {DeliveryState.COMPLETED.value, DeliveryState.ABORTED.value}


def apply_delivery_state(message: Message, state: DeliveryState, *, now: datetime) -> bool:
    """再生の通知を反映する。記録が変わったら True を返す。"""
    if message.delivery_state in _FINISHED:
        # 聞き直し。実際に話し終えた 1 回目の記録を上書きしない。
        return False

    if state is DeliveryState.PLAYING:
        if message.delivery_state == DeliveryState.PLAYING.value:
            return False
        message.delivery_state = state.value
        message.delivery_started_at = now
        return True

    # 再生を始める前に止めた場合、開始時刻は入れない。鳴っていない音声を
    # 「再生した」ことにしないため。
    message.delivery_state = state.value
    message.delivery_finished_at = now
    return True
