"""会話ごとの順序制御。

DB の書き込みロックを保持したまま生成を待つと他の処理が止まるため、
生成前にトランザクションを閉じている。その結果、同じ会話へ同時に発言が
届くと、返答の順序が入れ替わりうる。ここではプロセス内のロックで
会話単位に直列化し、DB ロックとは切り離して順序だけを守る。

設計書「3.2 技術と採用範囲」の会話・発話管理のうち、現在必要な
「同じ会話の処理を重ねない」部分にあたる。いまは単一プロセスのため
プロセス内のロックで足りる。複数プロセスに分ける段階になったら、
DB か外部のロックへ置き換える（ISSUE-007）。YouTube 入力の発話キュー
（設計書 §7 並行トラック）を実装する際に、同じ仕組みへ統合する。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class ConversationLocks:
    """会話 id ごとの asyncio ロック。待ち順は到着順になる。"""

    def __init__(self) -> None:
        self._locks: dict[int, asyncio.Lock] = {}
        self._waiters: dict[int, int] = {}

    @asynccontextmanager
    async def hold(self, conversation_id: int) -> AsyncIterator[None]:
        lock = self._locks.setdefault(conversation_id, asyncio.Lock())
        self._waiters[conversation_id] = self._waiters.get(conversation_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            remaining = self._waiters[conversation_id] - 1
            if remaining <= 0:
                # 誰も待っていない会話のロックは残さない。
                self._waiters.pop(conversation_id, None)
                self._locks.pop(conversation_id, None)
            else:
                self._waiters[conversation_id] = remaining

    def is_busy(self, conversation_id: int) -> bool:
        lock = self._locks.get(conversation_id)
        return lock is not None and lock.locked()


# 会話 API と振り返り API で共有する。
conversation_locks = ConversationLocks()
