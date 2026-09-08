"""DB 接続。SQLite + aiosqlite。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

_settings = get_settings()

sqlite_path = _settings.sqlite_path
if sqlite_path is not None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_async_engine(_settings.database_url, echo=False, future=True)

SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """接続の作り方そのものを渡す依存。

    リクエストの寿命より長く生きる処理（会話後の振り返りジョブ）は、ハンドラの
    セッションを使えない。ハンドラが返した時点で閉じるため。差し替えられる形に
    しておくのは、テストで一時DBへ向けるためでもある。通常利用のDBを、テストが
    書き換えないようにする。
    """
    return SessionLocal


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI の依存。ハンドラが正常終了したらコミットする。"""
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
