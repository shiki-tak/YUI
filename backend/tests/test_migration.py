"""マイグレーションそのものを実行して、既存データの補完を確かめる。

旧形式のデータを入れたDBに alembic upgrade を掛け、参照範囲が補完される
ことを確認する。アプリの経路を通す他のテストでは、この移行は検証できない。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic.config import Config

from alembic import command

BACKEND_ROOT = Path(__file__).resolve().parent.parent

# 前の版（visible_to_speaker_id を持たない）のリビジョン
BEFORE_SCOPE = "044af0bc28b0"

_OLD_DATA = """
insert into speakers (id, source, external_id, display_name, created_at) values
  (1,'local','alice','アリス','2026-09-01 00:00:00'),
  (2,'local','bob','ボブ','2026-09-01 00:00:00');
insert into conversations (id, mode, started_at, ended_at) values
  (1,'local','2026-09-01 00:00:00','2026-09-01 01:00:00'),
  (2,'local','2026-09-01 02:00:00',null);
insert into messages
  (id, conversation_id, speaker_kind, speaker_id, source, content, delivery_state, created_at)
values
  (1,1,'user',1,'local_text','アリスとの内緒の話','completed','2026-09-01 00:10:00'),
  (2,2,'user',2,'local_text','ボブとの話','completed','2026-09-01 02:10:00');
insert into memories
  (id, kind, content, subject_speaker_id, certainty, visibility, status, keywords,
   source_message_id, source_conversation_id, created_at, updated_at)
values
  (1,'experience','アリスと内緒の計画を立てた',null,'fact','private','active','内緒 計画',
   1,1,'2026-09-01 00:20:00','2026-09-01 00:20:00'),
  (2,'fact','公開してよい話',null,'fact','public','active','公開',
   1,1,'2026-09-01 00:21:00','2026-09-01 00:21:00'),
  (3,'experience','手で足した記憶',null,'fact','private','active','手動',
   null,null,'2026-09-01 00:22:00','2026-09-01 00:22:00');
insert into memory_candidates
  (id, conversation_id, kind, content, subject_speaker_id, certainty, visibility,
   keywords, source_message_id, status, created_at)
values
  (1,1,'experience','アリスと決めた未採用の予定',null,'fact','private','予定',1,'pending',
   '2026-09-01 00:30:00'),
  (2,2,'experience','ボブと決めた未採用の予定',null,'fact','private','予定',2,'pending',
   '2026-09-01 02:30:00');
"""


def _alembic_config(db_path: Path) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{db_path}")
    return config


def test_migration_backfills_visibility_scope(tmp_path, monkeypatch):
    db_path = tmp_path / "old.db"
    monkeypatch.setenv("YUI_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    from app.config import get_settings

    get_settings.cache_clear()
    config = _alembic_config(db_path)

    # 参照範囲を持たない版まで上げ、旧形式のデータを入れる。
    command.upgrade(config, BEFORE_SCOPE)
    with sqlite3.connect(db_path) as con:
        con.executescript(_OLD_DATA)

    command.upgrade(config, "head")

    with sqlite3.connect(db_path) as con:
        memories = dict(con.execute("select id, visible_to_speaker_id from memories"))
        candidates = dict(
            con.execute("select id, visible_to_speaker_id from memory_candidates")
        )
        reflections = dict(
            con.execute(
                "select id, reflection_completed_at is not null from conversations"
            )
        )

    # 非公開かつ由来する会話がある記憶・候補は、その会話の相手に紐づく。
    assert memories[1] == 1
    assert candidates[1] == 1
    assert candidates[2] == 2
    # 公開記憶と、由来する会話が無い手動追加の記憶は限定しない。
    assert memories[2] is None
    assert memories[3] is None
    # 終了済みの会話は振り返り完了として扱う。
    assert reflections[1] == 1
    assert reflections[2] == 0

    get_settings.cache_clear()


def _schema(db_path: Path) -> dict[str, tuple[dict, list[str]]]:
    """テーブルごとの列と索引を読み出す。比較できる形にそろえる。"""
    with sqlite3.connect(db_path) as con:
        tables = [
            name
            for (name,) in con.execute(
                "select name from sqlite_master where type='table' "
                "and name not like 'sqlite_%' and name != 'alembic_version'"
            )
        ]
        schema = {}
        for table in tables:
            columns = {
                row[1]: (row[2], row[3]) for row in con.execute(f"pragma table_info('{table}')")
            }
            indexes = sorted(
                row[1]
                for row in con.execute(f"pragma index_list('{table}')")
                if not row[1].startswith("sqlite_")
            )
            schema[table] = (columns, indexes)
    return schema


def test_migrations_match_the_models(tmp_path, monkeypatch):
    """マイグレーションを通した schema と、モデル定義が一致する。

    マイグレーションは手で書いている。列を1つ足し忘れても、テストは
    create_all で作った schema を使うため気づけない。実際に動かす DB は
    マイグレーションで作られるので、その2つがずれていないことを見る。
    """
    import asyncio

    from sqlalchemy.ext.asyncio import create_async_engine

    from app.models import Base

    migrated = tmp_path / "migrated.db"
    monkeypatch.setenv("YUI_DATABASE_URL", f"sqlite+aiosqlite:///{migrated}")
    from app.config import get_settings

    get_settings.cache_clear()
    command.upgrade(_alembic_config(migrated), "head")

    declared = tmp_path / "declared.db"

    async def create_from_models() -> None:
        engine = create_async_engine(f"sqlite+aiosqlite:///{declared}")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(create_from_models())

    assert _schema(migrated) == _schema(declared)

    get_settings.cache_clear()
