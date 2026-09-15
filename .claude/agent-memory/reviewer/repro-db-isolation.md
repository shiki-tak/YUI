---
name: repro-db-isolation
description: 再現スクリプトで alembic や app.db を使うときは YUI_DATABASE_URL を temp に向けないと cwd の backend/data/yui.db に書く
metadata:
  type: feedback
---

再現コードで `alembic` コマンドや `app.db` を直接使うときは、必ず環境変数
`YUI_DATABASE_URL=sqlite+aiosqlite:////<scratchpad>/x.db` を付けて実行する。
`Config.set_main_option("sqlalchemy.url", ...)` だけでは効かない。

**Why:** `backend/alembic/env.py` と `app/db.py` は `get_settings().database_url`
（既定 `./data/yui.db`、cwd 相対）を使う。2026-09-13 のレビューで、URL を
Config に入れただけの upgrade/downgrade が worktree の `backend/data/yui.db` を
新規作成してしまった（gitignore 済みで実害なし。`.env` があれば実 DB に当たる）。

**How to apply:** pytest の `session_factory` fixture 経由なら不要（tmp_path を使う）。
alembic を直接呼ぶ再現・比較スクリプトを書くときだけ、env 変数と検査先を同じ
パスにする。`tests/test_migration.py` が同じ手順（monkeypatch.setenv ＋
`get_settings.cache_clear()`）を踏んでいる。
