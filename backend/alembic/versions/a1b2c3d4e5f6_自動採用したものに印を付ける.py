"""自動採用したものに印を付ける

開発者の確認を経ずに採用した記憶・状態・目標を、手動の採用と区別する
（フェーズ4 PR11、設計書の実装内容9）。

**区別できないと、まとめて戻せない。** 設計書は「更新前後を比較し、評価できた
種類から自動採用へ移す」としており、移した結果が悪ければ戻せることが前提に
なる。履歴（revisions）からも辿れるが、一覧の表示で毎回引き直すのは重い。

Revision ID: a1b2c3d4e5f6
Revises: fb349a94d0fb
Create Date: 2026-09-09 10:30:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "fb349a94d0fb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLES = ("memories", "character_states", "goals")


def upgrade() -> None:
    for table in _TABLES:
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.add_column(
                sa.Column(
                    "auto_adopted", sa.Boolean(), server_default="0", nullable=False
                )
            )


def downgrade() -> None:
    for table in _TABLES:
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_column("auto_adopted")
