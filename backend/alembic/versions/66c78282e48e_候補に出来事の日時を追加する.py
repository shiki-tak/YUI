"""候補に出来事の日時を追加する

設計書フェーズ3の3B「好みや予定の訂正後、古い情報を現在の事実として回答しない」
のために、いつの出来事かを候補の段階から持てるようにする（ISSUE-004）。

- memory_candidates.occurred_at：会話に日付の手がかりがある場合だけ入る。
  好みや性格のように、いつのことか決まらない内容は NULL のまま。
  既存の行も NULL のままにする（作られた時点で読み取っていないため）。

memories.occurred_at は初期スキーマからある。これまで手動のAPIからしか
入らなかったものを、振り返りからも入るようにする。

Revision ID: 66c78282e48e
Revises: 94cd25a867e0
Create Date: 2026-09-07 13:05:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "66c78282e48e"
down_revision: Union[str, None] = "94cd25a867e0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("memory_candidates", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("memory_candidates", schema=None) as batch_op:
        batch_op.drop_column("occurred_at")
