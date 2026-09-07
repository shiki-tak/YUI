"""候補に近い記憶の控えを追加する

設計書フェーズ3の3C「同一イベントの重複保存を防ぎ、反復を独立した経験として
数えない」のために、候補が既存のどの記憶と近いかを残せるようにする。

- memory_candidates.similar_memory_ids：内容が近い既存の記憶のID。自動では
  捨てず、採用を判断するときに見せる。既存の行は NULL のままにする
  （作られた時点で照合していないため、後から付けても根拠がない）。

Revision ID: 94cd25a867e0
Revises: 5e83fe058050
Create Date: 2026-09-07 12:40:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "94cd25a867e0"
down_revision: Union[str, None] = "5e83fe058050"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("memory_candidates", schema=None) as batch_op:
        batch_op.add_column(sa.Column("similar_memory_ids", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("memory_candidates", schema=None) as batch_op:
        batch_op.drop_column("similar_memory_ids")
