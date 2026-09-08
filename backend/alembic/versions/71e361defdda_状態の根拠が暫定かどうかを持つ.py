"""状態の根拠が暫定かどうかを持つ

状態を採用するとき、その会話から採用済みの記憶を自動で並べて根拠にしている。
しかし後から採用された記憶は入らないため、完全に特定した根拠ではない。区別
しないと、後から採用した記憶を訂正しても再評価の印が届かない（ISSUE-016、
フェーズ3再々レビューの指摘1）。

- character_states.basis_is_provisional：自動で並べた根拠なら真。既存の行は
  偽にする。手で指定した根拠と、自動で並べたものを、後から見分けられないため。

Revision ID: 71e361defdda
Revises: 3187f6b349d3
Create Date: 2026-09-08 10:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "71e361defdda"
down_revision: Union[str, None] = "3187f6b349d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("character_states", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "basis_is_provisional",
                sa.Boolean(),
                server_default="0",
                nullable=False,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("character_states", schema=None) as batch_op:
        batch_op.drop_column("basis_is_provisional")
