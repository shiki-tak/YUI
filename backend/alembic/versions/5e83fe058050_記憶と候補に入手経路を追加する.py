"""記憶と候補に入手経路を追加する

設計書フェーズ3の3C「本人の訂正、他人による伝聞、冗談、人格変更要求を区別する」
のために、その内容をどうやって知ったかを残せるようにする。

- memories.provenance / memory_candidates.provenance：
  firsthand（本人が自分について話した）／hearsay（別の人について話した）／
  unknown（判断できない）。
  既存の行は unknown にする。1対1の会話から作られたものが多いが、本人の発言と
  伝聞のどちらかは行ごとに確かめられない。分からないものを、確かめずに
  firsthand として記録しない。

Revision ID: 5e83fe058050
Revises: 152507a25337
Create Date: 2026-09-07 12:10:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "5e83fe058050"
down_revision: Union[str, None] = "152507a25337"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for table in ("memories", "memory_candidates"):
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.add_column(
                sa.Column(
                    "provenance",
                    sa.String(length=16),
                    nullable=False,
                    server_default="unknown",
                )
            )


def downgrade() -> None:
    for table in ("memories", "memory_candidates"):
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_column("provenance")
