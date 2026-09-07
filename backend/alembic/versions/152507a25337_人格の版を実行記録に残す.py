"""人格の版を実行記録に残す

設計書フェーズ3の3A「人格設定を変更した版を比較し」「採用した定義と例示を
版管理し、以後の継続性評価の基準にする」のために、どの版で生成したかを
残せるようにする。

- run_records.persona_version：生成に使った固定人格の版（personas/<版>.toml）。
  既存の行は、人格をコードに直接書いていた時期のもので版を特定できないため
  NULL のままにする。分からないものを、いまの既定版として記録しない。

Revision ID: 152507a25337
Revises: d24b3a0aa761
Create Date: 2026-09-07 15:33:31.073089
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "152507a25337"
down_revision: Union[str, None] = "d24b3a0aa761"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("run_records", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("persona_version", sa.String(length=64), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("run_records", schema=None) as batch_op:
        batch_op.drop_column("persona_version")
