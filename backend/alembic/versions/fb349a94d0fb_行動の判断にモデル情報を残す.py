"""行動の判断にモデル情報を残す

action_records に、判断へ使ったモデル・版・生成設定と、読み取り失敗による
待機かどうかを足す（フェーズ4 PR7 の第2回レビュー）。

発言を伴わない判断は run_records が作られないため、ここに残さないと、どの
モデル・設定が待機を選んだのかを後から追えない。モデルを呼ばずに決めた判断は
NULL のままにして、呼んだ判断と区別する。

is_fallback は、出力を読み取れずに待機へ倒した場合の印。正常な待機の判断と
区別できないと、適切に待てた回数を数えられない。

Revision ID: fb349a94d0fb
Revises: 24c03fb94c47
Create Date: 2026-09-08 14:05:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "fb349a94d0fb"
down_revision: Union[str, None] = "24c03fb94c47"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("action_records", schema=None) as batch_op:
        batch_op.add_column(sa.Column("provider", sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column("model", sa.String(length=128), nullable=True))
        batch_op.add_column(sa.Column("model_digest", sa.String(length=128), nullable=True))
        batch_op.add_column(sa.Column("options", sa.JSON(), nullable=True))
        batch_op.add_column(
            sa.Column("is_fallback", sa.Boolean(), server_default="0", nullable=False)
        )


def downgrade() -> None:
    with op.batch_alter_table("action_records", schema=None) as batch_op:
        batch_op.drop_column("is_fallback")
        batch_op.drop_column("options")
        batch_op.drop_column("model_digest")
        batch_op.drop_column("model")
        batch_op.drop_column("provider")
