"""音声合成の実行記録と記憶検索の時間を追加する

設計書フェーズ2の実装内容5「送信から音声の再生開始までの時間を測り、待ち時間の
大きい部分を確認する」のために、区間ごとの時間を残せるようにする。

- speech_runs：合成用データの作成と音声生成を分けて記録する。聞き直すたびに
  1 行増やし、同じ文章の合成にかかる時間の変化を見られるようにする。
- run_records.retrieval_ms：記憶検索の時間。生成の時間と分ける。

Revision ID: d24b3a0aa761
Revises: a71c525f0285
Create Date: 2026-09-07 02:46:26.861109
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d24b3a0aa761"
down_revision: Union[str, None] = "a71c525f0285"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "speech_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("speaker_id", sa.Integer(), nullable=False),
        sa.Column("engine_version", sa.String(length=64), nullable=True),
        sa.Column("query_ms", sa.Integer(), nullable=True),
        sa.Column("synthesis_ms", sa.Integer(), nullable=True),
        sa.Column("audio_ms", sa.Integer(), nullable=True),
        sa.Column("byte_size", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("speech_runs", schema=None) as batch_op:
        batch_op.create_index(
            "ix_speech_runs_message", ["message_id", "id"], unique=False
        )

    with op.batch_alter_table("run_records", schema=None) as batch_op:
        batch_op.add_column(sa.Column("retrieval_ms", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("run_records", schema=None) as batch_op:
        batch_op.drop_column("retrieval_ms")

    with op.batch_alter_table("speech_runs", schema=None) as batch_op:
        batch_op.drop_index("ix_speech_runs_message")

    op.drop_table("speech_runs")
