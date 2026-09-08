"""行動の判断を記録する

回答・確認質問・話題提案・調査・待機のどれを選んだかを残す（設計書 6 の 3、
フェーズ4の完了条件5）。

**発言を伴わない判断も残すために、run_records とは別の表にする。** 待機を
選ぶと発言が作られず、run_records は message_id を持てないため記録できない。
設計書は「不要な質問、重複、誤った推測、適切な待機も確認する」としており、
待機を後から追えないと、適切に待てたのかを測れない。

Revision ID: 24c03fb94c47
Revises: e9759b8e6cfa
Create Date: 2026-09-08 13:20:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "24c03fb94c47"
down_revision: Union[str, None] = "e9759b8e6cfa"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "action_records",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("speaker_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(length=24), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("candidate_goal_ids", sa.JSON(), nullable=True),
        sa.Column("selected_goal_ids", sa.JSON(), nullable=True),
        sa.Column("message_id", sa.Integer(), nullable=True),
        sa.Column("is_proactive", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["speaker_id"], ["speakers.id"]),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("action_records", schema=None) as batch_op:
        batch_op.create_index(
            "ix_action_records_conversation", ["conversation_id", "id"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("action_records", schema=None) as batch_op:
        batch_op.drop_index("ix_action_records_conversation")
    op.drop_table("action_records")
