"""会話状態のテーブルを追加する

v0.2「会話理解と修復」のための器（docs/plan/v0.2.md 3節）。

- conversation_states：会話ごとの用件・未回答の質問・提示済みの内容・確認済み・
  延期・終了の意思・食い違い・訂正を、9種の kind で1つの表に持つ。
- 発言があったこと（responded_message_id）と解決したこと（status='resolved'）を
  分けて持つ。解決・取消は解釈（LLM）か開発者の操作でだけ起き、規則は作成と
  応答の記録しかしない（この版の PR1 では規則しか動かさない）。
- target_speaker_id で「誰に適用するか」を持ち、別の相手の発言では遷移しない。
- 変更履歴の表は作らない。遷移は1段で、根拠・確認・応答・解消の発言 ID を
  この行が持つため、後から辿れる。

Revision ID: 836089fcf4d9
Revises: f76fb885504f
Create Date: 2026-09-13 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "836089fcf4d9"
down_revision: Union[str, None] = "f76fb885504f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "conversation_states",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=24), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("speaker_id", sa.Integer(), nullable=True),
        sa.Column("target_speaker_id", sa.Integer(), nullable=True),
        sa.Column("source_message_id", sa.Integer(), nullable=False),
        sa.Column("ref_kind", sa.String(length=16), nullable=True),
        sa.Column("ref_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("withdraw_reason", sa.String(length=16), nullable=True),
        sa.Column("asked_message_id", sa.Integer(), nullable=True),
        sa.Column("responded_message_id", sa.Integer(), nullable=True),
        sa.Column("judged_message_id", sa.Integer(), nullable=True),
        sa.Column("followup_needed", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("resolved_message_id", sa.Integer(), nullable=True),
        sa.Column("detected_by", sa.String(length=16), nullable=False),
        sa.Column("decided_by", sa.String(length=16), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["speaker_id"], ["speakers.id"]),
        sa.ForeignKeyConstraint(["target_speaker_id"], ["speakers.id"]),
        sa.ForeignKeyConstraint(
            ["source_message_id"], ["messages.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["asked_message_id"], ["messages.id"]),
        sa.ForeignKeyConstraint(["responded_message_id"], ["messages.id"]),
        sa.ForeignKeyConstraint(["judged_message_id"], ["messages.id"]),
        sa.ForeignKeyConstraint(["resolved_message_id"], ["messages.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("conversation_states", schema=None) as batch_op:
        batch_op.create_index(
            "ix_conversation_states_conv_status",
            ["conversation_id", "status"],
            unique=False,
        )
        batch_op.create_index(
            "ix_conversation_states_conv_ref",
            ["conversation_id", "kind", "ref_kind", "ref_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("conversation_states", schema=None) as batch_op:
        batch_op.drop_index("ix_conversation_states_conv_ref")
        batch_op.drop_index("ix_conversation_states_conv_status")
    op.drop_table("conversation_states")
