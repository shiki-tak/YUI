"""目標のテーブルを追加する

設計書 5「目標」と 6「会話・自発的行動の流れ」のために、YUI が次に何を話したいかを
保存できるようにする（フェーズ4）。

- goals：目的・相手・実行条件・期限・状態・根拠・最後に実行した時刻。実行
  （last_executed_at）と達成（completed_at）を別の列に持つ。質問を投げても
  相手が答えなければ、実行済みで未達成であるため。
- goal_revisions：変更履歴。採用・実行・完了・取消を根拠付きで戻せるようにする。

関心・関係性（character_states）に kind を足すのではなく、別の表にする。目標
だけが実行条件と期限を必要とし、終わり方（達成・取消・期限切れ）も違うため。
根拠の記憶と再評価の印は同じ形にしてあり、ISSUE-016 の波及にそのまま乗せる。

Revision ID: c667f41b8e3c
Revises: f76fb885504f
Create Date: 2026-09-08 12:08:53.647684
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c667f41b8e3c"
down_revision: Union[str, None] = "f76fb885504f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "goals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("subject_speaker_id", sa.Integer(), nullable=True),
        sa.Column("trigger", sa.String(length=24), nullable=False),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("basis_memory_ids", sa.JSON(), nullable=True),
        sa.Column("basis_is_provisional", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("needs_review", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("review_reason", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("visibility", sa.String(length=16), nullable=False),
        sa.Column("visible_to_speaker_id", sa.Integer(), nullable=True),
        sa.Column("source_conversation_id", sa.Integer(), nullable=True),
        sa.Column("last_executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["subject_speaker_id"], ["speakers.id"]),
        sa.ForeignKeyConstraint(["visible_to_speaker_id"], ["speakers.id"]),
        sa.ForeignKeyConstraint(["source_conversation_id"], ["conversations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("goals", schema=None) as batch_op:
        batch_op.create_index("ix_goals_status_trigger", ["status", "trigger"], unique=False)

    op.create_table(
        "goal_revisions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("goal_id", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=24), nullable=False),
        sa.Column("before", sa.JSON(), nullable=True),
        sa.Column("after", sa.JSON(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["goal_id"], ["goals.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("goal_revisions", schema=None) as batch_op:
        batch_op.create_index("ix_goal_revisions_goal_id", ["goal_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("goal_revisions", schema=None) as batch_op:
        batch_op.drop_index("ix_goal_revisions_goal_id")
    op.drop_table("goal_revisions")

    with op.batch_alter_table("goals", schema=None) as batch_op:
        batch_op.drop_index("ix_goals_status_trigger")
    op.drop_table("goals")
