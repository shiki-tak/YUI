"""可変状態のテーブルを追加する

設計書 4.1「固定人格と変化する状態」と、フェーズ3の3B のために、YUI の関心と
相手との関係を保存できるようにする（ISSUE-015）。

- character_states：関心（interest）と関係性（relationship）。候補から採用までを
  status で追う。根拠になった記憶のIDを持ち、訂正・削除されたときに再評価の印を
  付けられるようにする（ISSUE-016）。単一の好感度の数値にはしない。
- character_state_revisions：変更履歴。何を根拠に、なぜ採用したかを残して戻せる。
- run_records.referenced_state_ids：返答に渡した可変状態。完了条件「参照した
  記憶・人格版・可変状態を追跡できる」のため、記憶と分けて残す。

目標（次に感想を聞く等）はフェーズ4の担当なので、この版では作らない。ただし
「候補 → 採用 → 根拠を残す」流れは同じ形にしてあり、フェーズ4では kind を1つ
増やし、実行条件・期限・完了条件の列を足せば足りるようにしている。

Revision ID: 3187f6b349d3
Revises: 66c78282e48e
Create Date: 2026-09-07 13:40:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "3187f6b349d3"
down_revision: Union[str, None] = "66c78282e48e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "character_states",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("subject_speaker_id", sa.Integer(), nullable=True),
        sa.Column("topic", sa.String(length=120), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("basis_memory_ids", sa.JSON(), nullable=True),
        sa.Column("needs_review", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("review_reason", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("visibility", sa.String(length=16), nullable=False),
        sa.Column("visible_to_speaker_id", sa.Integer(), nullable=True),
        sa.Column("superseded_by_id", sa.Integer(), nullable=True),
        sa.Column("source_conversation_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["subject_speaker_id"], ["speakers.id"]),
        sa.ForeignKeyConstraint(["visible_to_speaker_id"], ["speakers.id"]),
        sa.ForeignKeyConstraint(["superseded_by_id"], ["character_states.id"]),
        sa.ForeignKeyConstraint(["source_conversation_id"], ["conversations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("character_states", schema=None) as batch_op:
        batch_op.create_index(
            "ix_character_states_kind_status", ["kind", "status"], unique=False
        )

    op.create_table(
        "character_state_revisions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("state_id", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=24), nullable=False),
        sa.Column("before", sa.JSON(), nullable=True),
        sa.Column("after", sa.JSON(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["state_id"], ["character_states.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("character_state_revisions", schema=None) as batch_op:
        batch_op.create_index(
            "ix_character_state_revisions_state_id", ["state_id"], unique=False
        )

    with op.batch_alter_table("run_records", schema=None) as batch_op:
        batch_op.add_column(sa.Column("referenced_state_ids", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("run_records", schema=None) as batch_op:
        batch_op.drop_column("referenced_state_ids")

    with op.batch_alter_table("character_state_revisions", schema=None) as batch_op:
        batch_op.drop_index("ix_character_state_revisions_state_id")
    op.drop_table("character_state_revisions")

    with op.batch_alter_table("character_states", schema=None) as batch_op:
        batch_op.drop_index("ix_character_states_kind_status")
    op.drop_table("character_states")
