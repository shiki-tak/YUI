"""記憶に visible_to_speaker_id を追加する

「誰についての記憶か」（subject_speaker_id）と「誰との会話で参照してよいか」
（visible_to_speaker_id）を別の軸に分ける。非公開記憶が、由来と関係のない
相手との会話へ渡ることを防ぐため。

Revision ID: 72df18ddda0e
Revises: 044af0bc28b0
Create Date: 2026-09-06 17:29:04.512477
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "72df18ddda0e"
down_revision: Union[str, None] = "044af0bc28b0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# 既存の非公開記憶は、由来した会話の相手に紐づける。
# 手動で追加した記憶（source_conversation_id が無い）は限定しないまま残す。
_BACKFILL = sa.text(
    """
    UPDATE memories
       SET visible_to_speaker_id = (
           SELECT m.speaker_id
             FROM messages m
            WHERE m.conversation_id = memories.source_conversation_id
              AND m.speaker_kind = 'user'
              AND m.speaker_id IS NOT NULL
            ORDER BY m.id DESC
            LIMIT 1
       )
     WHERE visibility = 'private'
       AND source_conversation_id IS NOT NULL
    """
)


def upgrade() -> None:
    with op.batch_alter_table("memories", schema=None) as batch_op:
        batch_op.add_column(sa.Column("visible_to_speaker_id", sa.Integer(), nullable=True))
        batch_op.create_index("ix_memories_visible_to", ["visible_to_speaker_id"], unique=False)
        batch_op.create_foreign_key(
            "fk_memories_visible_to_speaker", "speakers", ["visible_to_speaker_id"], ["id"]
        )

    with op.batch_alter_table("memory_candidates", schema=None) as batch_op:
        batch_op.add_column(sa.Column("visible_to_speaker_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_memory_candidates_visible_to_speaker",
            "speakers",
            ["visible_to_speaker_id"],
            ["id"],
        )

    op.get_bind().execute(_BACKFILL)


def downgrade() -> None:
    # SQLite ではバッチモードが表を作り直すため、列を落とせば制約も消える。
    with op.batch_alter_table("memory_candidates", schema=None) as batch_op:
        batch_op.drop_column("visible_to_speaker_id")

    with op.batch_alter_table("memories", schema=None) as batch_op:
        batch_op.drop_index("ix_memories_visible_to")
        batch_op.drop_column("visible_to_speaker_id")
