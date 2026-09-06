"""振り返りの状態を追加し、既存の記憶候補の参照範囲を補完する

- conversations に振り返りの開始・完了を持たせ、「処理中」と「完了」を分ける。
- 前の版で memories だけを補完していたため、移行前から残っていた未採用の
  記憶候補が visible_to_speaker_id=NULL のままだった。採用すると相手を
  限定しない記憶になるため、由来した会話の相手で補完する。

Revision ID: a71c525f0285
Revises: 72df18ddda0e
Create Date: 2026-09-06
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a71c525f0285"
down_revision: Union[str, None] = "72df18ddda0e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# 由来した会話の相手を、その会話で最後に発言した利用者から決める。
# 相手を特定できない候補（利用者の発言が無い会話）は NULL のまま残す。
_BACKFILL_CANDIDATES = sa.text(
    """
    UPDATE memory_candidates
       SET visible_to_speaker_id = (
           SELECT m.speaker_id
             FROM messages m
            WHERE m.conversation_id = memory_candidates.conversation_id
              AND m.speaker_kind = 'user'
              AND m.speaker_id IS NOT NULL
            ORDER BY m.id DESC
            LIMIT 1
       )
     WHERE visibility = 'private'
       AND visible_to_speaker_id IS NULL
    """
)

# 既に終了した会話は、振り返りが完了したものとして扱う。
_BACKFILL_REFLECTION = sa.text(
    """
    UPDATE conversations
       SET reflection_started_at = ended_at,
           reflection_completed_at = ended_at
     WHERE ended_at IS NOT NULL
    """
)


def upgrade() -> None:
    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("reflection_started_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("reflection_completed_at", sa.DateTime(timezone=True), nullable=True)
        )

    bind = op.get_bind()
    bind.execute(_BACKFILL_CANDIDATES)
    bind.execute(_BACKFILL_REFLECTION)


def downgrade() -> None:
    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.drop_column("reflection_completed_at")
        batch_op.drop_column("reflection_started_at")
