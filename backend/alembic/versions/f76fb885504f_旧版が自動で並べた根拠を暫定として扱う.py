"""旧版が自動で並べた根拠を暫定として扱う

`71e361defdda` で basis_is_provisional を足したとき、既存の行はすべて偽にした。
しかし旧版が採用時に自動で並べた根拠も偽になり、**そこから漏れた記憶の訂正が
届かないまま**になる。新しく作る状態には効く修正が、すでに採用した状態には
届かない（フェーズ3第5回レビューの指摘1）。

自動で並べた根拠と、手で指定した根拠は区別できる。

- 自動：振り返りが作った状態。source_conversation_id を持ち、採用のときに
  その会話の記憶が入った。
- 手動：API から作った状態。source_conversation_id は入らず、根拠は指定された
  ものだけ。

そのため、source_conversation_id と根拠の両方を持つ行を暫定として扱う。
区別できないことを理由に、確定扱いへ寄せない。

Revision ID: f76fb885504f
Revises: 71e361defdda
Create Date: 2026-09-08 11:00:00.000000
"""

from typing import Sequence, Union

from alembic import op

revision: str = "f76fb885504f"
down_revision: Union[str, None] = "71e361defdda"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE character_states
           SET basis_is_provisional = 1
         WHERE source_conversation_id IS NOT NULL
           AND basis_memory_ids IS NOT NULL
        """
    )


def downgrade() -> None:
    # 戻すと、旧版と同じ「すべて確定扱い」に戻る。暫定だった行を確定へ変える
    # ことになるため、訂正が届かない状態へ戻ることを承知で実行する。
    op.execute(
        """
        UPDATE character_states
           SET basis_is_provisional = 0
         WHERE source_conversation_id IS NOT NULL
           AND basis_memory_ids IS NOT NULL
        """
    )
