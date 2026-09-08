"""実行記録に参照した目標と選んだ行動を残す

フェーズ4の完了条件「行動と状態変化の根拠を追え、悪化時に戻せる」のために、
返答へ渡した目標と、選んだ行動を run_records に残す。

- referenced_goal_ids：返答に渡した目標。記憶・可変状態と分けて持つ。評価は
  この記録を見て判定する。返答の言い回しからは、目標が渡ったのかを決められない。
- selected_action：回答・確認質問・話題提案・調査・待機のどれを選んだか。
  待機を選んだことも残す。合格は「よく喋ること」ではないため。

書き込むのは行動選択を入れる後続の版で、この版は列だけを足す。測る側を先に
用意し、実装が入ったときに前後を比べられるようにする。

Revision ID: 5067142efea2
Revises: c667f41b8e3c
Create Date: 2026-09-08 15:41:12.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "5067142efea2"
down_revision: Union[str, None] = "c667f41b8e3c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("run_records", schema=None) as batch_op:
        batch_op.add_column(sa.Column("referenced_goal_ids", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("selected_action", sa.String(length=24), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("run_records", schema=None) as batch_op:
        batch_op.drop_column("selected_action")
        batch_op.drop_column("referenced_goal_ids")
