"""振り返りの進行状態を残す

会話終了の振り返りを、応答を待たせない後続のジョブへ分けるため（フェーズ4 PR5）。
LLM を3回以上呼ぶので20秒以上かかり、終わるまで画面に何も出ない状態だった。

- reflection_step：いまどこを処理しているか（拾う／選ぶ／関心・関係性／目標）。
- reflection_error：失敗した理由。開始だけが残った状態と、失敗して終わった
  状態を区別する。失敗はやり直せるので、開始権は解放したうえで理由を残す。

進行はプロセス内の変数ではなく DB に置く。初期は単一プロセスのジョブで足りる
が、別プロセスへ移すときに進行の見え方を作り直さずに済む（ISSUE-025）。

Revision ID: e9759b8e6cfa
Revises: 5067142efea2
Create Date: 2026-09-08 18:05:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e9759b8e6cfa"
down_revision: Union[str, None] = "5067142efea2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.add_column(sa.Column("reflection_step", sa.String(length=24), nullable=True))
        batch_op.add_column(sa.Column("reflection_error", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.drop_column("reflection_error")
        batch_op.drop_column("reflection_step")
