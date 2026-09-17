"""中断時に届いた文字数を残す

ISSUE-051：`delivery_state` はメッセージ単位で、中断（aborted）されたとき
「どこまで実際に伝わったか」を記録する場所が無かった。3文の発話が2文目で
止まっても、全体が「未提示」として扱われるか、全体が「提示済み」として
扱われるかの二択で、どちらも誤りだった。

- messages.delivered_char_count：中断時に、画面が再生位置（currentTime /
  duration）から近似した文字数。completed のときは常に NULL のまま
  （「全文が届いた」を意味させる。content の全長を書き直して二重に表現し
  ない）。既存の行はすべて completed／generated／aborted のいずれかで、
  aborted の既存行も近似値を持たないため、遡って埋めず NULL のままにする。

Revision ID: d794e4b41323
Revises: 836089fcf4d9
Create Date: 2026-09-17 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d794e4b41323"
down_revision: Union[str, None] = "836089fcf4d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.add_column(sa.Column("delivered_char_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.drop_column("delivered_char_count")
