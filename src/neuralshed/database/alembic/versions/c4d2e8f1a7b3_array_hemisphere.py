"""array hemisphere

Revision ID: c4d2e8f1a7b3
Revises: a5f57b9d9432
Create Date: 2026-09-23 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "c4d2e8f1a7b3"
down_revision: Union[str, Sequence[str], None] = "a5f57b9d9432"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add ``arrays.hemisphere`` and make (subject, hemisphere, name) the array key."""
    with op.batch_alter_table("arrays", schema=None) as batch_op:
        batch_op.add_column(sa.Column("hemisphere", sa.String(), nullable=True))
        batch_op.drop_constraint("uq_array_subject_name", type_="unique")
        batch_op.create_unique_constraint(
            "uq_array_subject_hemisphere_name", ["subject_id", "hemisphere", "name"]
        )


def downgrade() -> None:
    """Drop ``arrays.hemisphere`` and restore the (subject, name) key."""
    with op.batch_alter_table("arrays", schema=None) as batch_op:
        batch_op.drop_constraint("uq_array_subject_hemisphere_name", type_="unique")
        batch_op.create_unique_constraint(
            "uq_array_subject_name", ["subject_id", "name"]
        )
        batch_op.drop_column("hemisphere")
