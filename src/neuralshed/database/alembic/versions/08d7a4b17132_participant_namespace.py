"""participant namespace

Revision ID: 08d7a4b17132
Revises: c4d2e8f1a7b3
Create Date: 2026-10-21 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "08d7a4b17132"
down_revision: Union[str, Sequence[str], None] = "c4d2e8f1a7b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Rename ``participants.consortium`` to ``participants.namespace``."""
    with op.batch_alter_table("participants", schema=None) as batch_op:
        batch_op.alter_column("consortium", new_column_name="namespace")


def downgrade() -> None:
    """Rename ``participants.namespace`` back to ``participants.consortium``."""
    with op.batch_alter_table("participants", schema=None) as batch_op:
        batch_op.alter_column("namespace", new_column_name="consortium")
