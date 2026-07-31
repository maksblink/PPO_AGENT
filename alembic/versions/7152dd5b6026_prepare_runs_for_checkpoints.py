"""prepare runs for checkpoints

Revision ID: 7152dd5b6026
Revises: 0ee779db8292
Create Date: 2026-07-31 11:19:28.099756

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7152dd5b6026'
down_revision: Union[str, Sequence[str], None] = '0ee779db8292'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Prepare runs for the checkpoints table."""
    op.add_column(
        "runs",
        sa.Column(
            "normalized_config_sha256",
            sa.String(length=64),
            nullable=False,
        ),
    )

    op.drop_constraint(
        op.f("ck_runs_continuation_fields"),
        "runs",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_runs_source_run_not_self"),
        "runs",
        type_="check",
    )

    op.drop_index(
        op.f("ix_runs_source_run_id"),
        table_name="runs",
    )
    op.drop_constraint(
        op.f("fk_runs_source_run_id_runs"),
        "runs",
        type_="foreignkey",
    )

    op.drop_column(
        "runs",
        "source_checkpoint",
    )
    op.drop_column(
        "runs",
        "source_run_id",
    )


def downgrade() -> None:
    """Restore the previous run continuation fields."""
    op.add_column(
        "runs",
        sa.Column(
            "source_run_id",
            sa.BigInteger(),
            nullable=True,
        ),
    )
    op.add_column(
        "runs",
        sa.Column(
            "source_checkpoint",
            sa.String(length=128),
            nullable=True,
        ),
    )

    op.create_foreign_key(
        op.f("fk_runs_source_run_id_runs"),
        "runs",
        "runs",
        ["source_run_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        op.f("ix_runs_source_run_id"),
        "runs",
        ["source_run_id"],
        unique=False,
    )

    op.create_check_constraint(
        op.f("ck_runs_source_run_not_self"),
        "runs",
        """
        source_run_id IS NULL
        OR source_run_id <> id
        """,
    )
    op.create_check_constraint(
        op.f("ck_runs_continuation_fields"),
        "runs",
        """
        (
            continuation_mode = 'fresh'
            AND source_run_id IS NULL
            AND source_checkpoint IS NULL
        )
        OR
        (
            continuation_mode = 'resume'
            AND source_run_id IS NOT NULL
            AND source_checkpoint IS NOT NULL
            AND length(trim(source_checkpoint)) > 0
        )
        """,
    )

    op.drop_column(
        "runs",
        "normalized_config_sha256",
    )
