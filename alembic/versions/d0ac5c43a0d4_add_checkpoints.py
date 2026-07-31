"""add checkpoints

Revision ID: d0ac5c43a0d4
Revises: 7152dd5b6026
Create Date: 2026-07-31 11:35:46.679257

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd0ac5c43a0d4'
down_revision: Union[str, Sequence[str], None] = '7152dd5b6026'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add immutable checkpoints and resume linkage."""
    op.create_table(
        "checkpoints",
        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "run_id",
            sa.BigInteger(),
            nullable=False,
        ),
        sa.Column(
            "run_step",
            sa.BigInteger(),
            nullable=False,
        ),
        sa.Column(
            "model_step",
            sa.BigInteger(),
            nullable=False,
        ),
        sa.Column(
            "save_reason",
            sa.Enum(
                "initial",
                "periodic",
                "final",
                "manual",
                "interrupted",
                name="checkpoint_save_reason",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column(
            "relative_path",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "sha256",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "size_bytes",
            sa.BigInteger(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "sha256 ~ '^[0-9a-f]{64}$'",
            name=op.f(
                "ck_checkpoints_sha256_lowercase_hex"
            ),
        ),
        sa.CheckConstraint(
            "model_step >= 0",
            name=op.f(
                "ck_checkpoints_model_step_nonnegative"
            ),
        ),
        sa.CheckConstraint(
            "model_step >= run_step",
            name=op.f(
                "ck_checkpoints_"
                "model_step_not_less_than_run_step"
            ),
        ),
        sa.CheckConstraint(
            "run_step >= 0",
            name=op.f(
                "ck_checkpoints_run_step_nonnegative"
            ),
        ),
        sa.CheckConstraint(
            "size_bytes > 0",
            name=op.f(
                "ck_checkpoints_size_bytes_positive"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
            name=op.f(
                "fk_checkpoints_run_id_runs"
            ),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_checkpoints"),
        ),
        sa.UniqueConstraint(
            "relative_path",
            name="uq_checkpoints_relative_path",
        ),
        sa.UniqueConstraint(
            "run_id",
            "run_step",
            name="uq_checkpoints_run_id_run_step",
        ),
    )

    op.create_index(
        op.f("ix_checkpoints_run_id"),
        "checkpoints",
        ["run_id"],
        unique=False,
    )

    op.create_index(
        "ux_checkpoints_one_final_per_run",
        "checkpoints",
        ["run_id"],
        unique=True,
        postgresql_where=sa.text(
            "save_reason = 'final'"
        ),
    )

    op.add_column(
        "runs",
        sa.Column(
            "source_checkpoint_id",
            sa.BigInteger(),
            nullable=True,
        ),
    )

    op.create_index(
        op.f("ix_runs_source_checkpoint_id"),
        "runs",
        ["source_checkpoint_id"],
        unique=False,
    )

    op.create_foreign_key(
        "fk_runs_source_checkpoint_id_checkpoints",
        "runs",
        "checkpoints",
        ["source_checkpoint_id"],
        ["id"],
        ondelete="RESTRICT",
    )

    op.create_check_constraint(
        op.f("ck_runs_continuation_fields"),
        "runs",
        """
        (
            continuation_mode = 'fresh'
            AND source_checkpoint_id IS NULL
        )
        OR
        (
            continuation_mode = 'resume'
            AND source_checkpoint_id IS NOT NULL
        )
        """,
    )


def downgrade() -> None:
    """Remove checkpoints and restore runs without resume linkage."""
    op.drop_constraint(
        op.f("ck_runs_continuation_fields"),
        "runs",
        type_="check",
    )

    op.drop_constraint(
        "fk_runs_source_checkpoint_id_checkpoints",
        "runs",
        type_="foreignkey",
    )

    op.drop_index(
        op.f("ix_runs_source_checkpoint_id"),
        table_name="runs",
    )

    op.drop_column(
        "runs",
        "source_checkpoint_id",
    )

    op.drop_index(
        "ux_checkpoints_one_final_per_run",
        table_name="checkpoints",
        postgresql_where=sa.text(
            "save_reason = 'final'"
        ),
    )

    op.drop_index(
        op.f("ix_checkpoints_run_id"),
        table_name="checkpoints",
    )

    op.drop_table("checkpoints")
