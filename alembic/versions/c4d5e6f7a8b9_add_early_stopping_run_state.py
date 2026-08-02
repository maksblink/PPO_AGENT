"""add early stopping run state

Revision ID: c4d5e6f7a8b9
Revises: 8f1c2d3e4a5b
Create Date: 2026-08-02 15:10:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "c4d5e6f7a8b9"
down_revision: Union[str, Sequence[str], None] = "8f1c2d3e4a5b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_OLD_RUN_STATUS_FIELDS = """
(
    status = 'pending'
    AND started_at IS NULL
    AND finished_at IS NULL
    AND training_steps_completed = 0
    AND data_epochs_completed = 0
    AND error_type IS NULL
    AND error_message IS NULL
)
OR
(
    status = 'running'
    AND started_at IS NOT NULL
    AND finished_at IS NULL
    AND error_type IS NULL
    AND error_message IS NULL
)
OR
(
    status = 'completed'
    AND started_at IS NOT NULL
    AND finished_at IS NOT NULL
    AND training_steps_completed = training_steps_requested
    AND error_type IS NULL
    AND error_message IS NULL
)
OR
(
    status = 'failed'
    AND started_at IS NOT NULL
    AND finished_at IS NOT NULL
    AND error_type IS NOT NULL
    AND length(trim(error_type)) > 0
    AND error_message IS NOT NULL
    AND length(trim(error_message)) > 0
)
OR
(
    status = 'cancelled'
    AND started_at IS NOT NULL
    AND finished_at IS NOT NULL
)
"""

_NEW_RUN_STATUS_FIELDS = """
(
    status = 'pending'
    AND started_at IS NULL
    AND finished_at IS NULL
    AND training_steps_completed = 0
    AND data_epochs_completed = 0
    AND stopped_early = false
    AND early_stop_reason IS NULL
    AND error_type IS NULL
    AND error_message IS NULL
)
OR
(
    status = 'running'
    AND started_at IS NOT NULL
    AND finished_at IS NULL
    AND stopped_early = false
    AND early_stop_reason IS NULL
    AND error_type IS NULL
    AND error_message IS NULL
)
OR
(
    status = 'completed'
    AND started_at IS NOT NULL
    AND finished_at IS NOT NULL
    AND
    (
        (
            stopped_early = false
            AND early_stop_reason IS NULL
            AND training_steps_completed = training_steps_requested
        )
        OR
        (
            stopped_early = true
            AND early_stop_reason IS NOT NULL
            AND length(trim(early_stop_reason)) > 0
            AND training_steps_completed > 0
            AND training_steps_completed < training_steps_requested
        )
    )
    AND error_type IS NULL
    AND error_message IS NULL
)
OR
(
    status = 'failed'
    AND started_at IS NOT NULL
    AND finished_at IS NOT NULL
    AND stopped_early = false
    AND early_stop_reason IS NULL
    AND error_type IS NOT NULL
    AND length(trim(error_type)) > 0
    AND error_message IS NOT NULL
    AND length(trim(error_message)) > 0
)
OR
(
    status = 'cancelled'
    AND started_at IS NOT NULL
    AND finished_at IS NOT NULL
    AND stopped_early = false
    AND early_stop_reason IS NULL
)
"""

_EARLY_STOP_FIELDS = """
(
    stopped_early = false
    AND early_stop_reason IS NULL
)
OR
(
    stopped_early = true
    AND early_stop_reason IS NOT NULL
    AND length(trim(early_stop_reason)) > 0
)
"""


def upgrade() -> None:
    op.add_column(
        "runs",
        sa.Column(
            "stopped_early",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "runs",
        sa.Column(
            "early_stop_reason",
            sa.Text(),
            nullable=True,
        ),
    )

    op.drop_constraint(
        op.f("ck_runs_status_fields"),
        "runs",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_runs_early_stop_fields"),
        "runs",
        _EARLY_STOP_FIELDS,
    )
    op.create_check_constraint(
        op.f("ck_runs_status_fields"),
        "runs",
        _NEW_RUN_STATUS_FIELDS,
    )

    op.drop_constraint(
        op.f("uq_checkpoints_run_id_run_step"),
        "checkpoints",
        type_="unique",
    )
    op.create_unique_constraint(
        op.f(
            "uq_checkpoints_run_id_run_step_save_reason"
        ),
        "checkpoints",
        ["run_id", "run_step", "save_reason"],
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f(
            "uq_checkpoints_run_id_run_step_save_reason"
        ),
        "checkpoints",
        type_="unique",
    )
    op.create_unique_constraint(
        op.f("uq_checkpoints_run_id_run_step"),
        "checkpoints",
        ["run_id", "run_step"],
    )

    op.drop_constraint(
        op.f("ck_runs_status_fields"),
        "runs",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_runs_early_stop_fields"),
        "runs",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_runs_status_fields"),
        "runs",
        _OLD_RUN_STATUS_FIELDS,
    )

    op.drop_column(
        "runs",
        "early_stop_reason",
    )
    op.drop_column(
        "runs",
        "stopped_early",
    )
