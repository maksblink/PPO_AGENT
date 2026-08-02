"""add run lifecycle constraints

Revision ID: 8f1c2d3e4a5b
Revises: 6935d7cd6858
Create Date: 2026-08-02 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


revision: str = "8f1c2d3e4a5b"
down_revision: Union[str, Sequence[str], None] = "6935d7cd6858"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_RUN_EXECUTION_TIME_ORDER = """
finished_at IS NULL
OR started_at IS NULL
OR finished_at >= started_at
"""

_RUN_STATUS_FIELDS = """
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


def upgrade() -> None:
    op.create_check_constraint(
        op.f("ck_runs_execution_time_order"),
        "runs",
        _RUN_EXECUTION_TIME_ORDER,
    )
    op.create_check_constraint(
        op.f("ck_runs_status_fields"),
        "runs",
        _RUN_STATUS_FIELDS,
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_runs_status_fields"),
        "runs",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_runs_execution_time_order"),
        "runs",
        type_="check",
    )
