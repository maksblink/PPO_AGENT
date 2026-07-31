"""allow dynamic long short threshold policy

Revision ID: 6935d7cd6858
Revises: 7379e6463f57
"""

from typing import Sequence, Union

from alembic import op


revision: str = '6935d7cd6858'
down_revision: Union[
    str,
    Sequence[str],
    None,
] = '7379e6463f57'
branch_labels: Union[
    str,
    Sequence[str],
    None,
] = None
depends_on: Union[
    str,
    Sequence[str],
    None,
] = None


NEW_POLICY_FIELDS = """
(
    policy_mode IN (
        'deterministic_argmax',
        'stochastic_sample'
    )
    AND threshold_action IS NULL
    AND probability_threshold IS NULL
)
OR
(
    policy_mode = 'probability_threshold'
    AND (
        threshold_action IS NULL
        OR threshold_action = 1
    )
    AND probability_threshold IS NOT NULL
    AND probability_threshold >= 0.0
    AND probability_threshold <= 1.0
)
"""


OLD_POLICY_FIELDS = """
(
    policy_mode IN (
        'deterministic_argmax',
        'stochastic_sample'
    )
    AND threshold_action IS NULL
    AND probability_threshold IS NULL
)
OR
(
    policy_mode = 'probability_threshold'
    AND threshold_action IS NOT NULL
    AND threshold_action >= 0
    AND probability_threshold IS NOT NULL
    AND probability_threshold >= 0.0
    AND probability_threshold <= 1.0
)
"""


def upgrade() -> None:
    """Allow dynamic LONG/SHORT threshold selection."""
    op.drop_constraint(
        op.f(
            "ck_evaluations_policy_fields"
        ),
        "evaluations",
        type_="check",
    )

    op.create_check_constraint(
        op.f(
            "ck_evaluations_policy_fields"
        ),
        "evaluations",
        NEW_POLICY_FIELDS,
    )


def downgrade() -> None:
    """Restore mandatory threshold_action."""
    op.drop_constraint(
        op.f(
            "ck_evaluations_policy_fields"
        ),
        "evaluations",
        type_="check",
    )

    op.create_check_constraint(
        op.f(
            "ck_evaluations_policy_fields"
        ),
        "evaluations",
        OLD_POLICY_FIELDS,
    )
