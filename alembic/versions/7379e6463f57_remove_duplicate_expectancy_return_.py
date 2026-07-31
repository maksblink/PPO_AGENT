"""remove duplicate expectancy return metric

Revision ID: 7379e6463f57
Revises: 1eced30b59b4
Create Date: 2026-07-31 14:54:59.307751

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7379e6463f57'
down_revision: Union[str, Sequence[str], None] = '1eced30b59b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Remove a metric duplicating avg_trade_return."""

    op.drop_constraint(
        op.f(
            "ck_evaluations_"
            "round_trip_metric_nullability"
        ),
        "evaluations",
        type_="check",
    )

    op.drop_column(
        "evaluations",
        "expectancy_return",
    )

    op.create_check_constraint(
        op.f(
            "ck_evaluations_"
            "round_trip_metric_nullability"
        ),
        "evaluations",
        """
        status <> 'completed'
        OR
        (
            (
                round_trips = 0
                AND win_rate IS NULL
                AND loss_rate IS NULL
                AND breakeven_rate IS NULL
                AND avg_trade_return IS NULL
                AND median_trade_return IS NULL
                AND min_bars_held IS NULL
                AND avg_bars_held IS NULL
                AND median_bars_held IS NULL
                AND max_bars_held IS NULL
                AND avg_fee_per_trade_return IS NULL
                AND avg_swap_per_trade_return IS NULL
                AND avg_cost_per_trade_return IS NULL
            )
            OR
            (
                round_trips > 0
                AND win_rate IS NOT NULL
                AND loss_rate IS NOT NULL
                AND breakeven_rate IS NOT NULL
                AND avg_trade_return IS NOT NULL
                AND median_trade_return IS NOT NULL
                AND min_bars_held IS NOT NULL
                AND avg_bars_held IS NOT NULL
                AND median_bars_held IS NOT NULL
                AND max_bars_held IS NOT NULL
                AND avg_fee_per_trade_return IS NOT NULL
                AND avg_swap_per_trade_return IS NOT NULL
                AND avg_cost_per_trade_return IS NOT NULL
            )
        )
        """,
    )


def downgrade() -> None:
    """Restore the historical duplicate metric."""

    op.drop_constraint(
        op.f(
            "ck_evaluations_"
            "round_trip_metric_nullability"
        ),
        "evaluations",
        type_="check",
    )

    op.add_column(
        "evaluations",
        sa.Column(
            "expectancy_return",
            sa.Float(),
            nullable=True,
        ),
    )

    # Under the historical definition expectancy_return and
    # avg_trade_return contain the same value.
    op.execute(
        """
        UPDATE evaluations
        SET expectancy_return = avg_trade_return
        WHERE status = 'completed'
          AND round_trips > 0
        """
    )

    op.create_check_constraint(
        op.f(
            "ck_evaluations_"
            "round_trip_metric_nullability"
        ),
        "evaluations",
        """
        status <> 'completed'
        OR
        (
            (
                round_trips = 0
                AND win_rate IS NULL
                AND loss_rate IS NULL
                AND breakeven_rate IS NULL
                AND avg_trade_return IS NULL
                AND median_trade_return IS NULL
                AND expectancy_return IS NULL
                AND min_bars_held IS NULL
                AND avg_bars_held IS NULL
                AND median_bars_held IS NULL
                AND max_bars_held IS NULL
                AND avg_fee_per_trade_return IS NULL
                AND avg_swap_per_trade_return IS NULL
                AND avg_cost_per_trade_return IS NULL
            )
            OR
            (
                round_trips > 0
                AND win_rate IS NOT NULL
                AND loss_rate IS NOT NULL
                AND breakeven_rate IS NOT NULL
                AND avg_trade_return IS NOT NULL
                AND median_trade_return IS NOT NULL
                AND expectancy_return IS NOT NULL
                AND min_bars_held IS NOT NULL
                AND avg_bars_held IS NOT NULL
                AND median_bars_held IS NOT NULL
                AND max_bars_held IS NOT NULL
                AND avg_fee_per_trade_return IS NOT NULL
                AND avg_swap_per_trade_return IS NOT NULL
                AND avg_cost_per_trade_return IS NOT NULL
            )
        )
        """,
    )

