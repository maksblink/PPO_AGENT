"""add critic diagnostics to training metrics

Revision ID: dc7b6c43a300
Revises: e5f6a7b8c9d0
Create Date: 2026-08-11 15:09:16.565598
"""

from alembic import op
import sqlalchemy as sa


revision = "dc7b6c43a300"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "training_metrics",
        sa.Column("value_target_mean", sa.Float(), nullable=True),
    )
    op.add_column(
        "training_metrics",
        sa.Column("value_target_std", sa.Float(), nullable=True),
    )
    op.add_column(
        "training_metrics",
        sa.Column("value_prediction_mean", sa.Float(), nullable=True),
    )
    op.add_column(
        "training_metrics",
        sa.Column("value_prediction_std", sa.Float(), nullable=True),
    )
    op.add_column(
        "training_metrics",
        sa.Column("value_error_mean", sa.Float(), nullable=True),
    )
    op.add_column(
        "training_metrics",
        sa.Column("value_error_std", sa.Float(), nullable=True),
    )
    op.add_column(
        "training_metrics",
        sa.Column(
            "value_target_prediction_corr",
            sa.Float(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column(
        "training_metrics",
        "value_target_prediction_corr",
    )
    op.drop_column(
        "training_metrics",
        "value_error_std",
    )
    op.drop_column(
        "training_metrics",
        "value_error_mean",
    )
    op.drop_column(
        "training_metrics",
        "value_prediction_std",
    )
    op.drop_column(
        "training_metrics",
        "value_prediction_mean",
    )
    op.drop_column(
        "training_metrics",
        "value_target_std",
    )
    op.drop_column(
        "training_metrics",
        "value_target_mean",
    )
