"""add post train critic diagnostics

Revision ID: 97d8e21241ba
Revises: dc7b6c43a300
Create Date: 2026-08-11 16:50:48.179567
"""

from alembic import op
import sqlalchemy as sa


revision = "97d8e21241ba"
down_revision = "dc7b6c43a300"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "training_metrics",
        sa.Column(
            "post_train_value_prediction_mean",
            sa.Float(),
            nullable=True,
        ),
    )
    op.add_column(
        "training_metrics",
        sa.Column(
            "post_train_value_prediction_std",
            sa.Float(),
            nullable=True,
        ),
    )
    op.add_column(
        "training_metrics",
        sa.Column(
            "post_train_value_error_mean",
            sa.Float(),
            nullable=True,
        ),
    )
    op.add_column(
        "training_metrics",
        sa.Column(
            "post_train_value_error_std",
            sa.Float(),
            nullable=True,
        ),
    )
    op.add_column(
        "training_metrics",
        sa.Column(
            "post_train_value_mse",
            sa.Float(),
            nullable=True,
        ),
    )
    op.add_column(
        "training_metrics",
        sa.Column(
            "post_train_explained_variance",
            sa.Float(),
            nullable=True,
        ),
    )
    op.add_column(
        "training_metrics",
        sa.Column(
            "post_train_value_target_prediction_corr",
            sa.Float(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column(
        "training_metrics",
        "post_train_value_target_prediction_corr",
    )
    op.drop_column(
        "training_metrics",
        "post_train_explained_variance",
    )
    op.drop_column(
        "training_metrics",
        "post_train_value_mse",
    )
    op.drop_column(
        "training_metrics",
        "post_train_value_error_std",
    )
    op.drop_column(
        "training_metrics",
        "post_train_value_error_mean",
    )
    op.drop_column(
        "training_metrics",
        "post_train_value_prediction_std",
    )
    op.drop_column(
        "training_metrics",
        "post_train_value_prediction_mean",
    )
