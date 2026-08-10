"""add training metrics

Revision ID: e5f6a7b8c9d0
Revises: c4d5e6f7a8b9
Create Date: 2026-08-10
"""

from alembic import op
import sqlalchemy as sa


revision = "e5f6a7b8c9d0"
down_revision = "c4d5e6f7a8b9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "training_metrics",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("run_step", sa.BigInteger(), nullable=False),
        sa.Column("model_step", sa.BigInteger(), nullable=False),
        sa.Column("rollout_number", sa.BigInteger(), nullable=False),
        sa.Column("ep_reward", sa.Float(), nullable=True),
        sa.Column("ep_len", sa.Float(), nullable=True),
        sa.Column("rollout_reward_mean", sa.Float(), nullable=True),
        sa.Column("rollout_reward_sum", sa.Float(), nullable=True),
        sa.Column("approx_kl", sa.Float(), nullable=True),
        sa.Column("clip_fraction", sa.Float(), nullable=True),
        sa.Column("clip_range", sa.Float(), nullable=True),
        sa.Column("entropy_loss", sa.Float(), nullable=True),
        sa.Column("explained_variance", sa.Float(), nullable=True),
        sa.Column("learning_rate", sa.Float(), nullable=True),
        sa.Column("loss", sa.Float(), nullable=True),
        sa.Column("n_updates", sa.BigInteger(), nullable=True),
        sa.Column("policy_gradient_loss", sa.Float(), nullable=True),
        sa.Column("value_loss", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "run_step >= 1",
            name="ck_training_metrics_run_step_positive",
        ),
        sa.CheckConstraint(
            "model_step >= run_step",
            name="ck_training_metrics_model_step_not_less_than_run_step",
        ),
        sa.CheckConstraint(
            "rollout_number >= 1",
            name="ck_training_metrics_rollout_number_positive",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["runs.id"],
            name="fk_training_metrics_run_id_runs",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_training_metrics"),
        sa.UniqueConstraint(
            "run_id",
            "run_step",
            name="uq_training_metrics_run_id_run_step",
        ),
    )
    op.create_index(
        "ix_training_metrics_run_id",
        "training_metrics",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        "ix_training_metrics_run_id_model_step",
        "training_metrics",
        ["run_id", "model_step"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_training_metrics_run_id_model_step",
        table_name="training_metrics",
    )
    op.drop_index(
        "ix_training_metrics_run_id",
        table_name="training_metrics",
    )
    op.drop_table("training_metrics")
