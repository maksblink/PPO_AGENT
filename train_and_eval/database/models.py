from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum as SQLAlchemyEnum,
    Float,
    ForeignKey,
    Integer,
    Index,
    MetaData,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


DEFAULT_RUN_DESCRIPTION = "Place for my notes"


NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": (
        "fk_%(table_name)s_%(column_0_name)s_"
        "%(referred_table_name)s"
    ),
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention=NAMING_CONVENTION
    )


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContinuationMode(str, enum.Enum):
    FRESH = "fresh"
    RESUME = "resume"


class TrainingDurationUnit(str, enum.Enum):
    DATA_EPOCHS = "data_epochs"
    TIMESTEPS = "timesteps"



class CheckpointSaveReason(str, enum.Enum):
    INITIAL = "initial"
    PERIODIC = "periodic"
    FINAL = "final"
    MANUAL = "manual"
    INTERRUPTED = "interrupted"



class EvaluationStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationTrigger(str, enum.Enum):
    SCHEDULED = "scheduled"
    FINAL = "final"
    MANUAL = "manual"


class EvaluationDataScope(str, enum.Enum):
    RUN_VALIDATION = "run_validation"
    EXTENDED_OUT_OF_SAMPLE = "extended_out_of_sample"
    CUSTOM_RANGE = "custom_range"



class EvaluationPolicyMode(str, enum.Enum):
    DETERMINISTIC_ARGMAX = "deterministic_argmax"
    STOCHASTIC_SAMPLE = "stochastic_sample"
    PROBABILITY_THRESHOLD = "probability_threshold"


class EvaluationCurrentStreakType(str, enum.Enum):
    NONE = "none"
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _enum_values(
    enum_type: type[enum.Enum],
) -> list[str]:
    return [
        str(member.value)
        for member in enum_type
    ]


class Run(Base):
    """One complete PPO training run."""

    __tablename__ = "runs"

    __table_args__ = (
        CheckConstraint(
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
            name="continuation_fields",
        ),
        UniqueConstraint(
            "name",
            name="uq_runs_name",
        ),
        CheckConstraint(
            "duration_amount >= 1",
            name="duration_amount_positive",
        ),
        CheckConstraint(
            "train_rows >= 1",
            name="train_rows_positive",
        ),
        CheckConstraint(
            "validation_rows >= 1",
            name="validation_rows_positive",
        ),
        CheckConstraint(
            "split_index = train_rows",
            name="split_matches_train_rows",
        ),
        CheckConstraint(
            "steps_per_data_epoch >= 1",
            name="steps_per_epoch_positive",
        ),
        CheckConstraint(
            "training_steps_requested >= 1",
            name="requested_steps_positive",
        ),
        CheckConstraint(
            """
            training_steps_completed >= 0
            AND
            training_steps_completed
                <= training_steps_requested
            """,
            name="completed_steps_range",
        ),
        CheckConstraint(
            "data_epochs_completed >= 0",
            name="completed_epochs_nonnegative",
        ),
        CheckConstraint(
            """
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
            """,
            name="early_stop_fields",
        ),
        CheckConstraint(
            """
            finished_at IS NULL
            OR started_at IS NULL
            OR finished_at >= started_at
            """,
            name="execution_time_order",
        ),
        CheckConstraint(
            """
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
                        AND training_steps_completed
                            = training_steps_requested
                    )
                    OR
                    (
                        stopped_early = true
                        AND early_stop_reason IS NOT NULL
                        AND length(trim(early_stop_reason)) > 0
                        AND training_steps_completed > 0
                        AND training_steps_completed
                            < training_steps_requested
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
            """,
            name="status_fields",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    status: Mapped[RunStatus] = mapped_column(
        SQLAlchemyEnum(
            RunStatus,
            name="run_status",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
        default=RunStatus.PENDING,
        server_default=RunStatus.PENDING.value,
    )

    continuation_mode: Mapped[
        ContinuationMode
    ] = mapped_column(
        SQLAlchemyEnum(
            ContinuationMode,
            name="continuation_mode",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )

    source_checkpoint_id: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        ForeignKey(
            "checkpoints.id",
            name=(
                "fk_runs_source_checkpoint_id_"
                "checkpoints"
            ),
            ondelete="RESTRICT",
            use_alter=True,
        ),
        nullable=True,
        index=True,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default=DEFAULT_RUN_DESCRIPTION,
        server_default=text(
            f"'{DEFAULT_RUN_DESCRIPTION}'"
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.now(),
    )

    modified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.now(),
    )

    started_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    finished_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    git_commit: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    git_branch: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    config_schema_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    seed: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    config_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    normalized_config_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    raw_config_yaml: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    normalized_config_json: Mapped[
        dict[str, Any]
    ] = mapped_column(
        JSONB,
        nullable=False,
    )

    data_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    data_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    duration_unit: Mapped[
        TrainingDurationUnit
    ] = mapped_column(
        SQLAlchemyEnum(
            TrainingDurationUnit,
            name="training_duration_unit",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )

    duration_amount: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    split_index: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    train_rows: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    validation_rows: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    steps_per_data_epoch: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    training_steps_requested: Mapped[
        int
    ] = mapped_column(
        BigInteger,
        nullable=False,
    )

    training_steps_completed: Mapped[
        int
    ] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    data_epochs_completed: Mapped[
        Decimal
    ] = mapped_column(
        Numeric(
            precision=18,
            scale=8,
        ),
        nullable=False,
        default=Decimal("0"),
        server_default=text("0"),
    )

    stopped_early: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default=text("false"),
    )

    early_stop_reason: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    error_type: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    error_message: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    checkpoints: Mapped[
        list["Checkpoint"]
    ] = relationship(
        "Checkpoint",
        back_populates="run",
        foreign_keys="Checkpoint.run_id",
    )

    source_checkpoint: Mapped[
        "Checkpoint | None"
    ] = relationship(
        "Checkpoint",
        back_populates="resumed_runs",
        foreign_keys=[source_checkpoint_id],
    )

    def update_description(
        self,
        description: str,
        *,
        modified_at: datetime | None = None,
    ) -> None:
        """
        Change user notes and update modified_at.

        System changes such as status or progress updates do not call this
        method and therefore do not modify modified_at.
        """
        if not isinstance(description, str):
            raise TypeError(
                "description must be a string"
            )

        timestamp = (
            utc_now()
            if modified_at is None
            else modified_at
        )

        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() is None
        ):
            raise ValueError(
                "modified_at must be timezone-aware"
            )

        self.description = description
        self.modified_at = timestamp


class Checkpoint(Base):
    """
    One immutable model file saved during a training run.

    `run_step` is local to the run that created the checkpoint.
    `model_step` includes the complete resume history.
    """

    __tablename__ = "checkpoints"

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "run_step",
            "save_reason",
            name=(
                "uq_checkpoints_run_id_run_step_"
                "save_reason"
            ),
        ),
        UniqueConstraint(
            "relative_path",
            name="uq_checkpoints_relative_path",
        ),
        CheckConstraint(
            "run_step >= 0",
            name="run_step_nonnegative",
        ),
        CheckConstraint(
            "model_step >= 0",
            name="model_step_nonnegative",
        ),
        CheckConstraint(
            "model_step >= run_step",
            name="model_step_not_less_than_run_step",
        ),
        CheckConstraint(
            "size_bytes > 0",
            name="size_bytes_positive",
        ),
        CheckConstraint(
            "sha256 ~ '^[0-9a-f]{64}$'",
            name="sha256_lowercase_hex",
        ),
        Index(
            "ux_checkpoints_one_final_per_run",
            "run_id",
            unique=True,
            postgresql_where=text(
                "save_reason = 'final'"
            ),
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    run_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "runs.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    run_step: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    model_step: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    save_reason: Mapped[
        CheckpointSaveReason
    ] = mapped_column(
        SQLAlchemyEnum(
            CheckpointSaveReason,
            name="checkpoint_save_reason",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )

    relative_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.now(),
    )

    run: Mapped["Run"] = relationship(
        "Run",
        back_populates="checkpoints",
        foreign_keys=[run_id],
    )

    evaluations: Mapped[
        list["Evaluation"]
    ] = relationship(
        "Evaluation",
        back_populates="checkpoint",
        foreign_keys="Evaluation.checkpoint_id",
    )

    resumed_runs: Mapped[
        list["Run"]
    ] = relationship(
        "Run",
        back_populates="source_checkpoint",
        foreign_keys="Run.source_checkpoint_id",
    )


class Evaluation(Base):
    """
    One attempt to evaluate one immutable checkpoint.

    The evaluated range is [evaluation_start_index,
    evaluation_end_index), while earlier rows may be used only as
    historical lookback.
    """

    __tablename__ = "evaluations"

    __table_args__ = (
        CheckConstraint(
            """
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
            """,
            name="policy_fields",
        ),
        CheckConstraint(
            """
            status <> 'completed'
            OR
            (
                agent_return IS NOT NULL
                AND always_long_return IS NOT NULL
                AND always_short_return IS NOT NULL
                AND agent_vs_always_long_return IS NOT NULL
                AND balanced_score IS NOT NULL

                AND agent_max_drawdown IS NOT NULL
                AND always_long_max_drawdown IS NOT NULL
                AND always_short_max_drawdown IS NOT NULL
                AND drawdown_improvement IS NOT NULL

                AND net_exposure IS NOT NULL
                AND long_exposure IS NOT NULL
                AND short_exposure IS NOT NULL
                AND flat_exposure IS NOT NULL
                AND market_exposure IS NOT NULL

                AND trade_events_total IS NOT NULL
                AND trade_event_rate IS NOT NULL
                AND round_trips IS NOT NULL
                AND round_trip_rate IS NOT NULL

                AND winning_trades IS NOT NULL
                AND losing_trades IS NOT NULL
                AND breakeven_trades IS NOT NULL
                AND long_round_trips IS NOT NULL
                AND short_round_trips IS NOT NULL

                AND open_long_count IS NOT NULL
                AND open_short_count IS NOT NULL
                AND close_long_count IS NOT NULL
                AND close_short_count IS NOT NULL
                AND swap_events IS NOT NULL

                AND gross_profit_return IS NOT NULL
                AND gross_loss_return IS NOT NULL
                AND net_profit_return IS NOT NULL

                AND max_consecutive_wins IS NOT NULL
                AND max_consecutive_losses IS NOT NULL
                AND current_streak_type IS NOT NULL
                AND current_streak IS NOT NULL

                AND total_fee_return IS NOT NULL
                AND total_swap_return IS NOT NULL
                AND total_cost_return IS NOT NULL

                AND cumulative_shaped_reward IS NOT NULL
                AND open_position_return_at_end IS NOT NULL
            )
            """,
            name="completed_core_metrics",
        ),
        CheckConstraint(
            """
            (agent_max_drawdown IS NULL
                OR agent_max_drawdown <= 0.0)
            AND
            (always_long_max_drawdown IS NULL
                OR always_long_max_drawdown <= 0.0)
            AND
            (always_short_max_drawdown IS NULL
                OR always_short_max_drawdown <= 0.0)

            AND
            (net_exposure IS NULL
                OR net_exposure BETWEEN -1.0 AND 1.0)
            AND
            (long_exposure IS NULL
                OR long_exposure BETWEEN 0.0 AND 1.0)
            AND
            (short_exposure IS NULL
                OR short_exposure BETWEEN 0.0 AND 1.0)
            AND
            (flat_exposure IS NULL
                OR flat_exposure BETWEEN 0.0 AND 1.0)
            AND
            (market_exposure IS NULL
                OR market_exposure BETWEEN 0.0 AND 1.0)

            AND
            (trade_event_rate IS NULL
                OR trade_event_rate >= 0.0)
            AND
            (round_trip_rate IS NULL
                OR round_trip_rate >= 0.0)

            AND
            (win_rate IS NULL
                OR win_rate BETWEEN 0.0 AND 1.0)
            AND
            (loss_rate IS NULL
                OR loss_rate BETWEEN 0.0 AND 1.0)
            AND
            (breakeven_rate IS NULL
                OR breakeven_rate BETWEEN 0.0 AND 1.0)
            AND
            (long_win_rate IS NULL
                OR long_win_rate BETWEEN 0.0 AND 1.0)
            AND
            (short_win_rate IS NULL
                OR short_win_rate BETWEEN 0.0 AND 1.0)

            AND
            (avg_win_return IS NULL
                OR avg_win_return > 0.0)
            AND
            (avg_loss_return IS NULL
                OR avg_loss_return < 0.0)
            AND
            (largest_win_return IS NULL
                OR largest_win_return > 0.0)
            AND
            (largest_loss_return IS NULL
                OR largest_loss_return < 0.0)

            AND
            (gross_profit_return IS NULL
                OR gross_profit_return >= 0.0)
            AND
            (gross_loss_return IS NULL
                OR gross_loss_return <= 0.0)
            AND
            (profit_factor IS NULL
                OR profit_factor >= 0.0)
            AND
            (payoff_ratio IS NULL
                OR payoff_ratio >= 0.0)
            """,
            name="metric_ranges",
        ),
        CheckConstraint(
            """
            (trade_events_total IS NULL
                OR trade_events_total >= 0)
            AND
            (round_trips IS NULL
                OR round_trips >= 0)
            AND
            (winning_trades IS NULL
                OR winning_trades >= 0)
            AND
            (losing_trades IS NULL
                OR losing_trades >= 0)
            AND
            (breakeven_trades IS NULL
                OR breakeven_trades >= 0)
            AND
            (long_round_trips IS NULL
                OR long_round_trips >= 0)
            AND
            (short_round_trips IS NULL
                OR short_round_trips >= 0)
            AND
            (open_long_count IS NULL
                OR open_long_count >= 0)
            AND
            (open_short_count IS NULL
                OR open_short_count >= 0)
            AND
            (close_long_count IS NULL
                OR close_long_count >= 0)
            AND
            (close_short_count IS NULL
                OR close_short_count >= 0)
            AND
            (swap_events IS NULL
                OR swap_events >= 0)
            AND
            (min_bars_held IS NULL
                OR min_bars_held >= 1)
            AND
            (max_bars_held IS NULL
                OR max_bars_held >= 1)
            AND
            (max_consecutive_wins IS NULL
                OR max_consecutive_wins >= 0)
            AND
            (max_consecutive_losses IS NULL
                OR max_consecutive_losses >= 0)
            AND
            (current_streak IS NULL
                OR current_streak >= 0)
            """,
            name="metric_counts_nonnegative",
        ),
        CheckConstraint(
            """
            status <> 'completed'
            OR
            (
                round_trips =
                    winning_trades
                    + losing_trades
                    + breakeven_trades
                AND
                round_trips =
                    long_round_trips
                    + short_round_trips
            )
            """,
            name="trade_count_consistency",
        ),
        CheckConstraint(
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
            name="round_trip_metric_nullability",
        ),
        CheckConstraint(
            """
            status <> 'completed'
            OR
            (
                (
                    long_round_trips = 0
                    AND long_win_rate IS NULL
                    AND long_avg_trade_return IS NULL
                    AND long_median_trade_return IS NULL
                )
                OR
                (
                    long_round_trips > 0
                    AND long_win_rate IS NOT NULL
                    AND long_avg_trade_return IS NOT NULL
                    AND long_median_trade_return IS NOT NULL
                )
            )
            """,
            name="long_metric_nullability",
        ),
        CheckConstraint(
            """
            status <> 'completed'
            OR
            (
                (
                    short_round_trips = 0
                    AND short_win_rate IS NULL
                    AND short_avg_trade_return IS NULL
                    AND short_median_trade_return IS NULL
                )
                OR
                (
                    short_round_trips > 0
                    AND short_win_rate IS NOT NULL
                    AND short_avg_trade_return IS NOT NULL
                    AND short_median_trade_return IS NOT NULL
                )
            )
            """,
            name="short_metric_nullability",
        ),
        CheckConstraint(
            """
            status <> 'completed'
            OR
            (
                (
                    winning_trades = 0
                    AND avg_win_return IS NULL
                    AND largest_win_return IS NULL
                    AND avg_win_streak IS NULL
                    AND max_consecutive_wins = 0
                )
                OR
                (
                    winning_trades > 0
                    AND avg_win_return IS NOT NULL
                    AND largest_win_return IS NOT NULL
                    AND avg_win_streak IS NOT NULL
                    AND max_consecutive_wins >= 1
                )
            )
            """,
            name="win_metric_nullability",
        ),
        CheckConstraint(
            """
            status <> 'completed'
            OR
            (
                (
                    losing_trades = 0
                    AND avg_loss_return IS NULL
                    AND largest_loss_return IS NULL
                    AND avg_loss_streak IS NULL
                    AND max_consecutive_losses = 0
                )
                OR
                (
                    losing_trades > 0
                    AND avg_loss_return IS NOT NULL
                    AND largest_loss_return IS NOT NULL
                    AND avg_loss_streak IS NOT NULL
                    AND max_consecutive_losses >= 1
                )
            )
            """,
            name="loss_metric_nullability",
        ),
        CheckConstraint(
            """
            status <> 'completed'
            OR
            (
                (
                    losing_trades = 0
                    AND profit_factor IS NULL
                )
                OR
                (
                    losing_trades > 0
                    AND profit_factor IS NOT NULL
                )
            )
            """,
            name="profit_factor_nullability",
        ),
        CheckConstraint(
            """
            status <> 'completed'
            OR
            (
                (
                    winning_trades > 0
                    AND losing_trades > 0
                    AND payoff_ratio IS NOT NULL
                )
                OR
                (
                    (
                        winning_trades = 0
                        OR losing_trades = 0
                    )
                    AND payoff_ratio IS NULL
                )
            )
            """,
            name="payoff_ratio_nullability",
        ),
        CheckConstraint(
            """
            status <> 'completed'
            OR
            round_trips = 0
            OR
            (
                min_bars_held <= avg_bars_held
                AND avg_bars_held <= max_bars_held
                AND min_bars_held <= median_bars_held
                AND median_bars_held <= max_bars_held
            )
            """,
            name="holding_period_order",
        ),
        CheckConstraint(
            """
            status <> 'completed'
            OR
            (
                (
                    round_trips = 0
                    AND current_streak_type = 'none'
                    AND current_streak = 0
                )
                OR
                (
                    round_trips > 0
                    AND current_streak_type <> 'none'
                    AND current_streak >= 1
                )
            )
            """,
            name="streak_consistency",
        ),
        CheckConstraint(
            "data_rows >= 1",
            name="data_rows_positive",
        ),
        CheckConstraint(
            "evaluation_start_index >= 0",
            name="start_index_nonnegative",
        ),
        CheckConstraint(
            """
            evaluation_start_index
                < evaluation_end_index
            """,
            name="range_nonempty",
        ),
        CheckConstraint(
            "evaluation_end_index <= data_rows",
            name="end_index_within_data",
        ),
        CheckConstraint(
            "lookback_rows >= 1",
            name="lookback_rows_positive",
        ),
        CheckConstraint(
            "steps_expected >= 1",
            name="steps_expected_positive",
        ),
        CheckConstraint(
            """
            steps_expected =
                evaluation_end_index
                - evaluation_start_index
            """,
            name="steps_match_range",
        ),
        CheckConstraint(
            """
            steps_completed >= 0
            AND steps_completed <= steps_expected
            """,
            name="steps_completed_range",
        ),
        CheckConstraint(
            """
            evaluation_start_at
                <= evaluation_end_at
            """,
            name="data_time_order",
        ),
        CheckConstraint(
            """
            finished_at IS NULL
            OR started_at IS NULL
            OR finished_at >= started_at
            """,
            name="execution_time_order",
        ),
        CheckConstraint(
            "data_sha256 ~ '^[0-9a-f]{64}$'",
            name="data_sha256_lowercase_hex",
        ),
        CheckConstraint(
            """
            (
                status = 'pending'
                AND started_at IS NULL
                AND finished_at IS NULL
                AND steps_completed = 0
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
                AND steps_completed = steps_expected
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
            """,
            name="status_fields",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    checkpoint_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "checkpoints.id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        index=True,
    )

    status: Mapped[
        EvaluationStatus
    ] = mapped_column(
        SQLAlchemyEnum(
            EvaluationStatus,
            name="evaluation_status",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
        default=EvaluationStatus.PENDING,
        server_default=EvaluationStatus.PENDING.value,
    )

    trigger: Mapped[
        EvaluationTrigger
    ] = mapped_column(
        SQLAlchemyEnum(
            EvaluationTrigger,
            name="evaluation_trigger",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )

    policy_mode: Mapped[
        EvaluationPolicyMode
    ] = mapped_column(
        SQLAlchemyEnum(
            EvaluationPolicyMode,
            name="evaluation_policy_mode",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )

    threshold_action: Mapped[
        int | None
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    probability_threshold: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    seed: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    data_scope: Mapped[
        EvaluationDataScope
    ] = mapped_column(
        SQLAlchemyEnum(
            EvaluationDataScope,
            name="evaluation_data_scope",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )

    data_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    data_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    data_rows: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    evaluation_start_index: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    evaluation_end_index: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    evaluation_start_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    evaluation_end_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    lookback_rows: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    steps_expected: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    steps_completed: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    agent_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    always_long_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    always_short_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    agent_vs_always_long_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    balanced_score: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    agent_max_drawdown: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    always_long_max_drawdown: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    always_short_max_drawdown: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    drawdown_improvement: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    net_exposure: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    long_exposure: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    short_exposure: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    flat_exposure: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    market_exposure: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    trade_event_rate: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    round_trip_rate: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    win_rate: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    loss_rate: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    breakeven_rate: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    long_win_rate: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    short_win_rate: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    avg_trade_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    avg_win_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    avg_loss_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    median_trade_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    long_avg_trade_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    short_avg_trade_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    long_median_trade_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    short_median_trade_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    largest_win_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    largest_loss_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    gross_profit_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    gross_loss_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    net_profit_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    profit_factor: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    payoff_ratio: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    avg_bars_held: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    median_bars_held: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    avg_win_streak: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    avg_loss_streak: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    total_fee_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    total_swap_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    total_cost_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    avg_fee_per_trade_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    avg_swap_per_trade_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    avg_cost_per_trade_return: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    cumulative_shaped_reward: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    open_position_return_at_end: Mapped[
        float | None
    ] = mapped_column(
        Float,
        nullable=True,
    )

    trade_events_total: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    round_trips: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    winning_trades: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    losing_trades: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    breakeven_trades: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    long_round_trips: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    short_round_trips: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    open_long_count: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    open_short_count: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    close_long_count: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    close_short_count: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    swap_events: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    min_bars_held: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    max_bars_held: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    max_consecutive_wins: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    max_consecutive_losses: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    current_streak: Mapped[
        int | None
    ] = mapped_column(
        BigInteger,
        nullable=True,
    )

    current_streak_type: Mapped[
        EvaluationCurrentStreakType | None
    ] = mapped_column(
        SQLAlchemyEnum(
            EvaluationCurrentStreakType,
            name="evaluation_current_streak_type",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.now(),
    )

    started_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    finished_at: Mapped[
        datetime | None
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    git_commit: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    git_branch: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    error_type: Mapped[
        str | None
    ] = mapped_column(
        String(255),
        nullable=True,
    )

    error_message: Mapped[
        str | None
    ] = mapped_column(
        Text,
        nullable=True,
    )

    checkpoint: Mapped["Checkpoint"] = relationship(
        "Checkpoint",
        back_populates="evaluations",
        foreign_keys=[checkpoint_id],
    )

