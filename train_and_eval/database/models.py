from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SQLAlchemyEnum,
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
            name="uq_checkpoints_run_id_run_step",
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

    deterministic: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
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

