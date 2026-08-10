from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)

from train_and_eval.environment.contexts import AVAILABLE_CONTEXTS
from train_and_eval.market_data.load_market_data import (
    DATA_DIRECTORY,
    MANIFEST_PATH,
    verify_market_data,
)


SCHEMA_VERSION = 1

SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


HH_MM_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


def parse_hh_mm(value: str) -> int:
    """Convert a validated HH:MM value to minutes after midnight."""
    if not HH_MM_PATTERN.fullmatch(value):
        raise ValueError("must use 24-hour HH:MM format")

    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


def validate_timezone(value: str) -> str:
    """Require a valid IANA time-zone name."""
    try:
        ZoneInfo(value)
    except ZoneInfoNotFoundError as error:
        raise ValueError(
            "must be a valid IANA time zone, "
            "for example America/New_York"
        ) from error

    return value


class RunConfigError(ValueError):
    """Raised when a run configuration cannot be safely loaded."""


class ResumeCompatibilityError(RunConfigError):
    """Raised when a source run is incompatible with resumed training."""


class StrictConfigModel(BaseModel):
    """Base model shared by all run-configuration sections."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class RunSection(StrictConfigModel):
    name: str
    seed: NonNegativeInt

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not SAFE_NAME_PATTERN.fullmatch(value):
            raise ValueError(
                "must start with a letter or digit and contain only "
                "letters, digits, dots, underscores, and hyphens"
            )

        return value



class FreshContinuationSection(StrictConfigModel):
    """Start a completely new model."""

    mode: Literal["fresh"]


class ResumeContinuationSection(StrictConfigModel):
    """Continue training from a checkpoint of an earlier run."""

    mode: Literal["resume"]
    source_run: str
    checkpoint: str = "best"

    @field_validator("source_run", "checkpoint")
    @classmethod
    def validate_reference_name(cls, value: str) -> str:
        if not SAFE_NAME_PATTERN.fullmatch(value):
            raise ValueError(
                "must start with a letter or digit and contain only "
                "letters, digits, dots, underscores, and hyphens"
            )

        return value


ContinuationSection = Annotated[
    FreshContinuationSection | ResumeContinuationSection,
    Field(discriminator="mode"),
]


class DataSection(StrictConfigModel):
    path: str
    train_ratio: float = Field(gt=0.0, lt=1.0)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError(
                "must use forward slashes"
            )

        path = PurePosixPath(value)

        if path.is_absolute():
            raise ValueError(
                "must be relative to the project root"
            )

        if not path.parts or path.parts[0] != "data":
            raise ValueError(
                "must point inside the top-level data directory"
            )

        if ".." in path.parts:
            raise ValueError(
                "cannot contain '..'"
            )

        if path.suffix.lower() != ".parquet":
            raise ValueError(
                "must point to a .parquet file"
            )

        return path.as_posix()



class TrainingSection(StrictConfigModel):
    """
    Define training duration using exactly one selected unit.

    data_epochs:
        Complete chronological passes through the TRAIN split.

    timesteps:
        Exact number of environment step() calls.
    """

    duration_unit: Literal[
        "data_epochs",
        "timesteps",
    ]

    duration_amount: int = Field(
        strict=True,
        ge=1,
    )

    def resolve_training_steps(
        self,
        *,
        steps_per_data_epoch: int,
    ) -> int:
        if steps_per_data_epoch <= 0:
            raise ValueError(
                "steps_per_data_epoch must be greater than zero"
            )

        if self.duration_unit == "data_epochs":
            return (
                self.duration_amount
                * steps_per_data_epoch
            )

        return self.duration_amount


class LoggingSection(StrictConfigModel):
    """Terminal progress-display cadence.

    These settings affect only how often the live dashboard is refreshed.
    They do not change PPO training, checkpointing, evaluation, or metrics.
    """

    training_progress_every_steps: PositiveInt = 10_000
    validation_progress_every_steps: PositiveInt = 2_000


class TrainingMetricsArtifactSection(StrictConfigModel):
    """Persistent PPO-training metric snapshots."""

    enabled: bool = False
    every_steps: PositiveInt = 10_000


class ValidationTrajectoryArtifactSection(StrictConfigModel):
    """Persistence policy for full validation trajectories."""

    mode: Literal[
        "disabled",
        "final_only",
        "all",
    ] = "disabled"


class PlotArtifactSection(StrictConfigModel):
    """Control eager rendering of derived plot artifacts."""

    during_run: bool = False


class ArtifactsSection(StrictConfigModel):
    """Artifact collection and eager-rendering settings.

    This section is intentionally independent from ``logging``. Terminal
    refresh cadence must never implicitly change persisted experiment data.
    """

    training_metrics: TrainingMetricsArtifactSection = Field(
        default_factory=TrainingMetricsArtifactSection
    )
    validation_trajectory: ValidationTrajectoryArtifactSection = Field(
        default_factory=ValidationTrajectoryArtifactSection
    )
    plots: PlotArtifactSection = Field(
        default_factory=PlotArtifactSection
    )


class EnvironmentSection(StrictConfigModel):
    window: PositiveInt
    context: str
    position_side: Literal[
        "long_only",
        "short_only",
        "long_short",
    ]

    market_timezone: str
    rth_open: str
    rth_close: str

    stake_pln: PositiveFloat
    fee_bps: float = Field(ge=0.0)

    swap_bps: float = Field(ge=0.0)
    swap_time: str
    swap_timezone: str

    force_close_on_done: bool

    reward_scale: PositiveFloat

    # Signed coefficients:
    # positive values act as penalties;
    # negative values act as activity bonuses.
    exposure_penalty: float
    turnover_penalty: float
    drawdown_penalty: float

    profit_reward_mult: PositiveFloat
    loss_reward_mult: PositiveFloat

    @field_validator("context")
    @classmethod
    def validate_context(cls, value: str) -> str:
        if not SAFE_NAME_PATTERN.fullmatch(value):
            raise ValueError(
                "must contain only letters, digits, dots, "
                "underscores, and hyphens"
            )

        if value not in AVAILABLE_CONTEXTS:
            available = ", ".join(AVAILABLE_CONTEXTS)
            raise ValueError(
                f"unknown context {value!r}; "
                f"available contexts: {available}"
            )

        return value

    @field_validator(
        "market_timezone",
        "swap_timezone",
    )
    @classmethod
    def validate_timezone_field(cls, value: str) -> str:
        return validate_timezone(value)

    @field_validator(
        "rth_open",
        "rth_close",
        "swap_time",
    )
    @classmethod
    def validate_time_field(cls, value: str) -> str:
        parse_hh_mm(value)
        return value

    @model_validator(mode="after")
    def validate_rth_hours(self) -> Self:
        open_minutes = parse_hh_mm(self.rth_open)
        close_minutes = parse_hh_mm(self.rth_close)

        if close_minutes <= open_minutes:
            raise ValueError(
                "rth_close must be later than rth_open "
                "within the same calendar day"
            )

        return self


class PPOSection(StrictConfigModel):
    policy: Literal["mlp"]
    device: Literal["auto", "cpu", "cuda"]

    hidden_sizes: tuple[PositiveInt, ...] = Field(
        min_length=1,
    )
    activation: Literal["tanh", "relu"]

    n_steps: int = Field(
        strict=True,
        ge=2,
    )
    batch_size: int = Field(
        strict=True,
        ge=2,
    )
    n_epochs: PositiveInt

    learning_rate: PositiveFloat

    gamma: float = Field(
        gt=0.0,
        le=1.0,
    )
    gae_lambda: float = Field(
        gt=0.0,
        le=1.0,
    )

    clip_range: float = Field(
        gt=0.0,
        le=1.0,
    )
    clip_range_vf: float | None
    normalize_advantage: bool

    ent_coef: float = Field(
        ge=0.0,
    )
    vf_coef: float = Field(
        ge=0.0,
    )

    max_grad_norm: PositiveFloat
    target_kl: float | None

    @field_validator("clip_range_vf")
    @classmethod
    def validate_clip_range_vf(
        cls,
        value: float | None,
    ) -> float | None:
        if value is not None and value <= 0.0:
            raise ValueError(
                "clip_range_vf must be greater than zero "
                "or null"
            )

        return value

    @field_validator("target_kl")
    @classmethod
    def validate_target_kl(
        cls,
        value: float | None,
    ) -> float | None:
        if value is not None and value <= 0.0:
            raise ValueError(
                "target_kl must be greater than zero "
                "or null"
            )

        return value

    @model_validator(mode="after")
    def validate_rollout_sizes(self) -> Self:
        if self.batch_size > self.n_steps:
            raise ValueError(
                "batch_size cannot be greater than n_steps"
            )

        if self.n_steps % self.batch_size != 0:
            raise ValueError(
                "batch_size must divide n_steps exactly"
            )

        return self


class EvaluationSection(StrictConfigModel):
    eval_every_steps: PositiveInt
    checkpoint_every_steps: PositiveInt

    device: Literal["auto", "cpu", "cuda"] = "cpu"

    policy_mode: Literal[
        "deterministic_argmax",
        "stochastic_sample",
        "probability_threshold",
    ]
    threshold_action: int | None
    probability_threshold: float | None

    best_metric: Literal["balanced_score"]
    early_stop_patience_evals: PositiveInt

    @model_validator(mode="after")
    def validate_policy_fields(self) -> Self:
        if self.policy_mode != "probability_threshold":
            if self.threshold_action is not None:
                raise ValueError(
                    "threshold_action must be null for "
                    "non-threshold policy modes"
                )

            if self.probability_threshold is not None:
                raise ValueError(
                    "probability_threshold must be null for "
                    "non-threshold policy modes"
                )

            return self

        if self.probability_threshold is None:
            raise ValueError(
                "probability_threshold is required for "
                "probability_threshold mode"
            )

        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError(
                "probability_threshold must be between 0 and 1"
            )

        if self.threshold_action not in {None, 1}:
            raise ValueError(
                "threshold_action must be 1 or null"
            )

        return self


class RunConfig(StrictConfigModel):
    config_schema_version: Literal[SCHEMA_VERSION]

    run: RunSection
    continuation: ContinuationSection
    data: DataSection
    training: TrainingSection
    logging: LoggingSection = Field(default_factory=LoggingSection)
    artifacts: ArtifactsSection = Field(default_factory=ArtifactsSection)
    environment: EnvironmentSection
    ppo: PPOSection
    evaluation: EvaluationSection

    @model_validator(mode="after")
    def validate_evaluation_action_space(self) -> Self:
        evaluation = self.evaluation

        if evaluation.policy_mode != "probability_threshold":
            return self

        if self.environment.position_side == "long_short":
            if evaluation.threshold_action is not None:
                raise ValueError(
                    "threshold_action must be null for "
                    "long_short probability-threshold evaluation"
                )
        elif evaluation.threshold_action != 1:
            raise ValueError(
                "threshold_action must be 1 for long_only or "
                "short_only probability-threshold evaluation"
            )

        return self



RESUME_IMMUTABLE_FIELDS = (
    "environment.window",
    "environment.context",
    "environment.position_side",
    "environment.market_timezone",
    "environment.rth_open",
    "environment.rth_close",
    "environment.swap_time",
    "environment.swap_timezone",
    "ppo.policy",
    "ppo.hidden_sizes",
    "ppo.activation",
)


def _resume_field_value(
    config: RunConfig,
    field_path: str,
) -> Any:
    value: Any = config

    for part in field_path.split("."):
        value = getattr(value, part)

    return value


def validate_resume_compatibility(
    current_config: RunConfig,
    source_config: RunConfig,
) -> None:
    """
    Verify that a source model can be safely loaded for continued training.

    Fresh runs require no source compatibility check. Resume runs cannot
    change parameters that define the neural-network architecture,
    observation space, or action space.
    """
    continuation = current_config.continuation

    if isinstance(continuation, FreshContinuationSection):
        return

    if continuation.source_run != source_config.run.name:
        raise ResumeCompatibilityError(
            "Resume source does not match the source configuration.\n"
            f"Requested source run: {continuation.source_run}\n"
            f"Loaded source run:    {source_config.run.name}"
        )

    if current_config.run.name == source_config.run.name:
        raise ResumeCompatibilityError(
            "A resumed run must have a different name than its source run."
        )

    differences: list[str] = []

    for field_path in RESUME_IMMUTABLE_FIELDS:
        source_value = _resume_field_value(
            source_config,
            field_path,
        )
        current_value = _resume_field_value(
            current_config,
            field_path,
        )

        if current_value != source_value:
            differences.append(
                f"{field_path}: "
                f"source={source_value!r}, "
                f"current={current_value!r}"
            )

    if differences:
        formatted = "\n".join(
            f"- {difference}"
            for difference in differences
        )

        raise ResumeCompatibilityError(
            "Resume configuration is incompatible with the source model.\n"
            "The following structural parameters cannot change:\n"
            f"{formatted}"
        )


@dataclass(frozen=True)
class LoadedRunConfig:
    """Validated config together with reproducibility metadata."""

    path: Path
    sha256: str
    raw_yaml: str
    normalized_json: str
    config: RunConfig
    data_manifest_entry: dict[str, Any] | None


def calculate_config_sha256(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def normalize_config(config: RunConfig) -> str:
    """
    Return stable JSON independent of YAML formatting and key order.
    """
    return json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def load_run_config(
    path: str | Path,
    *,
    verify_data: bool = True,
    data_directory: str | Path = DATA_DIRECTORY,
    manifest_path: str | Path = MANIFEST_PATH,
) -> LoadedRunConfig:
    """
    Load and validate one complete run configuration.

    By default, the referenced market-data file must already have status
    'okay' in the manifest and its current SHA-256 must match.
    """
    config_path = Path(path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Run config does not exist: {config_path}"
        )

    if not config_path.is_file():
        raise RunConfigError(
            f"Run config path is not a file: {config_path}"
        )

    if config_path.suffix.lower() not in {".yml", ".yaml"}:
        raise RunConfigError(
            f"Run config must use .yml or .yaml: {config_path}"
        )

    raw_bytes = config_path.read_bytes()

    try:
        raw_yaml = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RunConfigError(
            f"Run config must be UTF-8 encoded: {config_path}"
        ) from error

    try:
        raw_config = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as error:
        raise RunConfigError(
            f"Invalid YAML in run config: {config_path}\n{error}"
        ) from error

    if raw_config is None:
        raise RunConfigError(
            f"Run config is empty: {config_path}"
        )

    if not isinstance(raw_config, dict):
        raise RunConfigError(
            "The top level of a run config must be a mapping."
        )

    try:
        config = RunConfig.model_validate(raw_config)
    except ValidationError as error:
        raise RunConfigError(
            f"Run config validation failed:\n{error}"
        ) from error

    data_manifest_entry: dict[str, Any] | None = None

    if verify_data:
        _, data_manifest_entry = verify_market_data(
            config.data.path,
            data_directory=data_directory,
            manifest_path=manifest_path,
        )

    return LoadedRunConfig(
        path=config_path,
        sha256=calculate_config_sha256(raw_bytes),
        raw_yaml=raw_yaml,
        normalized_json=normalize_config(config),
        config=config,
        data_manifest_entry=data_manifest_entry,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a PPO_AGENT run configuration."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML run configuration.",
    )
    args = parser.parse_args()

    loaded = load_run_config(args.config)

    print("Run config validation: OK")
    print(f"Config: {loaded.path}")
    print(f"Config SHA-256: {loaded.sha256}")
    print(f"Run name: {loaded.config.run.name}")
    print(f"Seed: {loaded.config.run.seed}")
    print(f"Start mode: {loaded.config.continuation.mode}")

    if isinstance(
        loaded.config.continuation,
        ResumeContinuationSection,
    ):
        print(
            "Source run: "
            f"{loaded.config.continuation.source_run}"
        )
        print(
            "Source checkpoint: "
            f"{loaded.config.continuation.checkpoint}"
        )

    print(f"Data: {loaded.config.data.path}")
    print(
        "Training duration: "
        f"{loaded.config.training.duration_amount} "
        f"{loaded.config.training.duration_unit}"
    )

    if loaded.data_manifest_entry is not None:
        print(
            "Data SHA-256: "
            f"{loaded.data_manifest_entry['sha256']}"
        )

    print()
    print("Normalized config:")
    print(
        json.dumps(
            json.loads(loaded.normalized_json),
            indent=2,
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
