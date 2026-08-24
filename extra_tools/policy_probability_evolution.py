from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sqlalchemy import select

from extra_tools.policy_probability_diagnostic import (
    SourceRun,
    collect_probability_data,
    parse_int_list,
)
from train_and_eval.database.models import (
    CheckpointSaveReason,
    EvaluationDataScope,
    EvaluationPolicyMode,
    EvaluationStatus,
    TrainingMetric,
)
from train_and_eval.database.session import (
    create_database_engine,
    create_session_factory,
)
from train_and_eval.runs.queries import load_run


DEFAULT_RUNS = (34, 36, 37)

DEFAULT_OUTPUT_DIRECTORY = Path(
    "/tmp/ppo_policy_probability_evolution"
)

TRAINING_METRIC_FIELDS = (
    "ep_reward",
    "ep_len",
    "rollout_reward_mean",
    "rollout_reward_sum",
    "approx_kl",
    "clip_fraction",
    "clip_range",
    "entropy_loss",
    "explained_variance",
    "learning_rate",
    "loss",
    "n_updates",
    "policy_gradient_loss",
    "value_loss",
    "value_target_mean",
    "value_target_std",
    "value_prediction_mean",
    "value_prediction_std",
    "value_error_mean",
    "value_error_std",
    "value_target_prediction_corr",
    "post_train_value_prediction_mean",
    "post_train_value_prediction_std",
    "post_train_value_error_mean",
    "post_train_value_error_std",
    "post_train_value_mse",
    "post_train_explained_variance",
    "post_train_value_target_prediction_corr",
)


@dataclass(frozen=True)
class SourceCheckpoint:
    run_id: int
    run_name: str
    seed: int
    checkpoint_id: int
    run_step: int
    model_step: int
    save_reason: str
    evaluation_id: int
    evaluation_trigger: str


def enum_value(value):
    return getattr(value, "value", value)


def select_source_checkpoints(run) -> list[SourceCheckpoint]:
    eligible_reasons = {
        CheckpointSaveReason.PERIODIC,
        CheckpointSaveReason.FINAL,
    }

    checkpoints = sorted(
        (
            checkpoint
            for checkpoint in run.checkpoints
            if checkpoint.save_reason in eligible_reasons
        ),
        key=lambda checkpoint: (
            int(checkpoint.model_step),
            int(checkpoint.id),
        ),
    )

    if not checkpoints:
        raise RuntimeError(
            f"Run #{run.id} has no periodic or final checkpoints."
        )

    sources: list[SourceCheckpoint] = []

    for checkpoint in checkpoints:
        evaluations = [
            evaluation
            for evaluation in checkpoint.evaluations
            if (
                evaluation.status == EvaluationStatus.COMPLETED
                and evaluation.data_scope
                == EvaluationDataScope.RUN_VALIDATION
                and evaluation.policy_mode
                == EvaluationPolicyMode.DETERMINISTIC_ARGMAX
            )
        ]

        if len(evaluations) != 1:
            raise RuntimeError(
                f"Checkpoint #{checkpoint.id} from run #{run.id} "
                "expected exactly one completed deterministic-argmax "
                "run-validation evaluation, found "
                f"{len(evaluations)}."
            )

        evaluation = evaluations[0]

        sources.append(
            SourceCheckpoint(
                run_id=int(run.id),
                run_name=str(run.name),
                seed=int(run.seed),
                checkpoint_id=int(checkpoint.id),
                run_step=int(checkpoint.run_step),
                model_step=int(checkpoint.model_step),
                save_reason=str(
                    enum_value(checkpoint.save_reason)
                ),
                evaluation_id=int(evaluation.id),
                evaluation_trigger=str(
                    enum_value(evaluation.trigger)
                ),
            )
        )

    return sources


def load_source_checkpoints(
    session_factory,
    run_ids: list[int],
) -> list[SourceCheckpoint]:
    sources: list[SourceCheckpoint] = []

    with session_factory() as session:
        for run_id in run_ids:
            run = load_run(
                session,
                run_id=run_id,
            )
            sources.extend(
                select_source_checkpoints(run)
            )

    return sources


def distribution_metrics(
    values: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    if values.size == 0:
        raise ValueError(
            "Probability distribution must not be empty."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Probability distribution must contain only finite values."
        )

    return {
        "fraction_le_010": float(
            np.mean(values <= 0.10)
        ),
        "fraction_le_030": float(
            np.mean(values <= 0.30)
        ),
        "fraction_lt_050": float(
            np.mean(values < 0.50)
        ),
        "fraction_ge_090": float(
            np.mean(values >= 0.90)
        ),
    }


def collect_checkpoint_probability_data(
    *,
    source: SourceCheckpoint,
    session_factory,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame, summary = collect_probability_data(
        source=SourceRun(
            run_id=source.run_id,
            name=source.run_name,
            seed=source.seed,
            checkpoint_id=source.checkpoint_id,
        ),
        session_factory=session_factory,
    )

    frame = frame.copy()
    frame["run_step"] = source.run_step
    frame["model_step"] = source.model_step
    frame["save_reason"] = source.save_reason
    frame["evaluation_id"] = source.evaluation_id
    frame["evaluation_trigger"] = (
        source.evaluation_trigger
    )

    ordered_columns = [
        "run_id",
        "run_name",
        "seed",
        "checkpoint_id",
        "run_step",
        "model_step",
        "save_reason",
        "evaluation_id",
        "evaluation_trigger",
        "step",
        "execution_index",
        "execution_timestamp",
        "action",
        "p_flat",
        "p_long",
    ]

    frame = frame.loc[
        :,
        ordered_columns,
    ]

    p_long_values = frame[
        "p_long"
    ].to_numpy(dtype=float)

    summary = dict(summary)
    summary.update(
        {
            "run_step": source.run_step,
            "model_step": source.model_step,
            "save_reason": source.save_reason,
            "evaluation_id": source.evaluation_id,
            "evaluation_trigger": (
                source.evaluation_trigger
            ),
            **distribution_metrics(
                p_long_values
            ),
        }
    )

    return frame, summary


def empirical_wasserstein_distance(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    left = np.asarray(
        left,
        dtype=float,
    ).reshape(-1)
    right = np.asarray(
        right,
        dtype=float,
    ).reshape(-1)

    if left.size == 0 or right.size == 0:
        raise ValueError(
            "Both empirical distributions must be non-empty."
        )

    if (
        not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ValueError(
            "Empirical distributions must contain only finite values."
        )

    if left.size == right.size:
        return float(
            np.mean(
                np.abs(
                    np.sort(left)
                    - np.sort(right)
                )
            )
        )

    points = max(
        left.size,
        right.size,
    )
    quantiles = (
        np.arange(points, dtype=float)
        + 0.5
    ) / points

    return float(
        np.mean(
            np.abs(
                np.quantile(left, quantiles)
                - np.quantile(right, quantiles)
            )
        )
    )


def build_probability_transitions(
    probability_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
) -> pd.DataFrame:
    delta_columns = (
        "mean_p_long",
        "median_p_long",
        "std_p_long",
        "p10_p_long",
        "p90_p_long",
        "fraction_le_010",
        "fraction_le_030",
        "fraction_lt_050",
        "fraction_ge_050",
        "fraction_ge_070",
        "fraction_ge_090",
        "argmax_market_exposure",
        "argmax_round_trips",
        "argmax_agent_return",
        "argmax_max_drawdown",
        "argmax_balanced_score",
    )

    rows: list[dict[str, object]] = []

    for run_id, run_summary in summary_frame.groupby(
        "run_id",
        sort=False,
    ):
        ordered = run_summary.sort_values(
            ["model_step", "checkpoint_id"]
        ).reset_index(drop=True)

        for index in range(1, len(ordered)):
            previous = ordered.iloc[index - 1]
            current = ordered.iloc[index]

            previous_values = probability_frame.loc[
                probability_frame["checkpoint_id"]
                == int(previous["checkpoint_id"]),
                "p_long",
            ].to_numpy(dtype=float)

            current_values = probability_frame.loc[
                probability_frame["checkpoint_id"]
                == int(current["checkpoint_id"]),
                "p_long",
            ].to_numpy(dtype=float)

            row: dict[str, object] = {
                "run_id": int(run_id),
                "run_name": str(current["run_name"]),
                "seed": int(current["seed"]),
                "from_checkpoint_id": int(
                    previous["checkpoint_id"]
                ),
                "to_checkpoint_id": int(
                    current["checkpoint_id"]
                ),
                "from_evaluation_id": int(
                    previous["evaluation_id"]
                ),
                "to_evaluation_id": int(
                    current["evaluation_id"]
                ),
                "from_model_step": int(
                    previous["model_step"]
                ),
                "to_model_step": int(
                    current["model_step"]
                ),
                "model_step_delta": int(
                    current["model_step"]
                    - previous["model_step"]
                ),
                "p_long_wasserstein_distance": (
                    empirical_wasserstein_distance(
                        previous_values,
                        current_values,
                    )
                ),
            }

            for column in delta_columns:
                row[f"delta_{column}"] = float(
                    current[column]
                    - previous[column]
                )

            rows.append(row)

    return pd.DataFrame(rows)


def load_training_metrics_frame(
    session_factory,
    sources: list[SourceCheckpoint],
) -> pd.DataFrame:
    run_ids = sorted(
        {
            source.run_id
            for source in sources
        }
    )

    run_metadata = {
        source.run_id: (
            source.run_name,
            source.seed,
        )
        for source in sources
    }

    with session_factory() as session:
        records = list(
            session.scalars(
                select(TrainingMetric)
                .where(
                    TrainingMetric.run_id.in_(
                        run_ids
                    )
                )
                .order_by(
                    TrainingMetric.run_id,
                    TrainingMetric.model_step,
                )
            ).all()
        )

        rows: list[dict[str, object]] = []

        for record in records:
            run_id = int(record.run_id)
            run_name, seed = run_metadata[run_id]

            row: dict[str, object] = {
                "training_metric_id": int(record.id),
                "run_id": run_id,
                "run_name": run_name,
                "seed": seed,
                "run_step": int(record.run_step),
                "model_step": int(record.model_step),
                "rollout_number": int(
                    record.rollout_number
                ),
            }

            for field in TRAINING_METRIC_FIELDS:
                row[field] = getattr(record, field)

            rows.append(row)

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise RuntimeError(
            "No training metrics were found for the selected runs."
        )

    return frame


def build_checkpoint_diagnostics(
    summary_frame: pd.DataFrame,
    training_frame: pd.DataFrame,
) -> pd.DataFrame:
    training = training_frame.drop(
        columns=[
            "run_name",
            "seed",
        ]
    ).rename(
        columns={
            "run_step": "training_metric_run_step",
        }
    )

    merged = summary_frame.merge(
        training,
        on=[
            "run_id",
            "model_step",
        ],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing = merged.loc[
        merged["_merge"] != "both",
        [
            "run_id",
            "checkpoint_id",
            "model_step",
        ],
    ]

    if not missing.empty:
        raise RuntimeError(
            "Checkpoint steps without an exact training metric match: "
            f"{missing.to_dict(orient='records')}"
        )

    merged = merged.drop(
        columns=["_merge"]
    )

    mismatched_steps = merged.loc[
        merged["run_step"]
        != merged["training_metric_run_step"]
    ]

    if not mismatched_steps.empty:
        raise RuntimeError(
            "Checkpoint and training-metric run steps do not match."
        )

    return merged.sort_values(
        ["run_id", "model_step"],
        ignore_index=True,
    )


def _sources_by_run(
    sources: list[SourceCheckpoint],
) -> list[tuple[int, list[SourceCheckpoint]]]:
    order = list(
        dict.fromkeys(
            source.run_id
            for source in sources
        )
    )

    return [
        (
            run_id,
            sorted(
                (
                    source
                    for source in sources
                    if source.run_id == run_id
                ),
                key=lambda source: (
                    source.model_step,
                    source.checkpoint_id,
                ),
            ),
        )
        for run_id in order
    ]


def _run_colors(
    sources: list[SourceCheckpoint],
) -> dict[int, object]:
    run_ids = list(
        dict.fromkeys(
            source.run_id
            for source in sources
        )
    )
    color_map = plt.get_cmap("tab10")

    return {
        run_id: color_map(index % 10)
        for index, run_id in enumerate(run_ids)
    }


def plot_histogram_grid(
    probability_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    sources: list[SourceCheckpoint],
    *,
    bins: int,
    output_path: Path,
    dpi: int,
) -> None:
    grouped_sources = _sources_by_run(
        sources
    )
    columns = max(
        len(run_sources)
        for _, run_sources in grouped_sources
    )
    rows = len(grouped_sources)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            max(4.0 * columns, 8.0),
            max(3.2 * rows, 4.0),
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    for row_index, (
        run_id,
        run_sources,
    ) in enumerate(grouped_sources):
        checkpoint_colors = plt.get_cmap(
            "viridis"
        )(
            np.linspace(
                0.15,
                0.85,
                len(run_sources),
            )
        )

        for column_index, source in enumerate(
            run_sources
        ):
            axis = axes[
                row_index,
                column_index,
            ]

            values = probability_frame.loc[
                probability_frame["checkpoint_id"]
                == source.checkpoint_id,
                "p_long",
            ].to_numpy(dtype=float)

            summary = summary_frame.loc[
                summary_frame["checkpoint_id"]
                == source.checkpoint_id
            ].iloc[0]

            axis.hist(
                values,
                bins=bin_edges,
                density=True,
                histtype="stepfilled",
                alpha=0.65,
                color=checkpoint_colors[
                    column_index
                ],
            )
            axis.axvline(
                0.50,
                color="black",
                linestyle="--",
                linewidth=1.0,
            )
            axis.set_xlim(0.0, 1.0)
            axis.grid(True, alpha=0.20)
            axis.set_title(
                f"Run #{run_id} / seed {source.seed}\n"
                f"step {source.model_step:,} | "
                f"mean {summary.mean_p_long:.3f} | "
                f"std {summary.std_p_long:.3f}",
                fontsize=9,
            )

            if row_index == rows - 1:
                axis.set_xlabel("P(LONG)")

            if column_index == 0:
                axis.set_ylabel(
                    "Probability density"
                )

        for column_index in range(
            len(run_sources),
            columns,
        ):
            axes[
                row_index,
                column_index,
            ].set_visible(False)

    figure.suptitle(
        "P(LONG) distribution across training checkpoints",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_confidence_curve_grid(
    probability_frame: pd.DataFrame,
    sources: list[SourceCheckpoint],
    *,
    curve_points: int,
    output_path: Path,
    dpi: int,
) -> None:
    grouped_sources = _sources_by_run(
        sources
    )
    thresholds = np.linspace(
        0.0,
        1.0,
        curve_points,
    )

    figure, axes = plt.subplots(
        len(grouped_sources),
        1,
        figsize=(
            11.0,
            max(3.8 * len(grouped_sources), 5.0),
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row_index, (
        run_id,
        run_sources,
    ) in enumerate(grouped_sources):
        axis = axes[row_index, 0]
        colors = plt.get_cmap("viridis")(
            np.linspace(
                0.15,
                0.85,
                len(run_sources),
            )
        )

        for source, color in zip(
            run_sources,
            colors,
            strict=True,
        ):
            values = probability_frame.loc[
                probability_frame["checkpoint_id"]
                == source.checkpoint_id,
                "p_long",
            ].to_numpy(dtype=float)

            fractions = np.asarray(
                [
                    np.mean(values >= threshold)
                    for threshold in thresholds
                ],
                dtype=float,
            )

            axis.plot(
                thresholds,
                fractions,
                color=color,
                linewidth=2.0,
                label=f"step {source.model_step:,}",
            )

        axis.axvline(
            0.50,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.xaxis.set_major_formatter(
            PercentFormatter(
                xmax=1.0,
                decimals=0,
            )
        )
        axis.yaxis.set_major_formatter(
            PercentFormatter(
                xmax=1.0,
                decimals=0,
            )
        )
        axis.grid(True, alpha=0.25)
        axis.set_title(
            f"Run #{run_id} / seed {run_sources[0].seed}"
        )
        axis.set_ylabel(
            "Fraction with P(LONG) >= threshold"
        )
        axis.legend(
            ncol=min(4, len(run_sources)),
            fontsize=8,
        )

    axes[-1, 0].set_xlabel(
        "Minimum P(LONG) required"
    )
    figure.suptitle(
        "P(LONG) confidence curves across checkpoints",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_probability_evolution(
    summary_frame: pd.DataFrame,
    sources: list[SourceCheckpoint],
    *,
    output_path: Path,
    dpi: int,
) -> None:
    colors = _run_colors(sources)

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 9.0),
        sharex=True,
    )

    for run_id, group in summary_frame.groupby(
        "run_id",
        sort=False,
    ):
        group = group.sort_values(
            "model_step"
        )
        x = group["model_step"].to_numpy(
            dtype=float
        ) / 1000.0
        label = (
            f"Run #{int(run_id)} / "
            f"seed {int(group.iloc[0]['seed'])}"
        )
        color = colors[int(run_id)]

        axes[0, 0].plot(
            x,
            group["median_p_long"],
            marker="o",
            color=color,
            linewidth=2.0,
            label=label,
        )
        axes[0, 0].fill_between(
            x,
            group["p10_p_long"],
            group["p90_p_long"],
            color=color,
            alpha=0.12,
        )

        axes[0, 1].plot(
            x,
            group["std_p_long"],
            marker="o",
            color=color,
            linewidth=2.0,
            label=label,
        )

        axes[1, 0].plot(
            x,
            group["fraction_ge_050"],
            marker="o",
            color=color,
            linewidth=2.0,
            label=f"{label} / >=0.50",
        )
        axes[1, 0].plot(
            x,
            group["fraction_ge_070"],
            marker="s",
            linestyle="--",
            color=color,
            linewidth=1.5,
            label=f"{label} / >=0.70",
        )

        axes[1, 1].plot(
            x,
            group["fraction_lt_050"],
            marker="o",
            color=color,
            linewidth=2.0,
            label=f"{label} / <0.50",
        )
        axes[1, 1].plot(
            x,
            group["fraction_le_030"],
            marker="s",
            linestyle="--",
            color=color,
            linewidth=1.5,
            label=f"{label} / <=0.30",
        )

    axes[0, 0].set_title(
        "Median P(LONG) with P10-P90 band"
    )
    axes[0, 0].set_ylabel("P(LONG)")
    axes[0, 0].set_ylim(0.0, 1.0)

    axes[0, 1].set_title(
        "P(LONG) standard deviation"
    )
    axes[0, 1].set_ylabel("Standard deviation")
    axes[0, 1].set_ylim(bottom=0.0)

    axes[1, 0].set_title(
        "LONG-side probability mass"
    )
    axes[1, 0].set_ylabel("Fraction of observations")
    axes[1, 0].set_ylim(0.0, 1.0)

    axes[1, 1].set_title(
        "FLAT-side probability mass"
    )
    axes[1, 1].set_ylabel("Fraction of observations")
    axes[1, 1].set_ylim(0.0, 1.0)

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("Model steps (thousands)")

    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].legend(fontsize=7)

    axes[1, 0].yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )
    axes[1, 1].yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )

    figure.suptitle(
        "P(LONG) distribution evolution",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_training_metric_grid(
    training_frame: pd.DataFrame,
    sources: list[SourceCheckpoint],
    metric_specs: tuple[tuple[str, str], ...],
    *,
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    columns = 3
    rows = math.ceil(
        len(metric_specs) / columns
    )
    colors = _run_colors(sources)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(14.0, 3.6 * rows),
        sharex=True,
        squeeze=False,
    )

    checkpoint_steps = sorted(
        {
            source.model_step
            for source in sources
        }
    )

    for axis, (column, label) in zip(
        axes.flat,
        metric_specs,
        strict=False,
    ):
        for run_id, group in training_frame.groupby(
            "run_id",
            sort=False,
        ):
            group = group.sort_values(
                "model_step"
            )
            axis.plot(
                group["model_step"] / 1000.0,
                group[column],
                marker="o",
                markersize=3.0,
                linewidth=1.5,
                color=colors[int(run_id)],
                label=(
                    f"Run #{int(run_id)} / "
                    f"seed {int(group.iloc[0]['seed'])}"
                ),
            )

        for step in checkpoint_steps:
            axis.axvline(
                step / 1000.0,
                color="black",
                linestyle=":",
                linewidth=0.7,
                alpha=0.20,
            )

        axis.set_title(label)
        axis.set_xlabel(
            "Model steps (thousands)"
        )
        axis.grid(True, alpha=0.25)

    for axis in axes.flat[
        len(metric_specs):
    ]:
        axis.set_visible(False)

    axes[0, 0].legend(fontsize=8)
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_ppo_diagnostics(
    training_frame: pd.DataFrame,
    sources: list[SourceCheckpoint],
    *,
    output_path: Path,
    dpi: int,
) -> None:
    metric_specs = (
        ("approx_kl", "Approximate KL"),
        ("clip_fraction", "Clip fraction"),
        ("entropy_loss", "Entropy loss"),
        (
            "explained_variance",
            "SB3 explained variance",
        ),
        (
            "policy_gradient_loss",
            "Policy-gradient loss",
        ),
        ("value_loss", "Value loss"),
        (
            "rollout_reward_mean",
            "Rollout reward mean",
        ),
        (
            "post_train_value_mse",
            "Post-train value MSE",
        ),
        (
            "post_train_value_target_prediction_corr",
            "Post-train target/prediction correlation",
        ),
    )

    _plot_training_metric_grid(
        training_frame,
        sources,
        metric_specs,
        title="PPO diagnostics across rollout updates",
        output_path=output_path,
        dpi=dpi,
    )


def plot_critic_diagnostics(
    training_frame: pd.DataFrame,
    sources: list[SourceCheckpoint],
    *,
    output_path: Path,
    dpi: int,
) -> None:
    metric_specs = (
        ("value_target_std", "Value-target std"),
        (
            "value_prediction_std",
            "Pre-train prediction std",
        ),
        (
            "value_error_std",
            "Pre-train value-error std",
        ),
        (
            "value_target_prediction_corr",
            "Pre-train target/prediction correlation",
        ),
        (
            "post_train_value_prediction_std",
            "Post-train prediction std",
        ),
        (
            "post_train_value_error_std",
            "Post-train value-error std",
        ),
        (
            "post_train_value_mse",
            "Post-train value MSE",
        ),
        (
            "post_train_explained_variance",
            "Post-train explained variance",
        ),
        (
            "post_train_value_target_prediction_corr",
            "Post-train target/prediction correlation",
        ),
    )

    _plot_training_metric_grid(
        training_frame,
        sources,
        metric_specs,
        title="Critic diagnostics across rollout updates",
        output_path=output_path,
        dpi=dpi,
    )


def print_sources(
    sources: list[SourceCheckpoint],
) -> None:
    print("SOURCE CHECKPOINTS")
    print("-" * 112)

    for source in sources:
        print(
            f"run=#{source.run_id}"
            f" | seed={source.seed}"
            f" | step={source.model_step:,}"
            f" | checkpoint={source.checkpoint_id}"
            f" | evaluation={source.evaluation_id}"
            f" | reason={source.save_reason}"
        )


def print_transition_summary(
    transition_frame: pd.DataFrame,
) -> None:
    print()
    print("=" * 112)
    print("CHECKPOINT-TO-CHECKPOINT P(LONG) TRANSITIONS")
    print("=" * 112)
    print(
        f"{'run':>4} "
        f"{'seed':>4} "
        f"{'from':>10} "
        f"{'to':>10} "
        f"{'W1':>9} "
        f"{'d_mean':>9} "
        f"{'d_std':>9} "
        f"{'d_>=.50':>9} "
        f"{'d_>=.70':>9} "
        f"{'d_<=.30':>9}"
    )
    print("-" * 112)

    for row in transition_frame.itertuples(
        index=False
    ):
        print(
            f"{row.run_id:>4d} "
            f"{row.seed:>4d} "
            f"{row.from_model_step:>10,d} "
            f"{row.to_model_step:>10,d} "
            f"{row.p_long_wasserstein_distance:>9.4f} "
            f"{row.delta_mean_p_long:>9.4f} "
            f"{row.delta_std_p_long:>9.4f} "
            f"{row.delta_fraction_ge_050:>9.2%} "
            f"{row.delta_fraction_ge_070:>9.2%} "
            f"{row.delta_fraction_le_030:>9.2%}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze P(LONG) distribution evolution across "
            "periodic and final checkpoints of existing PPO runs."
        )
    )

    parser.add_argument(
        "--runs",
        default=",".join(
            str(value)
            for value in DEFAULT_RUNS
        ),
        help=(
            "Comma-separated source run IDs. "
            "Default: 34,36,37"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory for plots and CSV outputs. "
            "Default: /tmp/"
            "ppo_policy_probability_evolution"
        ),
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Histogram bin count. Default: 50",
    )

    parser.add_argument(
        "--curve-points",
        type=int,
        default=201,
        help=(
            "Number of threshold points between "
            "0 and 1. Default: 201"
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=170,
        help="PNG resolution. Default: 170",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.bins < 5:
        raise SystemExit(
            "--bins must be at least 5"
        )

    if args.curve_points < 2:
        raise SystemExit(
            "--curve-points must be at least 2"
        )

    if args.dpi < 50:
        raise SystemExit(
            "--dpi must be at least 50"
        )

    run_ids = parse_int_list(
        args.runs
    )

    if len(set(run_ids)) != len(run_ids):
        raise SystemExit(
            "--runs must not contain duplicate run IDs"
        )

    engine = create_database_engine()
    session_factory = create_session_factory(
        engine
    )

    try:
        sources = load_source_checkpoints(
            session_factory,
            run_ids,
        )

        print_sources(sources)

        all_frames: list[pd.DataFrame] = []
        summaries: list[
            dict[str, object]
        ] = []

        # Finish every replay before writing anything.
        # Historical replay requires a clean Git repository.
        for source in sources:
            frame, summary = (
                collect_checkpoint_probability_data(
                    source=source,
                    session_factory=(
                        session_factory
                    ),
                )
            )
            all_frames.append(frame)
            summaries.append(summary)

        training_frame = (
            load_training_metrics_frame(
                session_factory,
                sources,
            )
        )

    finally:
        engine.dispose()

    probability_frame = pd.concat(
        all_frames,
        ignore_index=True,
    ).sort_values(
        [
            "run_id",
            "model_step",
            "step",
        ],
        ignore_index=True,
    )

    summary_frame = pd.DataFrame(
        summaries
    ).sort_values(
        ["run_id", "model_step"],
        ignore_index=True,
    )

    transition_frame = (
        build_probability_transitions(
            probability_frame,
            summary_frame,
        )
    )

    checkpoint_diagnostics = (
        build_checkpoint_diagnostics(
            summary_frame,
            training_frame,
        )
    )

    print_transition_summary(
        transition_frame
    )

    output_directory = (
        args.output_dir
        .expanduser()
        .resolve()
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_paths = {
        "policy probabilities": (
            output_directory
            / "policy_probabilities.csv"
        ),
        "probability summary": (
            output_directory
            / "probability_summary.csv"
        ),
        "probability transitions": (
            output_directory
            / "probability_transitions.csv"
        ),
        "training metrics": (
            output_directory
            / "training_metrics.csv"
        ),
        "checkpoint diagnostics": (
            output_directory
            / "checkpoint_diagnostics.csv"
        ),
        "histogram grid": (
            output_directory
            / "p_long_histogram_grid.png"
        ),
        "confidence curves": (
            output_directory
            / "p_long_confidence_curve_grid.png"
        ),
        "probability evolution": (
            output_directory
            / "p_long_evolution.png"
        ),
        "PPO diagnostics": (
            output_directory
            / "ppo_diagnostics.png"
        ),
        "critic diagnostics": (
            output_directory
            / "critic_diagnostics.png"
        ),
    }

    probability_frame.to_csv(
        artifact_paths["policy probabilities"],
        index=False,
    )
    summary_frame.to_csv(
        artifact_paths["probability summary"],
        index=False,
    )
    transition_frame.to_csv(
        artifact_paths["probability transitions"],
        index=False,
    )
    training_frame.to_csv(
        artifact_paths["training metrics"],
        index=False,
    )
    checkpoint_diagnostics.to_csv(
        artifact_paths["checkpoint diagnostics"],
        index=False,
    )

    plot_histogram_grid(
        probability_frame,
        summary_frame,
        sources,
        bins=args.bins,
        output_path=artifact_paths[
            "histogram grid"
        ],
        dpi=args.dpi,
    )
    plot_confidence_curve_grid(
        probability_frame,
        sources,
        curve_points=args.curve_points,
        output_path=artifact_paths[
            "confidence curves"
        ],
        dpi=args.dpi,
    )
    plot_probability_evolution(
        summary_frame,
        sources,
        output_path=artifact_paths[
            "probability evolution"
        ],
        dpi=args.dpi,
    )
    plot_ppo_diagnostics(
        training_frame,
        sources,
        output_path=artifact_paths[
            "PPO diagnostics"
        ],
        dpi=args.dpi,
    )
    plot_critic_diagnostics(
        training_frame,
        sources,
        output_path=artifact_paths[
            "critic diagnostics"
        ],
        dpi=args.dpi,
    )

    print()
    print("=" * 112)
    print("ARTIFACTS")
    print("=" * 112)

    for label, path in artifact_paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
