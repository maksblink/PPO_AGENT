from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from train_and_eval.database.session import (
    create_database_engine,
    create_session_factory,
)
from train_and_eval.evaluation.service import (
    replay_run_validation_checkpoint,
)
from train_and_eval.runs.queries import load_run


DEFAULT_RUNS = (34, 36, 37)

DEFAULT_OUTPUT_DIRECTORY = Path(
    "/tmp/ppo_policy_probability_diag"
)


@dataclass(frozen=True)
class SourceRun:
    run_id: int
    name: str
    seed: int
    checkpoint_id: int


def enum_value(value):
    return getattr(value, "value", value)


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []

    for raw in value.split(","):
        raw = raw.strip()

        if not raw:
            continue

        parsed = int(raw)

        if parsed < 1:
            raise argparse.ArgumentTypeError(
                "Run IDs must be positive integers."
            )

        result.append(parsed)

    if not result:
        raise argparse.ArgumentTypeError(
            "At least one run ID is required."
        )

    return result


def load_source_runs(
    session_factory,
    run_ids: list[int],
) -> list[SourceRun]:
    sources: list[SourceRun] = []

    with session_factory() as session:
        for run_id in run_ids:
            run = load_run(
                session,
                run_id=run_id,
            )

            final_checkpoints = [
                checkpoint
                for checkpoint in run.checkpoints
                if enum_value(
                    checkpoint.save_reason
                ) == "final"
            ]

            if len(final_checkpoints) != 1:
                raise RuntimeError(
                    f"Run #{run_id} expected exactly "
                    "one final checkpoint, found "
                    f"{len(final_checkpoints)}."
                )

            checkpoint = final_checkpoints[0]

            sources.append(
                SourceRun(
                    run_id=int(run.id),
                    name=str(run.name),
                    seed=int(run.seed),
                    checkpoint_id=int(
                        checkpoint.id
                    ),
                )
            )

    return sources


def collect_probability_data(
    *,
    source: SourceRun,
    session_factory,
) -> tuple[pd.DataFrame, dict[str, object]]:
    print()
    print("=" * 88)
    print(
        f"RUN #{source.run_id}"
        f" | seed={source.seed}"
        f" | checkpoint={source.checkpoint_id}"
    )
    print(source.name)
    print("=" * 88)

    replay = replay_run_validation_checkpoint(
        checkpoint_id=source.checkpoint_id,
        session_factory=session_factory,
        policy_mode="deterministic_argmax",
        threshold_action=None,
        probability_threshold=None,
    )

    trace = replay.policy_trace

    lengths = {
        "indices": len(trace.execution_indices),
        "timestamps": len(
            trace.execution_timestamps
        ),
        "actions": len(trace.actions),
        "probabilities": len(
            trace.probabilities
        ),
    }

    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            f"Run #{source.run_id} trace lengths "
            f"do not agree: {lengths}"
        )

    rows: list[dict[str, object]] = []

    for step, (
        execution_index,
        timestamp,
        action,
        probabilities,
    ) in enumerate(
        zip(
            trace.execution_indices,
            trace.execution_timestamps,
            trace.actions,
            trace.probabilities,
            strict=True,
        )
    ):
        if len(probabilities) != 2:
            raise RuntimeError(
                f"Run #{source.run_id} uses "
                f"{len(probabilities)} actions; "
                "this diagnostic currently expects "
                "the two-action LONG/FLAT policy."
            )

        p_flat = float(probabilities[0])
        p_long = float(probabilities[1])

        probability_sum = p_flat + p_long

        if not np.isclose(
            probability_sum,
            1.0,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"Run #{source.run_id}, step {step}: "
                "policy probabilities do not sum "
                "to one."
            )

        rows.append(
            {
                "run_id": source.run_id,
                "run_name": source.name,
                "seed": source.seed,
                "checkpoint_id": (
                    source.checkpoint_id
                ),
                "step": step,
                "execution_index": int(
                    execution_index
                ),
                "execution_timestamp": timestamp,
                "action": int(action),
                "p_flat": p_flat,
                "p_long": p_long,
            }
        )

    frame = pd.DataFrame(rows)

    p_long_values = frame[
        "p_long"
    ].to_numpy(dtype=float)

    metrics = replay.metrics

    summary: dict[str, object] = {
        "run_id": source.run_id,
        "run_name": source.name,
        "seed": source.seed,
        "checkpoint_id": source.checkpoint_id,
        "steps": len(frame),
        "mean_p_long": float(
            np.mean(p_long_values)
        ),
        "std_p_long": float(
            np.std(
                p_long_values,
                ddof=0,
            )
        ),
        "min_p_long": float(
            np.min(p_long_values)
        ),
        "p05_p_long": float(
            np.quantile(
                p_long_values,
                0.05,
            )
        ),
        "p10_p_long": float(
            np.quantile(
                p_long_values,
                0.10,
            )
        ),
        "p25_p_long": float(
            np.quantile(
                p_long_values,
                0.25,
            )
        ),
        "median_p_long": float(
            np.quantile(
                p_long_values,
                0.50,
            )
        ),
        "p75_p_long": float(
            np.quantile(
                p_long_values,
                0.75,
            )
        ),
        "p90_p_long": float(
            np.quantile(
                p_long_values,
                0.90,
            )
        ),
        "p95_p_long": float(
            np.quantile(
                p_long_values,
                0.95,
            )
        ),
        "max_p_long": float(
            np.max(p_long_values)
        ),
        "fraction_ge_030": float(
            np.mean(
                p_long_values >= 0.30
            )
        ),
        "fraction_ge_040": float(
            np.mean(
                p_long_values >= 0.40
            )
        ),
        "fraction_ge_050": float(
            np.mean(
                p_long_values >= 0.50
            )
        ),
        "fraction_ge_060": float(
            np.mean(
                p_long_values >= 0.60
            )
        ),
        "fraction_ge_070": float(
            np.mean(
                p_long_values >= 0.70
            )
        ),
        "argmax_market_exposure": float(
            metrics.market_exposure
        ),
        "argmax_round_trips": int(
            metrics.round_trips
        ),
        "argmax_agent_return": float(
            metrics.agent_return
        ),
        "argmax_max_drawdown": float(
            metrics.agent_max_drawdown
        ),
        "argmax_balanced_score": float(
            metrics.balanced_score
        ),
        "argmax_profit_factor": (
            None
            if metrics.profit_factor is None
            else float(
                metrics.profit_factor
            )
        ),
    }

    print(
        "  mean P(LONG)   = "
        f"{summary['mean_p_long']:.4f}"
    )
    print(
        "  median P(LONG) = "
        f"{summary['median_p_long']:.4f}"
    )
    print(
        "  std P(LONG)    = "
        f"{summary['std_p_long']:.4f}"
    )
    print(
        "  P(LONG)>=0.30  = "
        f"{summary['fraction_ge_030']:.2%}"
    )
    print(
        "  P(LONG)>=0.50  = "
        f"{summary['fraction_ge_050']:.2%}"
    )
    print(
        "  P(LONG)>=0.70  = "
        f"{summary['fraction_ge_070']:.2%}"
    )
    print(
        "  argmax exposure = "
        f"{summary['argmax_market_exposure']:.2%}"
    )

    return frame, summary


def plot_overlay_histogram(
    probability_frame: pd.DataFrame,
    sources: list[SourceRun],
    *,
    bins: int,
    output_path: Path,
    dpi: int,
) -> None:
    figure, axis = plt.subplots(
        figsize=(11, 6.5)
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        bins + 1,
    )

    for source in sources:
        values = probability_frame.loc[
            probability_frame["run_id"]
            == source.run_id,
            "p_long",
        ]

        axis.hist(
            values,
            bins=bin_edges,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=(
                f"Run #{source.run_id} "
                f"/ seed {source.seed}"
            ),
        )

    axis.axvline(
        0.50,
        linestyle="--",
        linewidth=1.2,
        label="Argmax boundary (0.50)",
    )

    axis.set_xlim(
        0.0,
        1.0,
    )

    axis.set_xlabel(
        "P(LONG)"
    )
    axis.set_ylabel(
        "Probability density"
    )
    axis.set_title(
        "PPO policy P(LONG) distribution"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_confidence_curve(
    probability_frame: pd.DataFrame,
    sources: list[SourceRun],
    *,
    curve_points: int,
    output_path: Path,
    dpi: int,
) -> None:
    thresholds = np.linspace(
        0.0,
        1.0,
        curve_points,
    )

    figure, axis = plt.subplots(
        figsize=(11, 6.5)
    )

    for source in sources:
        values = probability_frame.loc[
            probability_frame["run_id"]
            == source.run_id,
            "p_long",
        ].to_numpy(dtype=float)

        fractions = np.asarray(
            [
                np.mean(
                    values >= threshold
                )
                for threshold in thresholds
            ],
            dtype=float,
        )

        axis.plot(
            thresholds,
            fractions,
            linewidth=2.0,
            label=(
                f"Run #{source.run_id} "
                f"/ seed {source.seed}"
            ),
        )

    axis.axvline(
        0.50,
        linestyle="--",
        linewidth=1.2,
        label="Argmax boundary (0.50)",
    )

    axis.set_xlim(
        0.0,
        1.0,
    )
    axis.set_ylim(
        0.0,
        1.0,
    )

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

    axis.set_xlabel(
        "Minimum P(LONG) required"
    )

    axis.set_ylabel(
        "Decisions with P(LONG) >= threshold"
    )

    axis.set_title(
        "PPO policy P(LONG) confidence curve"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)


def print_summary(
    summary_frame: pd.DataFrame,
) -> None:
    print()
    print("=" * 112)
    print("POLICY PROBABILITY SUMMARY")
    print("=" * 112)

    print(
        f"{'run':>4} "
        f"{'seed':>4} "
        f"{'mean':>8} "
        f"{'median':>8} "
        f"{'std':>8} "
        f"{'p10':>8} "
        f"{'p90':>8} "
        f"{'>=.30':>9} "
        f"{'>=.50':>9} "
        f"{'>=.70':>9} "
        f"{'exposure':>10}"
    )

    print("-" * 112)

    for row in summary_frame.itertuples(
        index=False
    ):
        print(
            f"{row.run_id:>4d} "
            f"{row.seed:>4d} "
            f"{row.mean_p_long:>8.3f} "
            f"{row.median_p_long:>8.3f} "
            f"{row.std_p_long:>8.3f} "
            f"{row.p10_p_long:>8.3f} "
            f"{row.p90_p_long:>8.3f} "
            f"{row.fraction_ge_030:>8.2%} "
            f"{row.fraction_ge_050:>8.2%} "
            f"{row.fraction_ge_070:>8.2%} "
            f"{row.argmax_market_exposure:>9.2%}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare P(LONG) probability distributions "
            "across existing PPO final checkpoints."
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
            "ppo_policy_probability_diag"
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

    if len(run_ids) < 2:
        raise SystemExit(
            "Cross-run policy probability diagnostic "
            "requires at least two runs."
        )

    engine = create_database_engine()

    session_factory = create_session_factory(
        engine
    )

    try:
        sources = load_source_runs(
            session_factory,
            run_ids,
        )

        print("SOURCE RUNS")
        print("-" * 88)

        for source in sources:
            print(
                f"run=#{source.run_id}"
                f" | seed={source.seed}"
                f" | final checkpoint="
                f"{source.checkpoint_id}"
                f" | {source.name}"
            )

        all_frames: list[
            pd.DataFrame
        ] = []

        summaries: list[
            dict[str, object]
        ] = []

        # Do every replay before writing anything.
        # replay_run_validation_checkpoint requires
        # a clean Git repository.
        for source in sources:
            frame, summary = (
                collect_probability_data(
                    source=source,
                    session_factory=(
                        session_factory
                    ),
                )
            )

            all_frames.append(frame)
            summaries.append(summary)

    finally:
        engine.dispose()

    probability_frame = pd.concat(
        all_frames,
        ignore_index=True,
    )

    summary_frame = pd.DataFrame(
        summaries
    ).sort_values(
        ["run_id"],
        ignore_index=True,
    )

    print_summary(
        summary_frame
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

    probabilities_path = (
        output_directory
        / "policy_probabilities.csv"
    )

    summary_path = (
        output_directory
        / "probability_summary.csv"
    )

    probability_frame.to_csv(
        probabilities_path,
        index=False,
    )

    summary_frame.to_csv(
        summary_path,
        index=False,
    )

    overlay_path = (
        output_directory
        / "p_long_histogram_overlay.png"
    )

    plot_overlay_histogram(
        probability_frame,
        sources,
        bins=args.bins,
        output_path=overlay_path,
        dpi=args.dpi,
    )

    confidence_curve_path = (
        output_directory
        / "p_long_confidence_curve.png"
    )

    plot_confidence_curve(
        probability_frame,
        sources,
        curve_points=args.curve_points,
        output_path=confidence_curve_path,
        dpi=args.dpi,
    )

    print()
    print("=" * 88)
    print("ARTIFACTS")
    print("=" * 88)
    print(probabilities_path)
    print(summary_path)
    print(overlay_path)
    print(confidence_curve_path)


if __name__ == "__main__":
    main()
