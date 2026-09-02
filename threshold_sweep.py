from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from train_and_eval.database.models import Checkpoint
from train_and_eval.database.session import (
    create_database_engine,
    create_session_factory,
)
from train_and_eval.evaluation.service import (
    replay_run_validation_checkpoint,
)
from train_and_eval.runs.queries import load_run


DEFAULT_RUNS = (34, 36, 37)

DEFAULT_THRESHOLDS = (
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
)


@dataclass(frozen=True)
class SourceRun:
    run_id: int
    name: str
    seed: int
    checkpoint_id: int
    checkpoint_run_step: int
    checkpoint_model_step: int
    checkpoint_save_reason: str


@dataclass(frozen=True)
class SweepResult:
    run_id: int
    run_name: str
    seed: int
    checkpoint_id: int
    checkpoint_run_step: int
    checkpoint_model_step: int
    checkpoint_save_reason: str
    threshold: float

    balanced_score: float
    agent_return: float
    always_long_return: float

    max_drawdown: float
    always_long_max_drawdown: float

    market_exposure: float
    flat_exposure: float

    round_trips: int
    profit_factor: float | None
    win_rate: float | None


def enum_value(value):
    return getattr(value, "value", value)


def parse_int_list(value: str) -> list[int]:
    result = []

    for raw in value.split(","):
        raw = raw.strip()

        if not raw:
            continue

        parsed = int(raw)

        if parsed < 1:
            raise argparse.ArgumentTypeError(
                "IDs must be positive integers."
            )

        result.append(parsed)

    if not result:
        raise argparse.ArgumentTypeError(
            "At least one ID is required."
        )

    return result


def parse_float_list(value: str) -> list[float]:
    result = []

    for raw in value.split(","):
        raw = raw.strip()

        if not raw:
            continue

        parsed = float(raw)

        if not 0.0 <= parsed <= 1.0:
            raise argparse.ArgumentTypeError(
                "Thresholds must be between 0 and 1."
            )

        result.append(parsed)

    if not result:
        raise argparse.ArgumentTypeError(
            "At least one threshold is required."
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
                    f"Run #{run_id} expected exactly one "
                    f"final checkpoint, found "
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
                    checkpoint_run_step=int(
                        checkpoint.run_step
                    ),
                    checkpoint_model_step=int(
                        checkpoint.model_step
                    ),
                    checkpoint_save_reason=str(
                        enum_value(
                            checkpoint.save_reason
                        )
                    ),
                )
            )

    return sources


def load_source_checkpoints(
    session_factory,
    checkpoint_ids: list[int],
) -> list[SourceRun]:
    sources: list[SourceRun] = []

    with session_factory() as session:
        for checkpoint_id in checkpoint_ids:
            statement = (
                select(Checkpoint)
                .options(
                    selectinload(Checkpoint.run)
                )
                .where(
                    Checkpoint.id == checkpoint_id
                )
            )
            checkpoint = session.scalars(
                statement
            ).one_or_none()

            if checkpoint is None:
                raise RuntimeError(
                    f"Checkpoint #{checkpoint_id} "
                    "does not exist."
                )

            run = checkpoint.run

            sources.append(
                SourceRun(
                    run_id=int(run.id),
                    name=str(run.name),
                    seed=int(run.seed),
                    checkpoint_id=int(
                        checkpoint.id
                    ),
                    checkpoint_run_step=int(
                        checkpoint.run_step
                    ),
                    checkpoint_model_step=int(
                        checkpoint.model_step
                    ),
                    checkpoint_save_reason=str(
                        enum_value(
                            checkpoint.save_reason
                        )
                    ),
                )
            )

    return sources


def run_sweep(
    *,
    sources: list[SourceRun],
    thresholds: list[float],
    session_factory,
) -> list[SweepResult]:
    results: list[SweepResult] = []

    total = len(sources) * len(thresholds)
    current = 0

    for source in sources:
        print()
        print("=" * 88)
        print(
            f"RUN #{source.run_id} "
            f"| seed={source.seed} "
            f"| checkpoint={source.checkpoint_id} "
            f"| run_step={source.checkpoint_run_step:,} "
            f"| reason={source.checkpoint_save_reason}"
        )
        print(source.name)
        print("=" * 88)

        for threshold in thresholds:
            current += 1

            print()
            print(
                f"[{current:02d}/{total:02d}] "
                f"run=#{source.run_id} "
                f"checkpoint=#{source.checkpoint_id} "
                f"seed={source.seed} "
                f"threshold={threshold:.4f}"
            )

            replay = replay_run_validation_checkpoint(
                checkpoint_id=source.checkpoint_id,
                session_factory=session_factory,
                policy_mode="probability_threshold",
                threshold_action=1,
                probability_threshold=threshold,
            )

            metrics = replay.metrics

            row = SweepResult(
                run_id=source.run_id,
                run_name=source.name,
                seed=source.seed,
                checkpoint_id=source.checkpoint_id,
                checkpoint_run_step=(
                    source.checkpoint_run_step
                ),
                checkpoint_model_step=(
                    source.checkpoint_model_step
                ),
                checkpoint_save_reason=(
                    source.checkpoint_save_reason
                ),
                threshold=threshold,
                balanced_score=float(
                    metrics.balanced_score
                ),
                agent_return=float(
                    metrics.agent_return
                ),
                always_long_return=float(
                    metrics.always_long_return
                ),
                max_drawdown=float(
                    metrics.agent_max_drawdown
                ),
                always_long_max_drawdown=float(
                    metrics.always_long_max_drawdown
                ),
                market_exposure=float(
                    metrics.market_exposure
                ),
                flat_exposure=float(
                    metrics.flat_exposure
                ),
                round_trips=int(
                    metrics.round_trips
                ),
                profit_factor=(
                    None
                    if metrics.profit_factor is None
                    else float(
                        metrics.profit_factor
                    )
                ),
                win_rate=(
                    None
                    if metrics.win_rate is None
                    else float(metrics.win_rate)
                ),
            )

            results.append(row)

            pf_text = (
                "-"
                if row.profit_factor is None
                else f"{row.profit_factor:.3f}"
            )

            print(
                f"  exposure={row.market_exposure:+.2%}"
                f" | trips={row.round_trips:5d}"
                f" | return={row.agent_return:+.2%}"
                f" | maxDD={row.max_drawdown:+.2%}"
                f" | PF={pf_text}"
                f" | score={row.balanced_score:+.5f}"
            )

    return results


def print_summary(
    results: list[SweepResult],
) -> None:
    width = 154

    print()
    print("=" * width)
    print("THRESHOLD SWEEP SUMMARY")
    print("=" * width)

    print(
        f"{'run':>4} "
        f"{'cp':>5} "
        f"{'run step':>10} "
        f"{'reason':>11} "
        f"{'seed':>4} "
        f"{'threshold':>9} "
        f"{'exposure':>10} "
        f"{'flat':>9} "
        f"{'trips':>7} "
        f"{'return':>10} "
        f"{'maxDD':>10} "
        f"{'PF':>8} "
        f"{'score':>10}"
    )

    print("-" * width)

    for row in results:
        pf_text = (
            "-"
            if row.profit_factor is None
            else f"{row.profit_factor:.3f}"
        )

        print(
            f"{row.run_id:>4d} "
            f"{row.checkpoint_id:>5d} "
            f"{row.checkpoint_run_step:>10,d} "
            f"{row.checkpoint_save_reason:>11} "
            f"{row.seed:>4d} "
            f"{row.threshold:>9.4f} "
            f"{row.market_exposure:>9.2%} "
            f"{row.flat_exposure:>8.2%} "
            f"{row.round_trips:>7d} "
            f"{row.agent_return:>+9.2%} "
            f"{row.max_drawdown:>+9.2%} "
            f"{pf_text:>8} "
            f"{row.balanced_score:>+10.5f}"
        )


def write_csv(
    results: list[SweepResult],
    output: Path,
) -> None:
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        SweepResult.__dataclass_fields__
    )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in results:
            writer.writerow(
                {
                    field: getattr(row, field)
                    for field in fieldnames
                }
            )

    print()
    print(f"Saved: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay existing PPO checkpoints "
            "over a LONG probability-threshold sweep."
        )
    )

    source_group = (
        parser.add_mutually_exclusive_group()
    )

    source_group.add_argument(
        "--runs",
        default=None,
        help=(
            "Comma-separated run IDs; the final "
            "checkpoint of each run is replayed. "
            "Default when no source option is given: "
            "34,36,37."
        ),
    )

    source_group.add_argument(
        "--checkpoint-ids",
        default=None,
        help=(
            "Comma-separated checkpoint IDs to replay "
            "directly. Cannot be combined with --runs."
        ),
    )

    parser.add_argument(
        "--thresholds",
        default=",".join(
            str(value)
            for value in DEFAULT_THRESHOLDS
        ),
        help=(
            "Comma-separated LONG probability "
            "thresholds."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/tmp/ppo_threshold_sweep.csv"
        ),
        help="CSV output path.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    thresholds = parse_float_list(
        args.thresholds
    )

    engine = create_database_engine()
    session_factory = create_session_factory(
        engine
    )

    try:
        if args.checkpoint_ids is not None:
            checkpoint_ids = parse_int_list(
                args.checkpoint_ids
            )
            sources = load_source_checkpoints(
                session_factory,
                checkpoint_ids,
            )
        else:
            runs_value = (
                args.runs
                if args.runs is not None
                else ",".join(
                    str(value)
                    for value in DEFAULT_RUNS
                )
            )
            run_ids = parse_int_list(
                runs_value
            )
            sources = load_source_runs(
                session_factory,
                run_ids,
            )

        print("SOURCE CHECKPOINTS")
        print("-" * 120)

        for source in sources:
            print(
                f"run=#{source.run_id}"
                f" | seed={source.seed}"
                f" | checkpoint=#{source.checkpoint_id}"
                f" | run_step={source.checkpoint_run_step:,}"
                f" | model_step={source.checkpoint_model_step:,}"
                f" | reason={source.checkpoint_save_reason}"
                f" | {source.name}"
            )

        results = run_sweep(
            sources=sources,
            thresholds=thresholds,
            session_factory=session_factory,
        )

        print_summary(results)

        # Write only after all replays finish.
        # This keeps the repository clean while
        # replay_run_validation_checkpoint performs
        # its clean-git verification.
        write_csv(
            results,
            args.output,
        )

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
