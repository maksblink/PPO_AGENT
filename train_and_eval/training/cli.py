from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from typing import Any, TextIO

from train_and_eval.database.session import (
    create_database_engine,
    create_session_factory,
)
from train_and_eval.reproducibility import (
    require_clean_git,
)
from train_and_eval.training.service import (
    TrainingServiceResult,
    train_ppo_run,
)


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a positive integer"
        ) from error

    if result < 1:
        raise argparse.ArgumentTypeError(
            "must be a positive integer"
        )

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m train_and_eval.training",
        description=(
            "Run one complete configuration-driven PPO training job. "
            "The YAML controls data, environment, PPO, checkpointing, "
            "evaluation, resume, and early stopping."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the complete YAML run configuration.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help=(
            "Stable-Baselines3 verbosity: 0=silent, 1=info, "
            "2=debug. Default: 1."
        ),
    )
    parser.add_argument(
        "--log-interval",
        type=_positive_integer,
        default=1,
        help=(
            "Log every N PPO rollout iterations. Default: 1."
        ),
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Enable the Stable-Baselines3 progress bar.",
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Print a Python traceback when training fails.",
    )
    return parser


def _score_text(value: float | None) -> str:
    if value is None:
        return "n/a"

    return f"{float(value):.8f}"


def _print_success(
    result: TrainingServiceResult,
    *,
    stream: TextIO,
) -> None:
    run = result.run
    training = result.training

    print("Training run: OK", file=stream)
    print(f"Run ID:                 {run.run_id}", file=stream)
    print(f"Run name:               {run.name}", file=stream)
    print(f"Status:                 {run.status.value}", file=stream)
    print(
        "Continuation source:    "
        + (
            "fresh"
            if result.source_checkpoint_id is None
            else f"checkpoint {result.source_checkpoint_id}"
        ),
        file=stream,
    )
    print(
        "Requested steps:        "
        f"{run.training_steps_requested:,}",
        file=stream,
    )
    print(
        "Completed steps:        "
        f"{run.training_steps_completed:,}",
        file=stream,
    )
    print(
        "Data epochs completed:  "
        f"{run.data_epochs_completed}",
        file=stream,
    )
    print(
        "Model steps:            "
        f"{training.model_steps_before:,} -> "
        f"{training.model_steps_after:,}",
        file=stream,
    )
    print(
        "Rollouts collected:     "
        f"{len(training.rollout_sizes):,}",
        file=stream,
    )
    print(
        "Stopped early:          "
        f"{'yes' if run.stopped_early else 'no'}",
        file=stream,
    )

    if run.early_stop_reason:
        print(
            "Early-stop reason:     "
            f"{run.early_stop_reason}",
            file=stream,
        )

    print(
        "Checkpoints persisted:  "
        f"{len(result.checkpoints):,}",
        file=stream,
    )
    print(
        "Evaluations completed:  "
        f"{len(result.evaluations):,}",
        file=stream,
    )
    print(
        "Best balanced score:    "
        f"{_score_text(result.best_evaluation.balanced_score)}",
        file=stream,
    )
    print(
        "Best checkpoint ID:     "
        f"{result.best_checkpoint.checkpoint_id}",
        file=stream,
    )
    print(
        "Best checkpoint step:   "
        f"{result.best_checkpoint.run_step:,}",
        file=stream,
    )
    print(
        "Final checkpoint ID:    "
        f"{result.checkpoint.checkpoint_id}",
        file=stream,
    )
    print(
        "Final checkpoint step:  "
        f"{result.checkpoint.run_step:,}",
        file=stream,
    )
    print(
        "Final checkpoint path:  "
        f"{result.checkpoint.relative_path}",
        file=stream,
    )


def _print_failure(
    error: BaseException,
    *,
    stream: TextIO,
) -> None:
    print("Training run: FAILED", file=stream)
    print(
        f"Error type: {type(error).__name__}",
        file=stream,
    )

    message = str(error).strip()

    if not message:
        message = repr(error)

    print(f"Error: {message}", file=stream)

    for note in getattr(error, "__notes__", ()):
        print(f"Note: {note}", file=stream)


def execute_training_cli(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Execute one parsed CLI request and return a process exit code."""
    engine: Any | None = None

    try:
        # Reject dirty repositories before reading database configuration.
        # The service repeats this check immediately before its own work.
        require_clean_git()
        engine = create_database_engine()
        session_factory = create_session_factory(
            engine
        )

        result = train_ppo_run(
            session_factory,
            config_path=args.config,
            verbose=args.verbose,
            log_interval=args.log_interval,
            progress_bar=args.progress_bar,
        )

    except KeyboardInterrupt as error:
        print(
            "Training run: INTERRUPTED",
            file=stderr,
        )
        print(
            "The run lifecycle service attempted to persist "
            "the interrupted state and checkpoint.",
            file=stderr,
        )

        for note in getattr(error, "__notes__", ()):
            print(f"Note: {note}", file=stderr)

        if args.traceback:
            traceback.print_exc(file=stderr)

        return 130

    except Exception as error:
        _print_failure(
            error,
            stream=stderr,
        )

        if args.traceback:
            traceback.print_exc(file=stderr)

        return 1

    finally:
        if engine is not None:
            engine.dispose()

    _print_success(
        result,
        stream=stdout,
    )
    return 0


def main(
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return execute_training_cli(args)
