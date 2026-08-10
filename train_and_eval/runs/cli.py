from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from typing import Any, TextIO
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from train_and_eval.database.models import (
    ContinuationMode,
    RunStatus,
)
from train_and_eval.database.session import (
    create_database_engine,
    create_session_factory,
)
from train_and_eval.runs.formatting import (
    render_checkpoints,
    render_evaluations,
    render_evaluations_full,
    render_run_list,
    render_run_show,
)
from train_and_eval.runs.queries import (
    load_run,
    load_runs,
)
from train_and_eval.reporting.service import (
    generate_run_report,
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


def _timezone(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as error:
        raise argparse.ArgumentTypeError(
            "must be a valid IANA time zone"
        ) from error


def _add_run_id(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-id",
        required=True,
        type=_positive_integer,
        help="Database ID of the run.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m train_and_eval.runs",
        description=(
            "Inspect persisted PPO runs and rebuild derived report artifacts. "
            "Report generation never inserts or modifies database evaluations."
        ),
    )
    parser.add_argument(
        "--timezone",
        type=_timezone,
        default=ZoneInfo("Europe/Paris"),
        help=(
            "IANA time zone used for displayed timestamps. "
            "Default: Europe/Paris."
        ),
    )
    parser.add_argument(
        "--traceback",
        action="store_true",
        help="Print a Python traceback when a query fails.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    list_parser = subparsers.add_parser(
        "list",
        help="List recent runs.",
        description="List recent runs with their best validation result.",
    )
    list_parser.add_argument(
        "--limit",
        type=_positive_integer,
        default=20,
        help="Maximum number of runs. Default: 20.",
    )
    list_parser.add_argument(
        "--status",
        choices=tuple(status.value for status in RunStatus),
        help="Only show runs with this status.",
    )
    list_parser.add_argument(
        "--continuation",
        choices=tuple(mode.value for mode in ContinuationMode),
        help="Only show fresh or resumed runs.",
    )
    list_parser.add_argument(
        "--name",
        help="Only show run names containing this text.",
    )
    list_parser.add_argument(
        "--stopped-early",
        action="store_true",
        help="Only show runs stopped by early stopping.",
    )

    show_parser = subparsers.add_parser(
        "show",
        help="Show one complete run.",
        description=(
            "Show identity, reproducibility, data, configuration, "
            "best/final results, checkpoints, and evaluations."
        ),
    )
    _add_run_id(show_parser)

    checkpoints_parser = subparsers.add_parser(
        "checkpoints",
        help="List checkpoints for one run.",
    )
    _add_run_id(checkpoints_parser)

    evaluations_parser = subparsers.add_parser(
        "evaluations",
        help="List evaluations for one run.",
    )
    _add_run_id(evaluations_parser)
    evaluations_parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Print every persisted evaluation column in a vertical view."
        ),
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Regenerate plot and trajectory artifacts for one run.",
        description=(
            "Build run-level plots from persisted metrics and replay missing "
            "validation trajectories from immutable checkpoints."
        ),
    )
    _add_run_id(report_parser)
    report_parser.add_argument(
        "--evaluation-id",
        type=_positive_integer,
        help="Only rebuild one completed evaluation from this run.",
    )

    return parser


def _dispatch(
    args: argparse.Namespace,
    *,
    session: Any,
    stdout: TextIO,
) -> None:
    if args.command == "list":
        runs = load_runs(
            session,
            status=(
                None
                if args.status is None
                else RunStatus(args.status)
            ),
            continuation_mode=(
                None
                if args.continuation is None
                else ContinuationMode(args.continuation)
            ),
            name_contains=args.name,
            stopped_early=args.stopped_early,
            limit=args.limit,
        )
        render_run_list(
            runs,
            stream=stdout,
            timezone=args.timezone,
        )
        return

    run = load_run(
        session,
        run_id=args.run_id,
    )

    if args.command == "show":
        render_run_show(
            run,
            stream=stdout,
            timezone=args.timezone,
        )
        return

    if args.command == "checkpoints":
        render_checkpoints(
            run,
            stream=stdout,
        )
        return

    if args.command == "evaluations":
        if args.full:
            render_evaluations_full(
                run,
                stream=stdout,
                timezone=args.timezone,
            )
        else:
            render_evaluations(
                run,
                stream=stdout,
                timezone=args.timezone,
            )
        return

    raise RuntimeError(
        f"Unsupported runs command: {args.command!r}"
    )


def execute_runs_cli(
    args: argparse.Namespace,
    *,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    engine: Any | None = None
    try:
        engine = create_database_engine()
        session_factory = create_session_factory(engine)
        if args.command == "report":
            paths = generate_run_report(
                session_factory,
                run_id=args.run_id,
                evaluation_id=args.evaluation_id,
            )
            print(f"Run #{args.run_id} report: OK", file=stdout)
            for path in paths:
                print(f"  {path}", file=stdout)
        else:
            with session_factory() as session:
                _dispatch(
                    args,
                    session=session,
                    stdout=stdout,
                )
    except Exception as error:
        print("Run registry query: FAILED", file=stderr)
        print(f"Error type: {type(error).__name__}", file=stderr)
        message = str(error).strip() or repr(error)
        print(f"Error: {message}", file=stderr)
        if args.traceback:
            traceback.print_exc(file=stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return execute_runs_cli(args)
