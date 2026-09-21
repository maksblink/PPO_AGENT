from __future__ import annotations

import enum
import math
from collections.abc import Iterable, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import PurePath
from typing import Any, TextIO
from zoneinfo import ZoneInfo

from train_and_eval.database.models import (
    Checkpoint,
    CheckpointSaveReason,
    Evaluation,
    EvaluationDataScope,
    EvaluationStatus,
    EvaluationTrigger,
    Run,
)
from train_and_eval.environment.contexts import (
    get_context_definition,
)


_LIST_MAX_WIDTHS = {
    "ID": 8,
    "NAME": 30,
    "STATUS": 10,
    "MODE": 7,
    "STEPS": 23,
    "EPOCHS": 12,
    "BEST SCORE": 14,
    "BEST STEP": 13,
    "STOP": 5,
    "FINISHED": 22,
}

_CHECKPOINT_MAX_WIDTHS = {
    "ID": 8,
    "RUN STEP": 14,
    "MODEL STEP": 14,
    "REASON": 12,
    "SIZE": 12,
    "EVAL": 11,
    "SCORE": 14,
    "FLAGS": 12,
    "PATH": 54,
}

_EVALUATION_MAX_WIDTHS = {
    "ID": 8,
    "CP": 8,
    "STEP": 14,
    "TRIGGER": 11,
    "STATUS": 11,
    "AGENT": 14,
    "LONG": 14,
    "SHORT": 14,
    "SCORE": 14,
    "DRAWDOWN": 14,
    "EXPOSURE": 12,
    "TRIPS": 10,
    "PROGRESS": 19,
    "FINISHED": 22,
}


def _enum_text(value: Any) -> str:
    if isinstance(value, enum.Enum):
        return str(value.value)
    return str(value)


def _is_enum_value(value: Any, expected: enum.Enum) -> bool:
    return _enum_text(value) == str(expected.value)


def _integer(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def _decimal(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, Decimal):
        return format(value, "f")
    return str(value)


def _float(value: Any, *, digits: int = 8) -> str:
    if value is None:
        return "n/a"
    try:
        result = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(result):
        return str(result)
    return f"{result:.{digits}f}"


def _bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _datetime(
    value: datetime | None,
    *,
    timezone: ZoneInfo,
) -> str:
    if value is None:
        return "n/a"
    if value.tzinfo is None or value.utcoffset() is None:
        return value.isoformat(sep=" ", timespec="seconds")
    local = value.astimezone(timezone)
    return local.strftime("%Y-%m-%d %H:%M:%S %Z")


def _datetime_short(
    value: datetime | None,
    *,
    timezone: ZoneInfo,
) -> str:
    if value is None:
        return "n/a"
    if value.tzinfo is None or value.utcoffset() is None:
        return value.isoformat(sep=" ", timespec="minutes")
    local = value.astimezone(timezone)
    return local.strftime("%Y-%m-%d %H:%M %Z")


def _duration(
    started_at: datetime | None,
    finished_at: datetime | None,
) -> str:
    if started_at is None:
        return "n/a"
    end = finished_at
    if end is None:
        if started_at.tzinfo is not None:
            end = datetime.now(started_at.tzinfo)
        else:
            end = datetime.now()
    seconds = max(0, int((end - started_at).total_seconds()))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def _size(value: Any) -> str:
    if value is None:
        return "n/a"
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            break
        size /= 1024.0
    if unit == "B":
        return f"{int(size):,} B"
    return f"{size:.1f} {unit}"


def _short_hash(value: Any) -> str:
    if not value:
        return "n/a"
    text = str(value)
    return text if len(text) <= 12 else text[:12]


def _truncate(value: str, maximum: int | None) -> str:
    if maximum is None or len(value) <= maximum:
        return value
    if maximum <= 1:
        return value[:maximum]
    return value[: maximum - 1] + "…"


def _render_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    stream: TextIO,
    max_widths: dict[str, int] | None = None,
    right_align: set[str] | None = None,
) -> None:
    materialized = [
        ["" if value is None else str(value) for value in row]
        for row in rows
    ]
    widths: list[int] = []
    for index, header in enumerate(headers):
        width = len(header)
        for row in materialized:
            width = max(width, len(row[index]))
        maximum = None if max_widths is None else max_widths.get(header)
        if maximum is not None:
            width = min(width, maximum)
        widths.append(width)

    def line(values: Sequence[str]) -> str:
        cells: list[str] = []
        for index, value in enumerate(values):
            header = headers[index]
            rendered = _truncate(value, widths[index])
            if right_align and header in right_align:
                cells.append(rendered.rjust(widths[index]))
            else:
                cells.append(rendered.ljust(widths[index]))
        return "  ".join(cells).rstrip()

    print(line(list(headers)), file=stream)
    print(
        line(["-" * width for width in widths]),
        file=stream,
    )
    for row in materialized:
        print(line(row), file=stream)


def _print_section(
    title: str,
    *,
    stream: TextIO,
    leading_blank: bool = True,
) -> None:
    if leading_blank:
        print(file=stream)
    print(f"===== {title} =====", file=stream)


def _print_fields(
    fields: Sequence[tuple[str, Any]],
    *,
    stream: TextIO,
) -> None:
    width = max((len(label) for label, _ in fields), default=0)
    for label, value in fields:
        print(f"{label + ':':<{width + 2}} {value}", file=stream)


def _config_value(
    config: Any,
    *keys: str,
    default: Any = "n/a",
) -> Any:
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _all_evaluations(run: Run) -> list[Evaluation]:
    result: list[Evaluation] = []
    for checkpoint in getattr(run, "checkpoints", ()):
        result.extend(getattr(checkpoint, "evaluations", ()))
    return result


def _checkpoint_by_id(run: Run) -> dict[int, Checkpoint]:
    return {
        int(checkpoint.id): checkpoint
        for checkpoint in getattr(run, "checkpoints", ())
    }


def _completed_run_validation_evaluations(
    run: Run,
) -> list[Evaluation]:
    result: list[Evaluation] = []
    for evaluation in _all_evaluations(run):
        if not _is_enum_value(
            evaluation.status,
            EvaluationStatus.COMPLETED,
        ):
            continue
        if not _is_enum_value(
            evaluation.data_scope,
            EvaluationDataScope.RUN_VALIDATION,
        ):
            continue
        if evaluation.balanced_score is None:
            continue
        try:
            score = float(evaluation.balanced_score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            result.append(evaluation)
    return result


def best_evaluation(run: Run) -> Evaluation | None:
    evaluations = _completed_run_validation_evaluations(run)
    if not evaluations:
        return None
    return max(
        evaluations,
        key=lambda evaluation: (
            float(evaluation.balanced_score),
            evaluation.finished_at or evaluation.created_at,
            int(evaluation.id),
        ),
    )


def final_checkpoint(run: Run) -> Checkpoint | None:
    checkpoints = [
        checkpoint
        for checkpoint in getattr(run, "checkpoints", ())
        if _is_enum_value(
            checkpoint.save_reason,
            CheckpointSaveReason.FINAL,
        )
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda checkpoint: int(checkpoint.id))


def final_evaluation(run: Run) -> Evaluation | None:
    checkpoint = final_checkpoint(run)
    if checkpoint is None:
        return None
    evaluations = list(getattr(checkpoint, "evaluations", ()))
    if not evaluations:
        return None
    return max(
        evaluations,
        key=lambda evaluation: (
            _is_enum_value(
                evaluation.trigger,
                EvaluationTrigger.FINAL,
            ),
            _is_enum_value(
                evaluation.status,
                EvaluationStatus.COMPLETED,
            ),
            int(evaluation.id),
        ),
    )


def _checkpoint_for_evaluation(
    run: Run,
    evaluation: Evaluation | None,
) -> Checkpoint | None:
    if evaluation is None:
        return None
    return _checkpoint_by_id(run).get(int(evaluation.checkpoint_id))


def _latest_checkpoint_evaluation(
    checkpoint: Checkpoint,
) -> Evaluation | None:
    evaluations = list(getattr(checkpoint, "evaluations", ()))
    if not evaluations:
        return None
    return max(evaluations, key=lambda evaluation: int(evaluation.id))


def _checkpoint_flags(
    run: Run,
    checkpoint: Checkpoint,
) -> str:
    flags: list[str] = []
    best = _checkpoint_for_evaluation(run, best_evaluation(run))
    final = final_checkpoint(run)
    if best is not None and int(checkpoint.id) == int(best.id):
        flags.append("best")
    if final is not None and int(checkpoint.id) == int(final.id):
        flags.append("final")
    return ",".join(flags) or "-"


def render_run_list(
    runs: Sequence[Run],
    *,
    stream: TextIO,
    timezone: ZoneInfo,
) -> None:
    if not runs:
        print("No runs found.", file=stream)
        return

    rows: list[tuple[str, ...]] = []
    for run in runs:
        evaluation = best_evaluation(run)
        checkpoint = _checkpoint_for_evaluation(run, evaluation)
        rows.append(
            (
                str(run.id),
                str(run.name),
                _enum_text(run.status),
                _enum_text(run.continuation_mode),
                (
                    f"{int(run.training_steps_completed):,}/"
                    f"{int(run.training_steps_requested):,}"
                ),
                _decimal(run.data_epochs_completed),
                _float(
                    None if evaluation is None else evaluation.balanced_score
                ),
                _integer(None if checkpoint is None else checkpoint.run_step),
                "yes" if run.stopped_early else "no",
                _datetime_short(run.finished_at, timezone=timezone),
            )
        )

    _render_table(
        (
            "ID",
            "NAME",
            "STATUS",
            "MODE",
            "STEPS",
            "EPOCHS",
            "BEST SCORE",
            "BEST STEP",
            "STOP",
            "FINISHED",
        ),
        rows,
        stream=stream,
        max_widths=_LIST_MAX_WIDTHS,
        right_align={
            "ID",
            "STEPS",
            "EPOCHS",
            "BEST SCORE",
            "BEST STEP",
        },
    )


def _evaluation_fields(
    evaluation: Evaluation | None,
) -> list[tuple[str, str]]:
    if evaluation is None:
        return [
            ("Evaluation", "n/a"),
        ]
    return [
        ("Evaluation ID", str(evaluation.id)),
        ("Status", _enum_text(evaluation.status)),
        ("Trigger", _enum_text(evaluation.trigger)),
        ("Policy mode", _enum_text(evaluation.policy_mode)),
        ("Balanced score", _float(evaluation.balanced_score)),
        ("Agent return", _float(evaluation.agent_return)),
        ("Always-long return", _float(evaluation.always_long_return)),
        ("Always-short return", _float(evaluation.always_short_return)),
        (
            "Agent vs always long",
            _float(evaluation.agent_vs_always_long_return),
        ),
        ("Agent max drawdown", _float(evaluation.agent_max_drawdown)),
        ("Drawdown improvement", _float(evaluation.drawdown_improvement)),
        ("Market exposure", _float(evaluation.market_exposure)),
        ("Net exposure", _float(evaluation.net_exposure)),
        ("Profit factor", _float(evaluation.profit_factor)),
        ("Win rate", _float(evaluation.win_rate)),
        ("Round trips", _integer(evaluation.round_trips)),
        (
            "Evaluation progress",
            f"{_integer(evaluation.steps_completed)}/"
            f"{_integer(evaluation.steps_expected)}",
        ),
    ]


def render_run_show(
    run: Run,
    *,
    stream: TextIO,
    timezone: ZoneInfo,
) -> None:
    config = run.normalized_config_json or {}
    checkpoints = sorted(
        getattr(run, "checkpoints", ()),
        key=lambda checkpoint: (
            int(checkpoint.run_step),
            int(checkpoint.id),
        ),
    )
    evaluations = sorted(
        _all_evaluations(run),
        key=lambda evaluation: int(evaluation.id),
    )
    best = best_evaluation(run)
    best_checkpoint = _checkpoint_for_evaluation(run, best)
    final_cp = final_checkpoint(run)
    final_eval = final_evaluation(run)

    context_name = _config_value(
        config,
        "environment",
        "context",
    )
    window = _config_value(config, "environment", "window")
    history_rows: Any = "n/a"
    try:
        history_rows = get_context_definition(
            str(context_name)
        ).required_history_rows(int(window))
    except (TypeError, ValueError, KeyError):
        pass

    _print_section("RUN", stream=stream, leading_blank=False)
    _print_fields(
        (
            ("Run ID", str(run.id)),
            ("Name", run.name),
            ("Description", run.description),
            ("Status", _enum_text(run.status)),
            ("Continuation", _enum_text(run.continuation_mode)),
            (
                "Source checkpoint ID",
                _integer(run.source_checkpoint_id),
            ),
            ("Created at", _datetime(run.created_at, timezone=timezone)),
            ("Started at", _datetime(run.started_at, timezone=timezone)),
            ("Finished at", _datetime(run.finished_at, timezone=timezone)),
            ("Duration", _duration(run.started_at, run.finished_at)),
        ),
        stream=stream,
    )

    _print_section("REPRODUCIBILITY", stream=stream)
    _print_fields(
        (
            ("Git commit", run.git_commit),
            ("Git branch", run.git_branch),
            ("Config schema", str(run.config_schema_version)),
            ("Config SHA-256", run.config_sha256),
            ("Normalized config SHA-256", run.normalized_config_sha256),
            ("Seed", str(run.seed)),
        ),
        stream=stream,
    )

    _print_section("DATA", stream=stream)
    _print_fields(
        (
            ("Data path", run.data_path),
            ("Data SHA-256", run.data_sha256),
            ("Dataset rows", _integer((getattr(run, "window_metadata", None) or {}).get("data_rows", run.train_rows + run.validation_rows))),
            ("TRAIN rows", _integer(run.train_rows)),
            ("Validation rows", _integer(run.validation_rows)),
            ("Legacy split index", "n/a (explicit time ranges)" if getattr(run, "window_metadata", None) else _integer(run.split_index)),
            (
                "Train ratio",
                _float(_config_value(config, "data", "train_ratio"), digits=6),
            ),
            ("Context", str(context_name)),
            ("Context history", f"{_integer(history_rows)} rows"),
            ("Window", str(window)),
            (
                "Position side",
                str(_config_value(config, "environment", "position_side")),
            ),
        ),
        stream=stream,
    )

    _print_section("TRAINING", stream=stream)
    _print_fields(
        (
            ("Duration unit", _enum_text(run.duration_unit)),
            ("Duration amount", _integer(run.duration_amount)),
            ("Steps per data epoch", _integer(run.steps_per_data_epoch)),
            (
                "Requested steps",
                _integer(run.training_steps_requested),
            ),
            (
                "Completed steps",
                _integer(run.training_steps_completed),
            ),
            (
                "Data epochs completed",
                _decimal(run.data_epochs_completed),
            ),
            ("Stopped early", _bool(run.stopped_early)),
            (
                "Early-stop reason",
                run.early_stop_reason or "n/a",
            ),
        ),
        stream=stream,
    )

    _print_section("ENVIRONMENT", stream=stream)
    _print_fields(
        (
            ("Stake PLN", str(_config_value(config, "environment", "stake_pln"))),
            ("Fee bps", str(_config_value(config, "environment", "fee_bps"))),
            ("Swap long bps", str(config.get("environment", {}).get(
                "swap_long_bps", config.get("environment", {}).get("swap_bps", "n/a")))),
            ("Swap short bps", str(config.get("environment", {}).get(
                "swap_short_bps", config.get("environment", {}).get("swap_bps", "n/a")))),
            ("Reward scale", str(_config_value(config, "environment", "reward_scale"))),
            ("Exposure penalty", str(_config_value(config, "environment", "exposure_penalty"))),
            ("Turnover penalty", str(_config_value(config, "environment", "turnover_penalty"))),
            ("Drawdown penalty", str(_config_value(config, "environment", "drawdown_penalty"))),
            ("Profit reward mult", str(_config_value(config, "environment", "profit_reward_mult"))),
            ("Loss reward mult", str(_config_value(config, "environment", "loss_reward_mult"))),
        ),
        stream=stream,
    )

    _print_section("PPO", stream=stream)
    _print_fields(
        tuple(
            (label, str(_config_value(config, "ppo", key)))
            for label, key in (
                ("Policy", "policy"),
                ("Device", "device"),
                ("Architecture", "hidden_sizes"),
                ("Activation", "activation"),
                ("n_steps", "n_steps"),
                ("batch_size", "batch_size"),
                ("n_epochs", "n_epochs"),
                ("Learning rate", "learning_rate"),
                ("Gamma", "gamma"),
                ("GAE lambda", "gae_lambda"),
                ("Clip range", "clip_range"),
                ("Entropy coefficient", "ent_coef"),
                ("Value coefficient", "vf_coef"),
                ("Max grad norm", "max_grad_norm"),
                ("Target KL", "target_kl"),
            )
        ),
        stream=stream,
    )

    _print_section("EVALUATION SETTINGS", stream=stream)
    _print_fields(
        tuple(
            (label, str(_config_value(config, "evaluation", key)))
            for label, key in (
                ("Evaluate every steps", "eval_every_steps"),
                ("Checkpoint every steps", "checkpoint_every_steps"),
                ("Policy mode", "policy_mode"),
                ("Threshold action", "threshold_action"),
                ("Probability threshold", "probability_threshold"),
                ("Best metric", "best_metric"),
                ("Early-stop patience", "early_stop_patience_evals"),
            )
        ),
        stream=stream,
    )

    _print_section("BEST RESULT", stream=stream)
    best_fields: list[tuple[str, str]] = [
        (
            "Checkpoint ID",
            "n/a" if best_checkpoint is None else str(best_checkpoint.id),
        ),
        (
            "Checkpoint run step",
            "n/a" if best_checkpoint is None else _integer(best_checkpoint.run_step),
        ),
        (
            "Checkpoint model step",
            "n/a" if best_checkpoint is None else _integer(best_checkpoint.model_step),
        ),
    ]
    best_fields.extend(_evaluation_fields(best))
    _print_fields(best_fields, stream=stream)

    _print_section("FINAL RESULT", stream=stream)
    final_fields: list[tuple[str, str]] = [
        (
            "Checkpoint ID",
            "n/a" if final_cp is None else str(final_cp.id),
        ),
        (
            "Checkpoint run step",
            "n/a" if final_cp is None else _integer(final_cp.run_step),
        ),
        (
            "Checkpoint model step",
            "n/a" if final_cp is None else _integer(final_cp.model_step),
        ),
        (
            "Checkpoint path",
            "n/a" if final_cp is None else final_cp.relative_path,
        ),
    ]
    final_fields.extend(_evaluation_fields(final_eval))
    _print_fields(final_fields, stream=stream)

    if run.error_type or run.error_message:
        _print_section("FAILURE", stream=stream)
        _print_fields(
            (
                ("Error type", run.error_type or "n/a"),
                ("Error message", run.error_message or "n/a"),
            ),
            stream=stream,
        )

    _print_section("CHECKPOINTS", stream=stream)
    if checkpoints:
        render_checkpoints(run, stream=stream)
    else:
        print("No checkpoints persisted.", file=stream)

    _print_section("EVALUATIONS", stream=stream)
    if evaluations:
        render_evaluations(run, stream=stream, timezone=timezone)
    else:
        print("No evaluations persisted.", file=stream)


def render_checkpoints(
    run: Run,
    *,
    stream: TextIO,
) -> None:
    checkpoints = sorted(
        getattr(run, "checkpoints", ()),
        key=lambda checkpoint: (
            int(checkpoint.run_step),
            int(checkpoint.id),
        ),
    )
    if not checkpoints:
        print("No checkpoints found.", file=stream)
        return

    rows: list[tuple[str, ...]] = []
    for checkpoint in checkpoints:
        evaluation = _latest_checkpoint_evaluation(checkpoint)
        rows.append(
            (
                str(checkpoint.id),
                _integer(checkpoint.run_step),
                _integer(checkpoint.model_step),
                _enum_text(checkpoint.save_reason),
                _size(checkpoint.size_bytes),
                "-" if evaluation is None else _enum_text(evaluation.status),
                _float(None if evaluation is None else evaluation.balanced_score),
                _checkpoint_flags(run, checkpoint),
                checkpoint.relative_path,
            )
        )

    _render_table(
        (
            "ID",
            "RUN STEP",
            "MODEL STEP",
            "REASON",
            "SIZE",
            "EVAL",
            "SCORE",
            "FLAGS",
            "PATH",
        ),
        rows,
        stream=stream,
        max_widths=_CHECKPOINT_MAX_WIDTHS,
        right_align={
            "ID",
            "RUN STEP",
            "MODEL STEP",
            "SIZE",
            "SCORE",
        },
    )


def render_evaluations(
    run: Run,
    *,
    stream: TextIO,
    timezone: ZoneInfo,
) -> None:
    evaluations = sorted(
        _all_evaluations(run),
        key=lambda evaluation: int(evaluation.id),
    )
    if not evaluations:
        print("No evaluations found.", file=stream)
        return

    checkpoints = _checkpoint_by_id(run)
    rows: list[tuple[str, ...]] = []
    for evaluation in evaluations:
        checkpoint = checkpoints.get(int(evaluation.checkpoint_id))
        rows.append(
            (
                str(evaluation.id),
                str(evaluation.checkpoint_id),
                _integer(None if checkpoint is None else checkpoint.run_step),
                _enum_text(evaluation.trigger),
                _enum_text(evaluation.status),
                _float(evaluation.agent_return),
                _float(evaluation.always_long_return),
                _float(evaluation.always_short_return),
                _float(evaluation.balanced_score),
                _float(evaluation.agent_max_drawdown),
                _float(evaluation.market_exposure),
                _integer(evaluation.round_trips),
                (
                    f"{_integer(evaluation.steps_completed)}/"
                    f"{_integer(evaluation.steps_expected)}"
                ),
                _datetime(evaluation.finished_at, timezone=timezone),
            )
        )

    _render_table(
        (
            "ID",
            "CP",
            "STEP",
            "TRIGGER",
            "STATUS",
            "AGENT",
            "LONG",
            "SHORT",
            "SCORE",
            "DRAWDOWN",
            "EXPOSURE",
            "TRIPS",
            "PROGRESS",
            "FINISHED",
        ),
        rows,
        stream=stream,
        max_widths=_EVALUATION_MAX_WIDTHS,
        right_align={
            "ID",
            "CP",
            "STEP",
            "AGENT",
            "LONG",
            "SHORT",
            "SCORE",
            "DRAWDOWN",
            "EXPOSURE",
            "TRIPS",
            "PROGRESS",
        },
    )


def _full_value(value: Any, *, timezone: ZoneInfo) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, datetime):
        return _datetime(value, timezone=timezone)
    if isinstance(value, enum.Enum):
        return str(value.value)
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, PurePath):
        return str(value)
    return str(value)


def render_evaluations_full(
    run: Run,
    *,
    stream: TextIO,
    timezone: ZoneInfo,
) -> None:
    evaluations = sorted(
        _all_evaluations(run),
        key=lambda evaluation: int(evaluation.id),
    )
    if not evaluations:
        print("No evaluations found.", file=stream)
        return

    checkpoint_map = _checkpoint_by_id(run)
    for index, evaluation in enumerate(evaluations):
        if index:
            print(file=stream)
        checkpoint = checkpoint_map.get(int(evaluation.checkpoint_id))
        print(
            f"===== EVALUATION {evaluation.id} =====",
            file=stream,
        )
        extra_fields = [
            (
                "checkpoint_run_step",
                "n/a" if checkpoint is None else _integer(checkpoint.run_step),
            ),
            (
                "checkpoint_model_step",
                "n/a" if checkpoint is None else _integer(checkpoint.model_step),
            ),
            (
                "checkpoint_save_reason",
                "n/a" if checkpoint is None else _enum_text(checkpoint.save_reason),
            ),
        ]
        fields = list(extra_fields)
        for column in Evaluation.__table__.columns:
            fields.append(
                (
                    column.name,
                    _full_value(
                        getattr(evaluation, column.name),
                        timezone=timezone,
                    ),
                )
            )
        _print_fields(fields, stream=stream)
