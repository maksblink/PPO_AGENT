from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pandas as pd
from sqlalchemy import text

from train_and_eval.database.session import create_database_engine


@dataclass(frozen=True)
class DashboardData:
    runs: pd.DataFrame
    checkpoints: pd.DataFrame
    evaluations: pd.DataFrame
    training_metrics: pd.DataFrame
    explorer: pd.DataFrame


def _read_table(
    engine: Any,
    table_name: str,
) -> pd.DataFrame:
    return pd.read_sql_query(
        text(f'SELECT * FROM "{table_name}"'),
        engine,
    )


def _convert_decimal_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    for column in frame.columns:
        sample = frame[column].dropna()

        if sample.empty:
            continue

        if isinstance(sample.iloc[0], Decimal):
            frame[column] = frame[column].map(
                lambda value: (
                    float(value)
                    if isinstance(value, Decimal)
                    else value
                )
            )

    return frame


def _mapping_columns(
    frame: pd.DataFrame,
) -> list[str]:
    result: list[str] = []

    for column in frame.columns:
        values = frame[column].dropna()

        if values.empty:
            continue

        if any(isinstance(value, dict) for value in values):
            result.append(column)

    return result


def _flatten_mapping_columns(
    frame: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for column in _mapping_columns(frame):
        records = [
            value if isinstance(value, dict) else {}
            for value in frame[column]
        ]

        normalized = pd.json_normalize(
            records,
            sep=".",
        )

        normalized.index = frame.index
        normalized.columns = [
            f"{prefix}{column}.{name}"
            for name in normalized.columns
        ]

        parts.append(normalized)

    if not parts:
        return pd.DataFrame(index=frame.index)

    return pd.concat(parts, axis=1)


def _prefix_columns(
    frame: pd.DataFrame,
    prefix: str,
    *,
    keep: set[str] | None = None,
) -> pd.DataFrame:
    keep = keep or set()

    return frame.rename(
        columns={
            column: (
                column
                if column in keep
                else f"{prefix}{column}"
            )
            for column in frame.columns
        }
    )


def _attach_checkpoint_run_id(
    evaluations: pd.DataFrame,
    checkpoints: pd.DataFrame,
) -> pd.DataFrame:
    if evaluations.empty:
        return evaluations.copy()

    required_evaluation = {"checkpoint_id"}
    required_checkpoint = {"id", "run_id"}

    if not required_evaluation.issubset(
        evaluations.columns
    ):
        return evaluations.copy()

    if not required_checkpoint.issubset(
        checkpoints.columns
    ):
        return evaluations.copy()

    checkpoint_lookup = checkpoints[
        ["id", "run_id"]
    ].rename(
        columns={
            "id": "_checkpoint_lookup_id",
            "run_id": "run_id",
        }
    )

    return evaluations.merge(
        checkpoint_lookup,
        left_on="checkpoint_id",
        right_on="_checkpoint_lookup_id",
        how="left",
    ).drop(
        columns=["_checkpoint_lookup_id"],
        errors="ignore",
    )


def _latest_final_evaluations(
    evaluations: pd.DataFrame,
) -> pd.DataFrame:
    if evaluations.empty:
        return evaluations.copy()

    frame = evaluations.copy()

    if "status" in frame.columns:
        completed = (
            frame["status"]
            .astype(str)
            .str.lower()
            == "completed"
        )

        if completed.any():
            frame = frame.loc[completed]

    if "trigger" in frame.columns:
        final = (
            frame["trigger"]
            .astype(str)
            .str.lower()
            == "final"
        )

        if final.any():
            frame = frame.loc[final]

    if "run_id" not in frame.columns:
        return frame

    sort_columns = [
        column
        for column in ("created_at", "id")
        if column in frame.columns
    ]

    if sort_columns:
        frame = frame.sort_values(sort_columns)

    return (
        frame
        .dropna(subset=["run_id"])
        .groupby("run_id", as_index=False)
        .tail(1)
    )


def _latest_training_metrics(
    training_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if training_metrics.empty:
        return training_metrics.copy()

    if "run_id" not in training_metrics.columns:
        return training_metrics.copy()

    frame = training_metrics.copy()

    sort_columns = [
        column
        for column in (
            "run_step",
            "model_step",
            "created_at",
            "id",
        )
        if column in frame.columns
    ]

    if sort_columns:
        frame = frame.sort_values(sort_columns)

    return (
        frame
        .dropna(subset=["run_id"])
        .groupby("run_id", as_index=False)
        .tail(1)
    )


def _build_explorer(
    runs: pd.DataFrame,
    checkpoints: pd.DataFrame,
    evaluations: pd.DataFrame,
    training_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()

    run_mapping_columns = _mapping_columns(runs)

    config_frame = _flatten_mapping_columns(
        runs,
        prefix="cfg.",
    )

    scalar_runs = runs.drop(
        columns=run_mapping_columns,
        errors="ignore",
    )

    scalar_runs = _prefix_columns(
        scalar_runs,
        "run.",
    )

    explorer = scalar_runs.copy()

    if "run.id" in explorer.columns:
        explorer.insert(
            0,
            "run_id",
            explorer["run.id"],
        )

    if not config_frame.empty:
        explorer = pd.concat(
            [
                explorer.reset_index(drop=True),
                config_frame.reset_index(drop=True),
            ],
            axis=1,
        )

    attached_evaluations = _attach_checkpoint_run_id(
        evaluations,
        checkpoints,
    )

    final_evaluations = _latest_final_evaluations(
        attached_evaluations
    )

    if (
        not final_evaluations.empty
        and "run_id" in final_evaluations.columns
    ):
        eval_frame = _prefix_columns(
            final_evaluations,
            "eval.",
            keep={"run_id"},
        )

        explorer = explorer.merge(
            eval_frame,
            on="run_id",
            how="left",
        )

    latest_training = _latest_training_metrics(
        training_metrics
    )

    if (
        not latest_training.empty
        and "run_id" in latest_training.columns
    ):
        training_frame = _prefix_columns(
            latest_training,
            "train.",
            keep={"run_id"},
        )

        explorer = explorer.merge(
            training_frame,
            on="run_id",
            how="left",
        )

    return _convert_decimal_columns(explorer)


def load_dashboard_data() -> DashboardData:
    engine = create_database_engine()

    try:
        runs = _convert_decimal_columns(
            _read_table(engine, "runs")
        )
        checkpoints = _convert_decimal_columns(
            _read_table(engine, "checkpoints")
        )
        evaluations = _convert_decimal_columns(
            _read_table(engine, "evaluations")
        )
        training_metrics = _convert_decimal_columns(
            _read_table(engine, "training_metrics")
        )
    finally:
        engine.dispose()

    evaluations_with_run = _attach_checkpoint_run_id(
        evaluations,
        checkpoints,
    )

    explorer = _build_explorer(
        runs,
        checkpoints,
        evaluations,
        training_metrics,
    )

    return DashboardData(
        runs=runs,
        checkpoints=checkpoints,
        evaluations=evaluations_with_run,
        training_metrics=training_metrics,
        explorer=explorer,
    )
