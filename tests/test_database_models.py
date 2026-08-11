from __future__ import annotations

from datetime import datetime, timezone

import pytest

from train_and_eval.database.models import (
    DEFAULT_RUN_DESCRIPTION,
    Base,
    CheckpointSaveReason,
    Run,
)


def test_runs_table_contains_required_columns() -> None:
    table = Base.metadata.tables["runs"]

    expected_columns = {
        "id",
        "name",
        "status",
        "continuation_mode",
        "source_checkpoint_id",
        "description",
        "created_at",
        "modified_at",
        "started_at",
        "finished_at",
        "git_commit",
        "git_branch",
        "config_schema_version",
        "seed",
        "config_sha256",
        "normalized_config_sha256",
        "raw_config_yaml",
        "normalized_config_json",
        "data_path",
        "data_sha256",
        "duration_unit",
        "duration_amount",
        "split_index",
        "train_rows",
        "validation_rows",
        "steps_per_data_epoch",
        "training_steps_requested",
        "training_steps_completed",
        "data_epochs_completed",
        "stopped_early",
        "early_stop_reason",
        "error_type",
        "error_message",
    }

    assert set(table.columns.keys()) == expected_columns


def test_runs_table_has_named_safety_constraints() -> None:
    table = Base.metadata.tables["runs"]

    constraint_names = {
        constraint.name
        for constraint in table.constraints
    }

    expected_names = {
        "pk_runs",
        "uq_runs_name",
        "ck_runs_continuation_fields",
        "ck_runs_duration_amount_positive",
        "ck_runs_train_rows_positive",
        "ck_runs_validation_rows_positive",
        "ck_runs_split_matches_train_rows",
        "ck_runs_steps_per_epoch_positive",
        "ck_runs_requested_steps_positive",
        "ck_runs_completed_steps_range",
        "ck_runs_completed_epochs_nonnegative",
        "ck_runs_early_stop_fields",
        "ck_runs_execution_time_order",
        "ck_runs_status_fields",
    }

    assert expected_names.issubset(
        constraint_names
    )




def test_description_has_database_default() -> None:
    column = Base.metadata.tables[
        "runs"
    ].c.description

    assert column.nullable is False
    assert DEFAULT_RUN_DESCRIPTION in str(
        column.server_default.arg
    )


def test_modified_at_has_no_automatic_onupdate() -> None:
    column = Base.metadata.tables[
        "runs"
    ].c.modified_at

    assert column.onupdate is None
    assert column.server_onupdate is None


def test_update_description_changes_notes_and_timestamp() -> None:
    original_time = datetime(
        2026,
        7,
        31,
        8,
        0,
        tzinfo=timezone.utc,
    )
    new_time = datetime(
        2026,
        7,
        31,
        9,
        0,
        tzinfo=timezone.utc,
    )

    run = Run(
        description=DEFAULT_RUN_DESCRIPTION,
        modified_at=original_time,
    )

    run.update_description(
        "Promising result on validation.",
        modified_at=new_time,
    )

    assert (
        run.description
        == "Promising result on validation."
    )
    assert run.modified_at == new_time


def test_update_description_rejects_naive_timestamp() -> None:
    run = Run(
        description=DEFAULT_RUN_DESCRIPTION,
    )

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        run.update_description(
            "Notes",
            modified_at=datetime(
                2026,
                7,
                31,
                10,
                0,
            ),
        )


def test_normalized_config_sha256_is_required() -> None:
    column = Base.metadata.tables[
        "runs"
    ].c.normalized_config_sha256

    assert column.nullable is False
    assert column.type.length == 64


def test_run_source_checkpoint_uses_foreign_key() -> None:
    table = Base.metadata.tables["runs"]
    foreign_keys = list(
        table.c.source_checkpoint_id.foreign_keys
    )

    assert len(foreign_keys) == 1
    assert (
        foreign_keys[0].target_fullname
        == "checkpoints.id"
    )
    assert foreign_keys[0].ondelete == "RESTRICT"
    assert foreign_keys[0].use_alter is True
    assert (
        foreign_keys[0].name
        == "fk_runs_source_checkpoint_id_checkpoints"
    )


def test_checkpoints_table_contains_required_columns() -> None:
    table = Base.metadata.tables["checkpoints"]

    assert set(table.columns.keys()) == {
        "id",
        "run_id",
        "run_step",
        "model_step",
        "save_reason",
        "relative_path",
        "sha256",
        "size_bytes",
        "created_at",
    }


def test_checkpoints_table_has_required_constraints() -> None:
    table = Base.metadata.tables["checkpoints"]

    constraint_names = {
        constraint.name
        for constraint in table.constraints
    }

    assert {
        "pk_checkpoints",
        (
            "uq_checkpoints_run_id_run_step_"
            "save_reason"
        ),
        "uq_checkpoints_relative_path",
        "ck_checkpoints_run_step_nonnegative",
        "ck_checkpoints_model_step_nonnegative",
        (
            "ck_checkpoints_"
            "model_step_not_less_than_run_step"
        ),
        "ck_checkpoints_size_bytes_positive",
        "ck_checkpoints_sha256_lowercase_hex",
    }.issubset(constraint_names)


def test_checkpoint_run_uses_restrict_foreign_key() -> None:
    table = Base.metadata.tables["checkpoints"]
    foreign_keys = list(
        table.c.run_id.foreign_keys
    )

    assert len(foreign_keys) == 1
    assert foreign_keys[0].target_fullname == "runs.id"
    assert foreign_keys[0].ondelete == "RESTRICT"


def test_checkpoint_indexes_are_defined() -> None:
    table = Base.metadata.tables["checkpoints"]

    indexes = {
        index.name: index
        for index in table.indexes
    }

    assert "ix_checkpoints_run_id" in indexes

    final_index = indexes[
        "ux_checkpoints_one_final_per_run"
    ]

    assert final_index.unique is True
    assert str(
        final_index.dialect_options[
            "postgresql"
        ]["where"]
    ) == "save_reason = 'final'"


def test_checkpoint_save_reasons_are_explicit() -> None:
    assert {
        reason.value
        for reason in CheckpointSaveReason
    } == {
        "initial",
        "periodic",
        "final",
        "manual",
        "interrupted",
    }



def test_evaluations_table_contains_required_columns() -> None:
    table = Base.metadata.tables["evaluations"]

    expected_columns = {
        "id",
        "checkpoint_id",
        "status",
        "trigger",
        "policy_mode",
        "threshold_action",
        "probability_threshold",
        "seed",
        "data_scope",
        "data_path",
        "data_sha256",
        "data_rows",
        "evaluation_start_index",
        "evaluation_end_index",
        "evaluation_start_at",
        "evaluation_end_at",
        "lookback_rows",
        "steps_expected",
        "steps_completed",
        "created_at",
        "started_at",
        "finished_at",
        "git_commit",
        "git_branch",
        "error_type",
        "error_message",
        "agent_return",
        "always_long_return",
        "always_short_return",
        "agent_vs_always_long_return",
        "balanced_score",
        "agent_max_drawdown",
        "always_long_max_drawdown",
        "always_short_max_drawdown",
        "drawdown_improvement",
        "net_exposure",
        "long_exposure",
        "short_exposure",
        "flat_exposure",
        "market_exposure",
        "trade_events_total",
        "trade_event_rate",
        "round_trips",
        "round_trip_rate",
        "winning_trades",
        "losing_trades",
        "breakeven_trades",
        "long_round_trips",
        "short_round_trips",
        "open_long_count",
        "open_short_count",
        "close_long_count",
        "close_short_count",
        "swap_events",
        "win_rate",
        "loss_rate",
        "breakeven_rate",
        "long_win_rate",
        "short_win_rate",
        "avg_trade_return",
        "avg_win_return",
        "avg_loss_return",
        "median_trade_return",
        "long_avg_trade_return",
        "short_avg_trade_return",
        "long_median_trade_return",
        "short_median_trade_return",
        "largest_win_return",
        "largest_loss_return",
        "gross_profit_return",
        "gross_loss_return",
        "net_profit_return",
        "profit_factor",
        "payoff_ratio",
        "min_bars_held",
        "avg_bars_held",
        "median_bars_held",
        "max_bars_held",
        "max_consecutive_wins",
        "max_consecutive_losses",
        "avg_win_streak",
        "avg_loss_streak",
        "current_streak_type",
        "current_streak",
        "total_fee_return",
        "total_swap_return",
        "total_cost_return",
        "avg_fee_per_trade_return",
        "avg_swap_per_trade_return",
        "avg_cost_per_trade_return",
        "cumulative_shaped_reward",
        "open_position_return_at_end",
    }

    assert set(table.columns.keys()) == expected_columns


def test_evaluations_table_has_required_constraints() -> None:
    table = Base.metadata.tables["evaluations"]

    constraint_names = {
        constraint.name
        for constraint in table.constraints
    }

    expected_names = {
        "pk_evaluations",
        "ck_evaluations_policy_fields",
        "ck_evaluations_completed_core_metrics",
        "ck_evaluations_metric_ranges",
        "ck_evaluations_metric_counts_nonnegative",
        "ck_evaluations_trade_count_consistency",
        "ck_evaluations_round_trip_metric_nullability",
        "ck_evaluations_long_metric_nullability",
        "ck_evaluations_short_metric_nullability",
        "ck_evaluations_win_metric_nullability",
        "ck_evaluations_loss_metric_nullability",
        "ck_evaluations_profit_factor_nullability",
        "ck_evaluations_payoff_ratio_nullability",
        "ck_evaluations_holding_period_order",
        "ck_evaluations_streak_consistency",
        "ck_evaluations_data_rows_positive",
        "ck_evaluations_start_index_nonnegative",
        "ck_evaluations_range_nonempty",
        "ck_evaluations_end_index_within_data",
        "ck_evaluations_lookback_rows_positive",
        "ck_evaluations_steps_expected_positive",
        "ck_evaluations_steps_match_range",
        "ck_evaluations_steps_completed_range",
        "ck_evaluations_data_time_order",
        "ck_evaluations_execution_time_order",
        "ck_evaluations_data_sha256_lowercase_hex",
        "ck_evaluations_status_fields",
    }

    assert expected_names.issubset(
        constraint_names
    )


def test_evaluation_checkpoint_uses_restrict_foreign_key() -> None:
    table = Base.metadata.tables["evaluations"]
    foreign_keys = list(
        table.c.checkpoint_id.foreign_keys
    )

    assert len(foreign_keys) == 1
    assert (
        foreign_keys[0].target_fullname
        == "checkpoints.id"
    )
    assert foreign_keys[0].ondelete == "RESTRICT"


def test_evaluation_checkpoint_id_is_indexed() -> None:
    table = Base.metadata.tables["evaluations"]

    index_names = {
        index.name
        for index in table.indexes
    }

    assert "ix_evaluations_checkpoint_id" in index_names


def test_evaluations_do_not_duplicate_run_id() -> None:
    table = Base.metadata.tables["evaluations"]

    assert "run_id" not in table.columns


def test_evaluation_enum_values_are_explicit() -> None:
    from train_and_eval.database.models import (
        EvaluationCurrentStreakType,
        EvaluationDataScope,
        EvaluationPolicyMode,
        EvaluationStatus,
        EvaluationTrigger,
    )

    assert {
        value.value
        for value in EvaluationStatus
    } == {
        "pending",
        "running",
        "completed",
        "failed",
        "cancelled",
    }

    assert {
        value.value
        for value in EvaluationTrigger
    } == {
        "scheduled",
        "final",
        "manual",
    }

    assert {
        value.value
        for value in EvaluationDataScope
    } == {
        "run_validation",
        "extended_out_of_sample",
        "custom_range",
    }

    assert {
        value.value
        for value in EvaluationPolicyMode
    } == {
        "deterministic_argmax",
        "stochastic_sample",
        "probability_threshold",
    }

    assert {
        value.value
        for value in EvaluationCurrentStreakType
    } == {
        "none",
        "win",
        "loss",
        "breakeven",
    }


def test_policy_mode_replaces_deterministic_column() -> None:
    table = Base.metadata.tables["evaluations"]

    assert "deterministic" not in table.columns
    assert table.c.policy_mode.nullable is False
    assert table.c.threshold_action.nullable is True
    assert table.c.probability_threshold.nullable is True


def test_evaluation_metric_columns_are_nullable() -> None:
    table = Base.metadata.tables["evaluations"]

    metric_names = {
        "agent_return",
        "balanced_score",
        "round_trips",
        "win_rate",
        "profit_factor",
        "min_bars_held",
        "avg_bars_held",
        "current_streak_type",
        "cumulative_shaped_reward",
    }

    assert all(
        table.c[name].nullable
        for name in metric_names
    )


def test_evaluation_count_metrics_use_bigint() -> None:
    from sqlalchemy import BigInteger, Integer

    table = Base.metadata.tables["evaluations"]

    bigint_names = {
        "trade_events_total",
        "round_trips",
        "winning_trades",
        "min_bars_held",
        "max_bars_held",
        "current_streak",
    }

    assert all(
        isinstance(table.c[name].type, BigInteger)
        for name in bigint_names
    )

    assert isinstance(
        table.c.threshold_action.type,
        Integer,
    )


def test_evaluation_return_metrics_use_float() -> None:
    from sqlalchemy import Float

    table = Base.metadata.tables["evaluations"]

    float_names = {
        "agent_return",
        "agent_max_drawdown",
        "market_exposure",
        "trade_event_rate",
        "avg_trade_return",
        "profit_factor",
        "avg_bars_held",
        "cumulative_shaped_reward",
    }

    assert all(
        isinstance(table.c[name].type, Float)
        for name in float_names
    )



def test_evaluation_policy_fields_allows_dynamic_threshold_action() -> None:
    from sqlalchemy import CheckConstraint

    from train_and_eval.database.models import Evaluation

    policy_constraint = next(
        constraint
        for constraint in Evaluation.__table__.constraints
        if (
            isinstance(constraint, CheckConstraint)
            and "policy_fields" in str(constraint.name)
        )
    )

    sql = " ".join(
        str(policy_constraint.sqltext)
        .lower()
        .split()
    )

    assert (
        "threshold_action is null "
        "or threshold_action = 1"
    ) in sql

    assert "threshold_action >= 0" not in sql


def test_checkpoint_step_uniqueness_includes_save_reason() -> None:
    from sqlalchemy import UniqueConstraint

    table = Base.metadata.tables["checkpoints"]
    constraint = next(
        item
        for item in table.constraints
        if (
            isinstance(item, UniqueConstraint)
            and item.name
            == (
                "uq_checkpoints_run_id_run_step_"
                "save_reason"
            )
        )
    )

    assert [
        column.name
        for column in constraint.columns
    ] == [
        "run_id",
        "run_step",
        "save_reason",
    ]


def test_training_metrics_table_contains_required_columns() -> None:
    table = Base.metadata.tables["training_metrics"]
    assert set(table.columns.keys()) == {
        "id",
        "run_id",
        "run_step",
        "model_step",
        "rollout_number",
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
        "created_at",
    }


def test_training_metrics_use_cascade_run_foreign_key() -> None:
    table = Base.metadata.tables["training_metrics"]
    foreign_keys = list(table.c.run_id.foreign_keys)
    assert len(foreign_keys) == 1
    assert foreign_keys[0].target_fullname == "runs.id"
    assert foreign_keys[0].ondelete == "CASCADE"
