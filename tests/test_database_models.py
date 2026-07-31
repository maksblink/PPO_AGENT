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
        "uq_checkpoints_run_id_run_step",
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

