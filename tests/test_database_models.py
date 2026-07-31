from __future__ import annotations

from datetime import datetime, timezone

import pytest

from train_and_eval.database.models import (
    DEFAULT_RUN_DESCRIPTION,
    Base,
    Run,
)


def test_runs_table_contains_required_columns() -> None:
    table = Base.metadata.tables["runs"]

    expected_columns = {
        "id",
        "name",
        "status",
        "continuation_mode",
        "source_run_id",
        "source_checkpoint",
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
        "ck_runs_source_run_not_self",
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


def test_source_run_uses_self_referencing_foreign_key() -> None:
    table = Base.metadata.tables["runs"]
    foreign_keys = list(
        table.c.source_run_id.foreign_keys
    )

    assert len(foreign_keys) == 1
    assert foreign_keys[0].target_fullname == "runs.id"
    assert foreign_keys[0].ondelete == "RESTRICT"


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
