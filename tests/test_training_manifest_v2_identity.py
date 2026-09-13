from types import SimpleNamespace

import pytest

from train_and_eval.training import persistence


def check_entry(monkeypatch, entry):
    monkeypatch.setattr(persistence, "require_run_name_available", lambda *a, **k: None)
    loaded = SimpleNamespace(
        config=SimpleNamespace(
            run=SimpleNamespace(name="manifest-v2-regression"),
            data=SimpleNamespace(path="data/expected.parquet"),
        ),
        data_manifest_entry={
            "path": "data/different.parquet",
            "sha256": "a" * 64,
            "rows": 1,
            **entry,
        },
    )
    # Stop at the next identity guard; no database or files are modified.
    persistence.create_pending_run(
        None,
        loaded_config=loaded,
        split=SimpleNamespace(),
        git_state=SimpleNamespace(),
        training_steps_requested=1,
    )


def test_passed_v2_file_reaches_path_identity_check(monkeypatch):
    with pytest.raises(persistence.RunIdentityError, match="Manifest data path"):
        check_entry(monkeypatch, {"validation_status": "passed"})


@pytest.mark.parametrize("entry", [
    {},
    {"status": "okay"},
    {"validation_status": "failed"},
    {"status": "okay", "validation_status": "failed"},
])
def test_unapproved_file_is_rejected_before_path_check(monkeypatch, entry):
    with pytest.raises(persistence.RunIdentityError, match="validation_status='passed'"):
        check_entry(monkeypatch, entry)
