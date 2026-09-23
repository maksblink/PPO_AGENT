"""Checkpoint-paired evaluation, scope isolation and RNG invariance."""
from dataclasses import replace
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from train_and_eval.database.models import EvaluationDataScope, EvaluationStatus
from train_and_eval.evaluation import service
from train_and_eval.dashboard.data import _latest_final_evaluations
from train_and_eval.reporting.artifacts import evaluation_source_paths
from tests.test_evaluation_service import _patch_setup, _session_factory, _state


@pytest.mark.parametrize("temporal", [False, True])
def test_train_scope_uses_effective_scored_bounds(monkeypatch, tmp_path, temporal):
    _patch_setup(monkeypatch, tmp_path, [])
    factory = _session_factory()
    factory.session.run.normalized_config_json["ppo"]["batch_size"] = 2
    if temporal:
        factory.session.run.window_metadata = {
            "data_rows": 8, "train": {"start_index": 3, "end_index": 4},
            "validation": {"start_index": 4, "end_index": 8},
        }
    captured = {}
    def pending(*args, **kwargs):
        captured.update(kwargs)
        return _state(EvaluationStatus.PENDING)
    monkeypatch.setattr(service, "create_pending_evaluation", pending)
    monkeypatch.setattr(service, "mark_evaluation_running", lambda *a, **k: None)
    monkeypatch.setattr(service, "load_persisted_ppo_checkpoint", lambda *a, **k: object())
    def replay(model, data, env, **kwargs):
        assert kwargs["evaluation_start_index"] == (3 if temporal else 2)
        assert kwargs["evaluation_end_index"] == 4
        return object()
    monkeypatch.setattr(service, "run_ppo_evaluation", replay)
    monkeypatch.setattr(service, "complete_evaluation", lambda *a, **k: _state(EvaluationStatus.COMPLETED))
    service.evaluate_run_validation_checkpoint(factory, checkpoint_id=9, trigger="final",
        policy_mode="deterministic_argmax", data_scope="run_training", project_root=tmp_path)
    assert captured["data_scope"] == EvaluationDataScope.RUN_TRAINING
    assert captured["lookback_rows"] == 2
    assert captured["evaluation_end_index"] == 4


@pytest.mark.parametrize("fails", [False, True])
def test_replay_preserves_all_cpu_random_streams(fails):
    random.seed(21); np.random.seed(21); torch.manual_seed(21)
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.seed(21); np.random.seed(21); torch.manual_seed(21)
    @service.preserve_random_state
    def evaluation():
        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        random.random(); np.random.random(); torch.rand(7)
        if fails:
            raise RuntimeError("failed replay")
    if fails:
        with pytest.raises(RuntimeError):
            evaluation()
    else:
        evaluation()
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


def test_dashboard_final_selection_ignores_train_even_when_newer_and_better():
    frame = pd.DataFrame([
        {"id": 1, "run_id": 1, "data_scope": "run_validation", "status": "completed", "trigger": "final", "balanced_score": .1},
        {"id": 2, "run_id": 1, "data_scope": "run_training", "status": "completed", "trigger": "final", "balanced_score": 100.},
    ])
    result = _latest_final_evaluations(frame)
    assert result.id.tolist() == [1]


def test_train_val_artifact_names_cannot_collide(tmp_path):
    train = evaluation_source_paths(tmp_path, EvaluationDataScope.RUN_TRAINING)
    val = evaluation_source_paths(tmp_path, EvaluationDataScope.RUN_VALIDATION)
    assert not set(train) & set(val)
    assert [p.name for p in train] == ['trajectory_train.parquet', 'trade_events_train.parquet', 'metrics_train.json']
    assert [p.name for p in val] == ['trajectory_val.parquet', 'trade_events_val.parquet', 'metrics_val.json']
