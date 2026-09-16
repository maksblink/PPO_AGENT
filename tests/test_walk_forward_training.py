from dataclasses import replace
import hashlib
from pathlib import Path

import pytest
import torch

from tests.test_ppo_checkpoints import _trained_model, _ppo_config, TinyEnvironment
from tests.test_training_service import _loaded_config, _patch_preflight, FakeModel, _run_state, _checkpoint, _training_result
from train_and_eval.database.models import RunStatus
from train_and_eval.ppo.adapter import save_ppo_model_file, load_ppo_model_file, ppo_resume_custom_objects
from train_and_eval.run_config import normalize_config
import train_and_eval.training.service as service
from train_and_eval.training.execution import learn_ppo_exact_timesteps


def test_checkpoint_copies_preserve_optimizer_without_sharing_mutations(tmp_path):
    source = _trained_model()
    path = save_ppo_model_file(source, tmp_path/"source.zip")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cfg = _ppo_config().model_copy(update={"learning_rate": .000075})
    a = load_ppo_model_file(path, environment=TinyEnvironment(), device="cpu", custom_objects=ppo_resume_custom_objects(cfg, seed=123))
    b = load_ppo_model_file(path, environment=TinyEnvironment(), device="cpu", custom_objects=ppo_resume_custom_objects(cfg, seed=123))
    original = source.policy.optimizer.state_dict()["state"]
    for model in (a,b):
        restored = model.policy.optimizer.state_dict()["state"]
        for parameter, entries in original.items():
            for name, value in entries.items():
                torch.testing.assert_close(restored[parameter][name], value)
    unchanged_weights = {k: v.clone() for k,v in b.policy.state_dict().items()}
    unchanged_moments = {k: v["exp_avg"].clone() for k,v in b.policy.optimizer.state_dict()["state"].items()}
    learn_ppo_exact_timesteps(a, total_timesteps=8)
    assert a.policy.optimizer.param_groups[0]["lr"] == .000075
    for key, value in b.policy.state_dict().items():
        torch.testing.assert_close(value, unchanged_weights[key])
    for key, entries in b.policy.optimizer.state_dict()["state"].items():
        torch.testing.assert_close(entries["exp_avg"], unchanged_moments[key])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_training_without_validation_completes_without_best(monkeypatch, tmp_path):
    loaded = _loaded_config()
    cfg = loaded.config.model_copy(update={"evaluation": loaded.config.evaluation.model_copy(update={"training_mode":"none"})})
    loaded = replace(loaded, config=cfg, normalized_json=normalize_config(cfg))
    events = []
    _patch_preflight(monkeypatch, tmp_path, events, loaded=loaded)
    model = FakeModel(num_timesteps=0)
    monkeypatch.setattr(service, "create_pending_run", lambda *a,**k: _run_state(RunStatus.PENDING))
    monkeypatch.setattr(service, "mark_run_running", lambda *a,**k: _run_state(RunStatus.RUNNING))
    monkeypatch.setattr(service, "TradingEnvironment", lambda *a,**k: object())
    monkeypatch.setattr(service, "create_ppo_model", lambda *a,**k: model)
    def train(*args, **kwargs):
        model.num_timesteps = kwargs["total_timesteps"]
        return _training_result(before=0, completed=model.num_timesteps)
    monkeypatch.setattr(service, "learn_ppo_exact_timesteps", train)
    monkeypatch.setattr(service, "update_run_progress", lambda *a,**k: None)
    monkeypatch.setattr(service, "persist_ppo_checkpoint", lambda *a,**k: _checkpoint(save_reason=k['save_reason'],run_step=k['run_step'],model_step=model.num_timesteps))
    monkeypatch.setattr(service, "complete_run", lambda *a,**k: _run_state(RunStatus.COMPLETED, completed=10))
    def forbidden(*args, **kwargs):
        pytest.fail("Training-only stage attempted to evaluate or select a checkpoint")
    monkeypatch.setattr(service, "evaluate_run_validation_checkpoint", forbidden)
    result = service.train_ppo_run(object(), config_path="run.yml", project_root=tmp_path)
    assert result.best_checkpoint is None
    assert result.best_evaluation is None
    assert result.evaluations == ()
    assert result.checkpoint.run_step == 10


def test_final_short_rollout_still_uses_full_minibatches():
    model = _trained_model()
    # Tiny equivalent of 1024+512, with rollout 8 and batch 4.
    counts = []
    handle = model.policy.optimizer.register_step_post_hook(lambda *args: counts.append(1))
    result = learn_ppo_exact_timesteps(model, total_timesteps=12)
    handle.remove()
    assert result.rollout_sizes == (8,4)
    assert len(counts) == 3 * model.n_epochs


def test_resume_applies_mutable_ppo_settings_before_learning(tmp_path):
    source = _trained_model()
    path = save_ppo_model_file(source, tmp_path/"mutable.zip")
    cfg = _ppo_config().model_copy(update={
        "n_steps": 12, "batch_size": 4, "n_epochs": 2, "learning_rate": .0001,
        "gamma": .9, "gae_lambda": .8, "clip_range": .3, "clip_range_vf": .2,
        "normalize_advantage": False, "ent_coef": .01, "vf_coef": .7,
        "max_grad_norm": .2, "target_kl": .3,
    })
    model = load_ppo_model_file(path, environment=TinyEnvironment(), device="cpu",
                               custom_objects=ppo_resume_custom_objects(cfg, seed=123))
    assert model.rollout_buffer.buffer_size == 12
    assert model.rollout_buffer.gamma == .9 and model.rollout_buffer.gae_lambda == .8
    assert model.clip_range(1.) == .3 and model.clip_range_vf(1.) == .2
    assert model.normalize_advantage is False
    assert model.ent_coef == .01 and model.vf_coef == .7
    assert model.max_grad_norm == .2 and model.target_kl == .3
    result = learn_ppo_exact_timesteps(model, total_timesteps=24)
    assert result.rollout_sizes == (12, 12)
    assert model.batch_size == 4 and model.n_epochs == 2
    assert model.policy.optimizer.param_groups[0]["lr"] == .0001
