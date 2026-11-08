"""Full PPO updates and checkpoint reloads, without a database or market data."""
import copy
import json

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from stable_baselines3 import PPO

from tests.test_ppo_adapter import TinyEnvironment, _ppo_config
from train_and_eval.ppo.adapter import create_ppo_model, ppo_resume_custom_objects, load_ppo_model_file
from train_and_eval.ppo.experimental_split_lr import SplitLRPolicy, verify_resume_factor
from train_and_eval.run_config import PPOSection


def config(factor=None):
    data = _ppo_config().model_dump()
    if factor is not None:
        data['experimental_critic_lr_factor'] = factor
    return PPOSection.model_validate(data)


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_equal(x, y)
    else:
        assert a == b


def named_state(model):
    return {n: copy.deepcopy(model.policy.optimizer.state[p]) for n, p in model.policy.named_parameters()}


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, '0.5'])
def test_invalid_factor(value):
    data = _ppo_config().model_dump()
    data['experimental_critic_lr_factor'] = value
    with pytest.raises(ValidationError): PPOSection.model_validate(data)


def test_disabled_serialization_and_optimizer_unchanged():
    data = _ppo_config().model_dump()
    assert 'experimental_critic_lr_factor' not in data
    assert 'experimental_critic_lr_factor' not in json.loads(_ppo_config().model_dump_json())
    data['experimental_critic_lr_factor'] = None
    assert PPOSection.model_validate(data).model_dump() == _ppo_config().model_dump()
    model = create_ppo_model(TinyEnvironment(), config(), seed=1)
    assert type(model.policy.optimizer) is torch.optim.Adam
    assert len(model.policy.optimizer.param_groups) == 1


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_equal_lr_control_identical_weights_adam_rng_after_training(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable; run this control on the training laptop')
    torch.set_num_threads(1)
    baseline = create_ppo_model(TinyEnvironment(), config().model_copy(update={"device": device}), seed=7)
    baseline.learn(64)
    expected_rng = torch.get_rng_state().clone()
    expected_cuda = torch.cuda.get_rng_state_all() if device == "cuda" else []
    expected_numpy = np.random.get_state()
    control = create_ppo_model(TinyEnvironment(), config(1.0).model_copy(update={"device": device}), seed=7)
    control.learn(64)
    assert_equal(baseline.policy.state_dict(), control.policy.state_dict())
    assert_equal(named_state(baseline), named_state(control))
    assert torch.equal(expected_rng, torch.get_rng_state())
    for a, b in zip(expected_cuda, torch.cuda.get_rng_state_all() if device == "cuda" else []):
        assert torch.equal(a, b)
    actual_numpy = np.random.get_state()
    assert expected_numpy[0] == actual_numpy[0]
    assert np.array_equal(expected_numpy[1], actual_numpy[1])
    assert expected_numpy[2:] == actual_numpy[2:]


def test_ratio_survives_ppo_schedule_updates_and_has_separate_groups():
    model = create_ppo_model(TinyEnvironment(), config(.5), seed=1)
    seen = []
    def record(opt, args, kwargs):
        seen.append(tuple(g['lr'] for g in opt.param_groups))
    handle = model.policy.optimizer.register_step_post_hook(record)
    model.lr_schedule = lambda progress: 1e-4 + 2e-4 * progress
    model.learn(32)
    handle.remove()
    assert len(seen) == 16
    assert len({a for a, c in seen}) > 1
    assert all(c == a * .5 for a, c in seen)
    assert model.policy.optimizer.defaults['eps'] == 1e-5
    actor, critic = model.policy.optimizer.param_groups
    assert set(map(id, actor['params'])).isdisjoint(map(id, critic['params']))
    assert set(map(id, actor['params'] + critic['params'])) == set(map(id, model.policy.parameters()))


def test_checkpoint_roundtrip_and_next_training_segment(tmp_path):
    model = create_ppo_model(TinyEnvironment(), config(.5), seed=4)
    model.learn(32)
    path = tmp_path/'split.zip'
    model.save(path)
    restored = PPO.load(path, env=TinyEnvironment(), device='cpu')
    verify_resume_factor(restored, .5)
    assert_equal(model.policy.state_dict(), restored.policy.state_dict())
    assert_equal(model.policy.optimizer.state_dict(), restored.policy.optimizer.state_dict())
    model.set_random_seed(13)
    model.learn(32, reset_num_timesteps=True)
    restored.set_random_seed(13)
    restored.learn(32, reset_num_timesteps=True)
    assert_equal(model.policy.state_dict(), restored.policy.state_dict())
    assert_equal(named_state(model), named_state(restored))
    # Same pathway used by project resume, with a new base/actor LR.
    new_config = config(.5).model_copy(update={'learning_rate': 1e-4})
    resumed = load_ppo_model_file(path, environment=TinyEnvironment(), device='cpu',
        custom_objects=ppo_resume_custom_objects(new_config, seed=4))
    verify_resume_factor(resumed, .5)
    resumed.learn(8)
    assert [g['lr'] for g in resumed.policy.optimizer.param_groups] == [1e-4, 5e-5]
    evaluation = PPO.load(path, device='cpu')
    assert np.array_equal(evaluation.predict(np.zeros(4), deterministic=True)[0],
                          PPO.load(path, device='cpu').predict(np.zeros(4), deterministic=True)[0])


def test_resume_factor_mismatch_and_optimizer_tampering_rejected():
    model = create_ppo_model(TinyEnvironment(), config(.5), seed=1)
    for factor in (None, 1.0, .25):
        with pytest.raises(ValueError, match='resume mismatch'): verify_resume_factor(model, factor)
    normal = create_ppo_model(TinyEnvironment(), config(), seed=1)
    with pytest.raises(ValueError, match='resume mismatch'): verify_resume_factor(normal, .5)
    state = copy.deepcopy(model.policy.optimizer.state_dict())
    state['param_groups'][1]['split_lr_factor'] = .1
    with pytest.raises(ValueError, match='metadata mismatch'):
        model.policy.optimizer.load_state_dict(state)


def test_standalone_policy_roundtrip(tmp_path):
    model = create_ppo_model(TinyEnvironment(), config(.5), seed=1)
    path = tmp_path/'policy.pt'
    model.policy.save(path)
    restored = SplitLRPolicy.load(path, device='cpu')
    assert_equal(model.policy.state_dict(), restored.state_dict())
    assert type(restored.optimizer) is torch.optim.Adam


def test_run_config_resume_rejects_factor_change():
    from train_and_eval.run_config import validate_resume_compatibility, ResumeCompatibilityError
    # Reuse a real validated fixture rather than recreating schema assumptions.
    from tests.test_run_config import _valid_config
    from train_and_eval.run_config import RunConfig, normalize_config
    raw = _valid_config()
    source = RunConfig.model_validate(raw)
    assert 'experimental_critic_lr_factor' not in json.loads(normalize_config(source))['ppo']
    raw['run']['name'] = 'experiment_child'
    raw['continuation'] = {'mode': 'resume', 'source_run': source.run.name, 'checkpoint': 'final'}
    raw['ppo']['experimental_critic_lr_factor'] = .5
    current = RunConfig.model_validate(raw)
    assert json.loads(normalize_config(current))['ppo']['experimental_critic_lr_factor'] == .5
    with pytest.raises(ResumeCompatibilityError, match='experimental_critic_lr_factor'):
        validate_resume_compatibility(current, source)
    enabled_source = source.model_copy(update={'ppo': source.ppo.model_copy(update={'experimental_critic_lr_factor': .5})})
    validate_resume_compatibility(current, enabled_source)


def test_name_migration_refuses_to_erase_experiment_identity():
    from extra_tools.maintenance.audit_run_names import proposed_names
    from tests.test_run_config import _valid_config
    raw = _valid_config()
    raw['ppo']['experimental_critic_lr_factor'] = .5
    with pytest.raises(ValueError, match='outside canonical name migration'):
        proposed_names({raw['run']['name']: raw})


def test_project_checkpoint_loader_checks_config_against_zip(tmp_path):
    from types import SimpleNamespace
    from tests.test_ppo_checkpoints import FakeSession, FakeSessionFactory
    from train_and_eval.artifact_storage.storage import ArtifactStorage
    from train_and_eval.ppo.checkpoints import persist_ppo_checkpoint, load_persisted_ppo_checkpoint
    model = create_ppo_model(TinyEnvironment(), config(.5), seed=5)
    model.learn(8)
    result = persist_ppo_checkpoint(FakeSessionFactory(FakeSession()),
        ArtifactStorage(project_root=tmp_path), model, run_id=1, run_step=8, save_reason='final')
    checkpoint = SimpleNamespace(id=result.checkpoint_id, relative_path=result.relative_path,
        sha256=result.sha256, size_bytes=result.size_bytes, model_step=result.model_step)
    kwargs = dict(environment=TinyEnvironment(), device='cpu', seed=5, project_root=tmp_path)
    loaded = load_persisted_ppo_checkpoint(checkpoint, training_config=config(.5), **kwargs)
    assert_equal(named_state(model), named_state(loaded))
    with pytest.raises(ValueError, match='resume mismatch'):
        load_persisted_ppo_checkpoint(checkpoint, training_config=config(), **kwargs)


def test_identical_gradients_change_critic_only():
    control = create_ppo_model(TinyEnvironment(), config(1.0), seed=19)
    experiment = create_ppo_model(TinyEnvironment(), config(.5), seed=19)
    before = {n: p.detach().clone() for n, p in control.policy.named_parameters()}
    for model in (control, experiment):
        for p in model.policy.parameters(): p.grad = torch.ones_like(p)
        model.policy.optimizer.step()
    control_params = dict(control.policy.named_parameters())
    for name, p in experiment.policy.named_parameters():
        if name.startswith(('mlp_extractor.policy_net.', 'action_net.')):
            assert torch.equal(p, control_params[name])
        else:
            assert not torch.equal(p, control_params[name])
            assert torch.linalg.vector_norm(p - before[name]) < torch.linalg.vector_norm(control_params[name] - before[name])
    assert_equal(named_state(control), named_state(experiment))


def test_unsupported_feature_extractor_rejected():
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    with pytest.raises(ValueError, match='FlattenExtractor'):
        SplitLRPolicy(TinyEnvironment().observation_space, TinyEnvironment().action_space,
            lambda _: 1e-4, critic_lr_factor=.5, features_extractor_class=BaseFeaturesExtractor)
