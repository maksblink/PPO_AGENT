"""Multi-capture isolation, exact replay, and fail-closed report checks."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
gym = pytest.importorskip('gymnasium')
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from train_and_eval.training.critic_diagnostics import train_observed
from train_and_eval.training.critic_rollout_experiment import (
    MULTI_CAPTURE_STEPS, CONTROL_STEPS, SPIKE_STEPS, capture_steps, step_category,
    parameter_state, rng_state, restore_rng, exact, verify_capture, run_trial,
)


def test_selection_and_modes():
    assert len(MULTI_CAPTURE_STEPS) == 16
    assert list(MULTI_CAPTURE_STEPS) == sorted(set(MULTI_CAPTURE_STEPS))
    assert all(s > 0 and s % 1024 == 0 for s in MULTI_CAPTURE_STEPS)
    assert capture_steps(False) == ()
    assert capture_steps(True) == (404480,)
    assert capture_steps('multi') == MULTI_CAPTURE_STEPS
    assert sum(step_category(s) == 'known_spike' for s in MULTI_CAPTURE_STEPS) == 3
    assert sum(step_category(s) == 'spike_neighbor' for s in MULTI_CAPTURE_STEPS) == 6
    assert sum(step_category(s) == 'scheduled_control' for s in MULTI_CAPTURE_STEPS) == 7
    with pytest.raises(ValueError): capture_steps('wrong')
    with pytest.raises(ValueError): step_category(1)


@pytest.mark.parametrize('mode,steps', [('multi', (1024, 128000, 475136)),
                                         ('gae85', (199680, 248832, 273408))])
def test_multiple_captures_preserve_training_and_reproduce(tmp_path, mode, steps):
    model = PPO('MlpPolicy', gym.make('CartPole-v1'), n_steps=16, batch_size=8,
                n_epochs=2, max_grad_norm=1e6, seed=19, device='cpu',
                policy_kwargs={'net_arch': [8, 8]})
    model.set_logger(configure(folder=str(tmp_path/'logger'), format_strings=[]))
    model.learn(16)
    _, callback = model._setup_learn(64, None, reset_num_timesteps=False)
    artifact_root = tmp_path / "artifacts" / "runs" / "00000001"
    assert not artifact_root.exists()
    # First capture must work before any run artifacts have been written.
    for step in steps:
        assert model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=16)
        original = copy.deepcopy(model)
        state = rng_state(False)
        original.train()
        expected = {'model': parameter_state(original), 'rng': rng_state(False)}
        restore_rng(state)
        train_observed(model, artifact_root/'trace.jsonl', step, capture=mode)
        exact(expected, {'model': parameter_state(model), 'rng': rng_state(False)})
        directory = artifact_root/f'critic_capture_{step}'
        assert verify_capture(directory)['run_step'] == step
        expected_row = json.loads((directory/'expected_diagnostics.json').read_text())
        assert expected_row['detailed'] and len(expected_row['updates']) == 4
        out = tmp_path/f'trial_{step}'; out.mkdir()
        for label, factor in [('original', None), ('split', 1.0)]:
            result, row = run_trial(directory, out, label, factor)
            exact(expected, result)
            exact(expected_row, row)
        # Offline replay must not alter continuation of the capture training.
        restore_rng(expected['rng'])
    assert len(list(artifact_root.glob('critic_capture_*'))) == 3


def load_sweep():
    path = Path(__file__).resolve().parents[1]/'extra_tools/diagnostics/critic_lr_sweep.py'
    spec = importlib.util.spec_from_file_location('critic_sweep_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_report():
    def trial(factor, mse):
        return {'critic_lr_factor': factor, 'before_mse': 1e-10, 'after_mse': mse,
                'peak_mse': mse, 'best_mse': mse, 'after_nmse': None, 'after_ev': None,
                'updates': [{'update': 1, 'epoch': 1, 'mse': mse}],
                'clipped_updates': 0, 'actor_changed_tensors': 0}
    return {'status': 'complete', 'baseline_reproduced': True,
            'split_control_reproduced': True, 'trials': [trial(1, 1e-7), trial(.5, 1e-12)]}


def test_summary_preserves_small_errors_and_flags():
    report = sample_report()
    report['trials'][1]['actor_changed_tensors'] = 2
    report['trials'][1]['clipped_updates'] = 1
    result = load_sweep().summarize(report, [1, .5])
    assert result[1]['final_over_original'] == pytest.approx(1e-5)
    assert result[1]['final_over_initial'] == pytest.approx(.01)
    assert result[1]['actor_changed_tensors'] == 2
    assert result[1]['clipped_updates'] == 1
    assert result[1]['updates_above_initial'] == 0
    assert result[0]['updates_above_initial'] == 1


@pytest.mark.parametrize('key', ['baseline_reproduced', 'split_control_reproduced'])
def test_summary_rejects_failed_control(key):
    report = sample_report(); report[key] = False
    with pytest.raises(RuntimeError, match='controls'):
        load_sweep().summarize(report, [1, .5])


def test_summary_checks_factor_and_initial_identity():
    report = sample_report()
    with pytest.raises(RuntimeError, match='factors'):
        load_sweep().summarize(report, [1, .25])
    report['trials'][1]['before_mse'] *= 2
    with pytest.raises(RuntimeError, match='same error'):
        load_sweep().summarize(report, [1, .5])


def test_summary_handles_zero_baseline():
    report = sample_report()
    report['trials'][0]['after_mse'] = 0
    assert load_sweep().summarize(report, [1, .5])[1]['final_over_original'] is None


def test_sweep_executes_guarded_child_and_combines(tmp_path, monkeypatch):
    import sys
    import train_and_eval.training.critic_rollout_experiment as core
    model = PPO('MlpPolicy', gym.make('CartPole-v1'), n_steps=16, batch_size=4,
                n_epochs=9, seed=31, device='cpu', policy_kwargs={'net_arch': [8, 8]})
    model.set_logger(configure(folder=str(tmp_path/'logger'), format_strings=[]))
    model.learn(16)
    _, callback = model._setup_learn(16, None, reset_num_timesteps=False)
    model.collect_rollouts(model.env, callback, model.rollout_buffer, n_rollout_steps=16)
    train_observed(model, tmp_path/'trace.jsonl', 64512, capture='multi')
    monkeypatch.setattr(core, 'MULTI_CAPTURE_STEPS', (64512,))
    monkeypatch.setattr(sys, 'argv', ['sweep', '--run-dir', str(tmp_path), '--factors', '.5'])
    load_sweep().main()
    reports = list((tmp_path/'critic_lr_sweep').glob('*/report.json'))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report['status'] == 'complete'
    assert report['planned_steps'] == [64512]
    assert report['rollouts'][0]['category'] == 'scheduled_control'
    assert len(report['rollouts'][0]['trials']) == 2
    assert len(report['rollouts'][0]['trials'][1]['updates']) == 36


def test_sweep_missing_capture_stops_before_outputs(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, 'argv', ['sweep', '--run-dir', str(tmp_path)])
    with pytest.raises(FileNotFoundError): load_sweep().main()
    assert not (tmp_path/'critic_lr_sweep').exists()


def test_gae85_capture_plan():
    from train_and_eval.training.critic_rollout_experiment import capture_steps, step_category
    base, extended = set(capture_steps('multi')), set(capture_steps('gae85'))
    assert len(base) == 16 and len(extended) == 19
    assert extended - base == {199680, 248832, 273408}
    assert all(step_category(s) == 'gae85_mse_increase' for s in extended - base)
