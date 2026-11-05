"""Small CPU equivalence test; no database, market data or real checkpoints."""
import copy
import hashlib
import inspect
import json
import random
import numpy as np
import pytest

torch = pytest.importorskip('torch')
gym = pytest.importorskip('gymnasium')
from stable_baselines3 import PPO
from train_and_eval.training.critic_diagnostics import train_observed, EXPECTED_TRAIN_HASH


def rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def restore(state):
    random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])


def equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else: assert a==b


@pytest.mark.parametrize('step', [1024,2048])
def test_observer_preserves_update_and_rng(tmp_path,step):
    from stable_baselines3.common.logger import configure
    model=PPO('MlpPolicy',gym.make('CartPole-v1'),n_steps=16,batch_size=8,n_epochs=2,seed=17,device='cpu',policy_kwargs={'net_arch':[8,8]})
    model.set_logger(configure(folder=str(tmp_path/'logger'),format_strings=[]))
    _,callback=model._setup_learn(16,None,reset_num_timesteps=False)
    assert model.collect_rollouts(model.env,callback,model.rollout_buffer,n_rollout_steps=16)
    state=rng(); original=copy.deepcopy(model)
    restore(state);original.train(); expected_rng=rng()
    restore(state);path=tmp_path/'diagnostics.jsonl';train_observed(model,path,step,detail_steps=(1024,))
    equal(expected_rng,rng())
    equal(original.policy.state_dict(),model.policy.state_dict())
    equal(original.policy.optimizer.state_dict(),model.policy.optimizer.state_dict())
    equal(original.logger.name_to_value,model.logger.name_to_value)
    row=json.loads(path.read_text())
    assert len(row['updates']) == (4 if step==1024 else 0)
    assert row['run_step']==step
    if step==1024:
        assert [x['epoch'] for x in row['updates']]==[1,1,2,2]
        assert all('critic_grad_before_clip' in x and 'critic_parameter_delta' in x for x in row['updates'])
    assert not model.policy.optimizer._optimizer_step_pre_hooks
    assert not model.policy.optimizer._optimizer_step_post_hooks


def test_reviewed_source():
    assert hashlib.sha256(inspect.getsource(PPO.train).encode()).hexdigest()==EXPECTED_TRAIN_HASH


def test_debugger_guard(tmp_path):
    import sys
    model=PPO('MlpPolicy',gym.make('CartPole-v1'),n_steps=16,batch_size=8,device='cpu')
    prior=sys.gettrace()
    try:
        sys.settrace(lambda *args:None)
        with pytest.raises(RuntimeError,match='debugger'):
            train_observed(model,tmp_path/'x',1024)
    finally:sys.settrace(prior)


def test_hooks_removed_after_measurement_failure(tmp_path, monkeypatch):
    import sys
    from stable_baselines3.common.logger import configure
    from train_and_eval.training import critic_diagnostics as diagnostics
    model=PPO('MlpPolicy',gym.make('CartPole-v1'),n_steps=16,batch_size=8,n_epochs=2,seed=17,device='cpu',policy_kwargs={'net_arch':[8,8]})
    model.set_logger(configure(folder=str(tmp_path/'logger'),format_strings=[]))
    _,callback=model._setup_learn(16,None,reset_num_timesteps=False)
    model.collect_rollouts(model.env,callback,model.rollout_buffer,n_rollout_steps=16)
    original=diagnostics.measure
    calls=[]
    def fail(model):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('injected measurement failure')
        return original(model)
    monkeypatch.setattr(diagnostics,'measure',fail)
    with pytest.raises(RuntimeError,match='injected'):
        diagnostics.train_observed(model,tmp_path/'x',1024,force_detailed=True)
    assert sys.gettrace() is None
    assert not model.policy.optimizer._optimizer_step_pre_hooks
    assert not model.policy.optimizer._optimizer_step_post_hooks
