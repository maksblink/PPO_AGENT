"""Capture integrity and exact replay controls on a small CPU PPO model."""
import copy
import json
import pytest

torch=pytest.importorskip('torch')
gym=pytest.importorskip('gymnasium')
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from train_and_eval.training.critic_diagnostics import train_observed
from train_and_eval.training.critic_rollout_experiment import (
    CAPTURE_STEP,capture_before,capture_after,verify_capture,run_trial,
    parameter_state,rng_state,restore_rng,exact,split_critic_lr,
)


def model_and_capture(tmp_path):
    model=PPO('MlpPolicy',gym.make('CartPole-v1'),n_steps=16,batch_size=8,n_epochs=2,
              max_grad_norm=1e6,seed=19,device='cpu',policy_kwargs={'net_arch':[8,8]})
    model.set_logger(configure(folder=str(tmp_path/'logger'),format_strings=[]))
    # Warm up Adam so tests cover existing per-parameter optimizer history.
    model.learn(16)
    _,callback=model._setup_learn(16,None,reset_num_timesteps=False)
    assert model.collect_rollouts(model.env,callback,model.rollout_buffer,n_rollout_steps=16)
    source=copy.deepcopy(model)
    state=rng_state(False)
    restore_rng(state);source.train();expected={'model':parameter_state(source),'rng':rng_state(False)}
    restore_rng(state)
    path=tmp_path/'diagnostics.jsonl'
    train_observed(model,path,CAPTURE_STEP,capture=True)
    exact(expected,{'model':parameter_state(model),'rng':rng_state(False)})
    directory=tmp_path/f'critic_capture_{CAPTURE_STEP}'
    return model,directory


def test_capture_and_both_baselines(tmp_path):
    model,directory=model_and_capture(tmp_path)
    meta=verify_capture(directory)
    assert meta['run_step']==CAPTURE_STEP
    expected=torch.load(directory/'expected_after.pt',weights_only=False,map_location='cpu')
    expected_row=json.loads((directory/'expected_diagnostics.json').read_text())
    output=tmp_path/'trials';output.mkdir()
    for label,factor in [('original',None),('control',1.0)]:
        result,row=run_trial(directory,output,label,factor)
        exact(expected,result);exact(expected_row,row)


def test_lower_lr_changes_only_critic_without_clipping(tmp_path):
    _,directory=model_and_capture(tmp_path)
    output=tmp_path/'trials';output.mkdir()
    baseline,row=run_trial(directory,output,'original')
    variant,other=run_trial(directory,output,'smaller',0.25)
    exact(baseline['rng'],variant['rng'])
    for name,value in baseline['model']['weights'].items():
        if name.startswith(('mlp_extractor.policy_net.','action_net.')):
            exact(value,variant['model']['weights'][name],name)
    assert any(not torch.equal(value,variant['model']['weights'][name])
               for name,value in baseline['model']['weights'].items()
               if name.startswith(('mlp_extractor.value_net.','value_net.')))
    for name,state in variant['model']['optimizer_by_name'].items():
        expected_lr=baseline['model']['optimizer_by_name'][name]['options']['lr']
        if name.startswith(('mlp_extractor.value_net.','value_net.')):expected_lr*=0.25
        assert state['options']['lr']==expected_lr
    assert len(other['updates'])==len(row['updates'])==4


def test_capture_tampering_rejected(tmp_path):
    _,directory=model_and_capture(tmp_path)
    with (directory/'before.zip').open('ab') as f:f.write(b'changed')
    with pytest.raises(RuntimeError,match='checksum'):verify_capture(directory)


def test_incomplete_capture_rejected(tmp_path):
    (tmp_path/'runtime.json').write_text('{}')
    with pytest.raises(FileNotFoundError):verify_capture(tmp_path)


def test_capture_refuses_existing_directory(tmp_path):
    model,directory=model_and_capture(tmp_path)
    with pytest.raises(FileExistsError):capture_before(model,tmp_path/'diagnostics.jsonl',CAPTURE_STEP)


@pytest.mark.parametrize('factor',[0,-1,float('nan'),float('inf')])
def test_invalid_factors_rejected(factor):
    with pytest.raises(ValueError):split_critic_lr(None,factor)


def test_cli_full_experiment(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    _,directory=model_and_capture(tmp_path)
    script=Path(__file__).resolve().parents[1]/'extra_tools/diagnostics/critic_lr_experiment.py'
    result=subprocess.run([sys.executable,str(script),'--capture',str(directory)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    reports=list((directory.parent/'critic_lr_experiment').glob('*/report.json'))
    assert len(reports)==1
    report=json.loads(reports[0].read_text())
    assert report['status']=='complete'
    assert report['baseline_reproduced'] and report['split_control_reproduced']
    assert [x['critic_lr_factor'] for x in report['trials']]==[1.0,0.5,0.25,0.1]
