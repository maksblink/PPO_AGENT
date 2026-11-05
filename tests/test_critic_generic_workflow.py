"""Generic capture/sweep integration: arbitrary steps and non-36 update counts."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import pytest

torch=pytest.importorskip('torch')
gym=pytest.importorskip('gymnasium')
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from train_and_eval.training.critic_diagnostics import train_observed
from train_and_eval.training.critic_rollout_experiment import parameter_state,rng_state,restore_rng,exact,verify_capture


def test_generic_multicapture_and_sweep(tmp_path):
    model=PPO('MlpPolicy',gym.make('CartPole-v1'),n_steps=16,batch_size=8,n_epochs=2,
              seed=13,device='cpu',policy_kwargs={'net_arch':[8,8]})
    model.set_logger(configure(folder=str(tmp_path/'logger'),format_strings=[]))
    model.learn(16)
    _,callback=model._setup_learn(32,None,reset_num_timesteps=False)
    directory=tmp_path/'new'/'run'
    for step in (16,32):
        model.collect_rollouts(model.env,callback,model.rollout_buffer,n_rollout_steps=16)
        original=copy.deepcopy(model);state=rng_state(False)
        original.train();expected={'model':parameter_state(original),'rng':rng_state(False)}
        restore_rng(state)
        train_observed(model,directory/'critic_diagnostics.jsonl',step,capture_steps=(16,32))
        exact(expected,{'model':parameter_state(model),'rng':rng_state(False)})
        assert verify_capture(directory/f'critic_capture_{step}')['rollout_samples']==16
    (directory/'critic_experiment.json').write_text(json.dumps({
        'format':1,'run_id':999,'capture_steps':[16,32],'normalized_config':{}}))
    script=Path(__file__).resolve().parents[1]/'extra_tools/diagnostics/critic/sweep.py'
    result=subprocess.run([sys.executable,str(script),'--run-dir',str(directory),
                           '--factors','.5'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    reports=list((directory/'critic_lr_sweep').glob('*/report.json'))
    assert len(reports)==1
    report=json.loads(reports[0].read_text())
    assert report['status']=='complete'
    assert report['planned_steps']==[16,32]
    assert all(len(t['updates'])==4 for r in report['rollouts'] for t in r['trials'])
    compare=script.with_name('compare.py')
    result=subprocess.run([sys.executable,str(compare),'--report',str(reports[0]),
                           '--report',str(reports[0])],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'Shared selected steps: 2' in result.stdout
