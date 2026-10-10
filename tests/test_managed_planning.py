from types import SimpleNamespace
import pytest
import yaml
from train_and_eval.queue_execution.core import plan_jobs, runnable, ThermalGate


def config(root, name, source=None):
    path=root/f'{name}.yml'
    path.write_text(yaml.safe_dump({'run':{'name':name},'continuation':
        {'mode':'resume','source_run':source,'checkpoint':'final'} if source else {'mode':'fresh'}}))
    return SimpleNamespace(path=path,name=name)


def test_dependency_readiness_uses_parent_status_not_position(tmp_path):
    child=config(tmp_path,'child','parent');parent=config(tmp_path,'parent')
    jobs=plan_jobs([(1,child),(2,parent)])
    assert not runnable(jobs[0],{})
    for state in ('running','pending','failed','cancelled'):
        assert not runnable(jobs[0],{'parent':state})
    assert runnable(jobs[0],{'parent':'completed'})


def test_cycles_and_duplicates_rejected(tmp_path):
    a=config(tmp_path,'a','b');b=config(tmp_path,'b','a')
    with pytest.raises(ValueError):plan_jobs([(1,a),(2,b)])
    with pytest.raises(ValueError):plan_jobs([(1,a),(2,a)])


def test_production_threshold_boundaries():
    gate=ThermalGate(91.5,84.5)
    assert gate.admit(91.49)
    assert not gate.admit(91.5)
    assert not gate.admit(85)
    assert gate.admit(84.5)
    assert not gate.admit(None)
    assert not gate.admit(float('nan'))


def test_external_parent_is_a_dependency(tmp_path):
    job=plan_jobs([(1,config(tmp_path,'child','outside'))])[0]
    assert job.source=='outside'
    assert not runnable(job,{})
