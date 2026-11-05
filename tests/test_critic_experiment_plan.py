import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def load_file(relative, name):
    path=Path(__file__).resolve().parents[1]/relative
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


plan=load_file('train_and_eval/training/critic_experiment_plan.py','plan_test')
prepare=load_file('extra_tools/diagnostics/critic/prepare.py','prepare_test')
compare=load_file('extra_tools/diagnostics/critic/compare.py','compare_test')


def test_absent_sidecar_is_disabled(tmp_path):
    assert plan.load_plan(tmp_path/'x.yml',root=tmp_path,config_sha256='abc',source=None,
                          segment_sizes=[32],rollout_steps=16) is None


def make_plan(tmp_path):
    config=tmp_path/'x.yml';config.write_text('example')
    payload={'format':1,'config_sha256':plan.sha(config),'source_checkpoint_id':None,
             'capture_steps':[16,24,40],'detail_steps':[]}
    plan.sidecar(config).write_text(json.dumps(payload))
    return config,payload


def test_segment_boundary_validation(tmp_path):
    config,payload=make_plan(tmp_path)
    assert plan.load_plan(config,root=tmp_path,config_sha256=plan.sha(config),source=None,
                          segment_sizes=[24,16],rollout_steps=16)==payload


@pytest.mark.parametrize('failure',['checksum','step','duplicate','source'])
def test_plan_rejects_mismatch(tmp_path,failure):
    config,payload=make_plan(tmp_path)
    if failure=='checksum':payload['config_sha256']='changed'
    if failure=='step':payload['capture_steps']=[17]
    if failure=='duplicate':payload['capture_steps']=[16,16]
    if failure=='source':payload['source_checkpoint_id']=1
    plan.sidecar(config).write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        plan.load_plan(config,root=tmp_path,config_sha256=plan.sha(config),source=None,
                       segment_sizes=[24,16],rollout_steps=16)


def test_checkpoint_bytes_are_checked(tmp_path):
    config,payload=make_plan(tmp_path)
    checkpoint=tmp_path/'cp.zip';checkpoint.write_bytes(b'original')
    payload.update(source_checkpoint_id=123,source_checkpoint_sha256=plan.sha(checkpoint))
    plan.sidecar(config).write_text(json.dumps(payload))
    source=SimpleNamespace(checkpoint_id=123,relative_path='cp.zip')
    plan.load_plan(config,root=tmp_path,config_sha256=plan.sha(config),source=source,segment_sizes=[24,16],rollout_steps=16)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        plan.load_plan(config,root=tmp_path,config_sha256=plan.sha(config),source=source,segment_sizes=[24,16],rollout_steps=16)


def test_selection_uses_neighbor_records_and_explicit_steps():
    records=[dict(run_step=s,value_error_std=1.,value_error_mean=0.,post_train_value_mse=m)
             for s,m in [(16,1),(32,8),(48,2),(64,1)]]
    assert prepare.select_steps(records,[64],1,1)==([16,32,48,64],[32])
    with pytest.raises(ValueError):prepare.select_steps(records,[17],0,0)


def test_empty_selection_rejected():
    with pytest.raises(ValueError):prepare.select_steps([],[],0,0)


def test_report_accepts_dynamic_step_and_update_count(tmp_path):
    data={'status':'complete','planned_steps':[42],'factors':[1,.5],
          'rollouts':[{'run_step':42,'baseline_reproduced':True,'split_control_reproduced':True,
                       'trials':[{'factor':1,'final_over_original':1}, {'factor':.5,'final_over_original':.4}]}]}
    path=tmp_path/'report.json';path.write_text(json.dumps(data))
    assert list(compare.load_report(path)[1])==[42]
    data['rollouts'][0]['split_control_reproduced']=False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):compare.load_report(path)
