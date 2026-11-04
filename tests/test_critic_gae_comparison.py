import importlib.util
import json
from pathlib import Path
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'extra_tools/diagnostics/compare_critic_gae.py'
    spec = importlib.util.spec_from_file_location('gae_compare', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def report():
    return {'status': 'complete', 'source_run_id': 2143, 'planned_steps': [1024],
            'factors': [1, .5, .25, .1], 'rollouts': [
                {'run_step': 1024, 'baseline_reproduced': True,
                 'split_control_reproduced': True, 'trials': [
                     {'factor': f, 'final_over_original': f, 'updates': [{}]*36}
                     for f in [1, .5, .25, .1]]}]}


def test_valid_report(tmp_path):
    path = tmp_path/'report.json'; path.write_text(json.dumps(report()))
    assert list(module().load_report(path, 2143)) == [1024]


@pytest.mark.parametrize('failure', ['source', 'control', 'duplicate', 'trace', 'nan'])
def test_reject_invalid_report(tmp_path, failure):
    data = report()
    if failure == 'source': data['source_run_id'] = 3732
    if failure == 'control': data['rollouts'][0]['baseline_reproduced'] = False
    if failure == 'duplicate': data['rollouts'] *= 2
    if failure == 'trace': data['rollouts'][0]['trials'][0]['updates'] = []
    if failure == 'nan': data['rollouts'][0]['trials'][0]['final_over_original'] = float('nan')
    path = tmp_path/'report.json'; path.write_text(json.dumps(data))
    with pytest.raises(ValueError): module().load_report(path, 2143)
