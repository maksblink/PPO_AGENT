"""Fixed-width names and name-only configuration rewrites."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from extra_tools.maintenance.audit_run_names import scientific_token, proposed_names
from extra_tools.maintenance.rename_runs import rename_config, sha
from train_and_eval.run_config import RunConfig, normalize_config


@pytest.mark.parametrize('rate,token', [
    (0.00035, '3p50000em4'), (0.000025, '2p50000em5'),
    (0.000015, '1p50000em5'), (0.00005, '5p00000em5'),
    ('0.000000001', '1p00000em9'), ('0', '0p00000ep0'),
])
def test_exact_fixed_width(rate, token):
    assert scientific_token(rate) == token
    assert len(token) == 10


@pytest.mark.parametrize('rate', ['0.0000000001', '0.00001500001', 'NaN', 'Infinity', '-0.1'])
def test_no_rounding_or_invalid_lr(rate):
    with pytest.raises(ValueError):
        scientific_token(rate)


def chain():
    cfg = yaml.safe_load((Path(__file__).parent/'fixtures/temporal_run.yml').read_text())
    cfg['run']['name'] = 'fresh'
    cfg['ppo']['learning_rate'] = 0.000025
    cfg['ppo']['hidden_sizes'] = [384]*3
    cfg['ppo']['gamma'] = 0.9
    cfg['ppo']['n_epochs'] = 3
    cfg['run']['seed'] = 1
    cfg['continuation'] = {'mode': 'fresh'}
    child = copy.deepcopy(cfg)
    child['run']['name'] = 'resume'
    child['continuation'] = {'mode': 'resume', 'source_run': 'fresh', 'checkpoint': 'final'}
    return {'fresh': cfg, 'resume': child}


def test_parent_child_names_and_raw_yaml_hashes():
    configs = chain()
    names = {old: value['new_name'] for old, value in proposed_names(configs).items()}
    assert names == {'fresh': 'lr2p50000em5_f_v00_gae8p50000em1_ent2p00000em4_g9p00000em1_w384x3_ne03_s01',
                     'resume': 'lr2p50000em5_r_v01_gae8p50000em1_ent2p00000em4_g9p00000em1_w384x3_ne03_s01'}
    raw = yaml.safe_dump(configs['resume'], sort_keys=False).replace('2.5e-05', '0.000025') + '# preserve comment\n'
    renamed, norm, digest = rename_config(raw, names)
    assert 'learning_rate: 0.000025' in renamed
    assert renamed.endswith('# preserve comment\n')
    assert norm['continuation']['source_run'] == names['fresh']
    assert norm['run']['name'] == names['resume']
    expected = copy.deepcopy(configs['resume'])
    expected['run']['name'] = names['resume']
    expected['continuation']['source_run'] = names['fresh']
    assert json.loads(normalize_config(RunConfig.model_validate(expected))) == norm
    assert sha(normalize_config(RunConfig.model_validate(norm))) == digest
    assert sha(renamed) != sha(raw)
    assert rename_config(renamed, {value: value for value in names.values()})[0] == renamed


def test_different_parent_lr_rejected():
    configs = chain()
    configs['resume']['ppo']['learning_rate'] = 0.000015
    with pytest.raises(ValueError, match='Different learning_rate'):
        proposed_names(configs)


def test_duplicate_destination_rejected():
    configs = chain()
    configs['duplicate'] = copy.deepcopy(configs['fresh'])
    configs['duplicate']['run']['name'] = 'duplicate'
    with pytest.raises(ValueError, match='not unique'):
        proposed_names(configs)


def test_file_only_recovery_rolls_forward_but_rejects_new_db_run(monkeypatch):
    from extra_tools.maintenance import rename_runs as tool
    plan = {'runs': [], 'file_only': True}
    monkeypatch.setattr(tool, 'run_rows', lambda _: {})
    assert tool.db_side(None, plan) == 'after'
    monkeypatch.setattr(tool, 'run_rows', lambda _: {1: {}})
    with pytest.raises(ValueError, match='Database differs'):
        tool.db_side(None, plan)
