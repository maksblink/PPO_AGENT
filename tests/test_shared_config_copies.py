"""Equivalent config copies are one run; all paths remain auditable."""
import copy
import json
from pathlib import Path
import pytest
import yaml
from extra_tools.maintenance import audit_run_names as audit
from extra_tools.maintenance import rename_runs as rename
from train_and_eval.dashboard.coverage import read_plan

SAMPLE = {'config_schema_version': 1,
 'run': {'name': 'fixture', 'seed': 1},
 'continuation': {'mode': 'fresh'},
 'data': {'path': 'data/test.parquet',
          'train_range': {'start': '2020-01-06T00:00:00+00:00',
                          'end': '2020-02-03T00:00:00+00:00'},
          'validation_range': {'start': '2020-02-03T00:00:00+00:00',
                               'end': '2020-02-10T00:00:00+00:00'},
          'alignment': 'trim_start'},
 'training': {'duration_unit': 'data_epochs', 'duration_amount': 1},
 'environment': {'window': 40,
                 'context': 'baseline_multiscale_v1',
                 'position_side': 'long_only',
                 'market_timezone': 'America/New_York',
                 'rth_open': '09:30',
                 'rth_close': '16:00',
                 'stake_pln': 1000.0,
                 'fee_bps': 0.2,
                 'swap_long_bps': 2.12,
                 'swap_short_bps': 0.11,
                 'swap_time': '17:00',
                 'swap_timezone': 'America/New_York',
                 'force_close_on_done': True,
                 'reward_scale': 1.0,
                 'exposure_penalty': 0.0,
                 'turnover_penalty': 0.0,
                 'drawdown_penalty': 0.0,
                 'profit_reward_mult': 1.0,
                 'loss_reward_mult': 1.0},
 'ppo': {'policy': 'mlp',
         'device': 'cuda',
         'hidden_sizes': [384, 384, 384],
         'activation': 'tanh',
         'n_steps': 1024,
         'batch_size': 256,
         'n_epochs': 3,
         'learning_rate': 4e-05,
         'gamma': 0.9,
         'gae_lambda': 0.85,
         'clip_range': 0.2,
         'clip_range_vf': None,
         'normalize_advantage': True,
         'ent_coef': 0.0002,
         'vf_coef': 0.5,
         'max_grad_norm': 0.5,
         'target_kl': None,
         'value_head_init_scale': 0.003,
         'initial_long_probability': 0.55},
 'evaluation': {'eval_every_steps': 638976,
                'checkpoint_every_steps': 638976,
                'policy_mode': 'deterministic_argmax',
                'threshold_action': None,
                'probability_threshold': None,
                'best_metric': 'balanced_score',
                'early_stop_patience_evals': 5,
                'device': 'cpu',
                'training_mode': 'scheduled'},
 'logging': {'training_progress_every_steps': 1024,
             'validation_progress_every_steps': 1024},
 'artifacts': {'training_metrics': {'enabled': True, 'every_steps': 1024},
               'evaluation_trajectory': {'mode': 'all'},
               'plots': {'during_run': True}}}


def write(root, rel, cfg, comment=''):
    p=root/rel
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(comment+yaml.safe_dump(cfg,sort_keys=False))
    return p


@pytest.fixture
def copies(tmp_path):
    cfg=copy.deepcopy(SAMPLE);cfg['run']['name']='old_fresh'
    a=write(tmp_path,'configs/stage_one/grid1/old_fresh.yml',cfg)
    b=write(tmp_path,'configs/stage_one/grid3/old_fresh.yml',cfg,'# Same config, different YAML bytes\n')
    return tmp_path,cfg,a,b


def entries(root):
    configs,paths,loaded,groups=audit.collect_configs(root)
    names=audit.proposed_names(configs)
    result=[]
    for name,group in groups.items():
        target=names[name]['new_name']
        items=[dict(x,new_config_path=str(Path(x['old_config_path']).with_name(target+'.yml'))) for x in group]
        result.append(dict(items[0],config_copies=items,old_name=name,new_name=target,
            run_id=None,old_source_run=configs[name]['continuation'].get('source_run')))
    return result


def test_normalized_copies_with_different_raw_bytes(copies):
    root,cfg,a,b=copies
    configs,paths,loaded,groups=audit.collect_configs(root)
    assert len(configs)==1 and len(groups['old_fresh'])==2
    assert len({x['disk_config_sha256'] for x in groups['old_fresh']})==2
    assert len(rename.validate_disk_configs(root,entries(root)))==1


@pytest.mark.parametrize('field,value', [('learning_rate',0.00004),('ent_coef',0.002),('gae_lambda',0.95)])
def test_conflicting_copies_rejected(copies,field,value):
    root,cfg,a,b=copies
    cfg['ppo'][field]=value if cfg['ppo'][field]!=value else value*2
    write(root,b.relative_to(root),cfg)
    with pytest.raises(ValueError,match='Conflicting config copies'):
        audit.collect_configs(root)


def test_change_after_audit_rejected(copies):
    root,cfg,a,b=copies;mapping=entries(root)
    b.write_text(b.read_text()+'# changed\n')
    with pytest.raises(ValueError,match='changed since audit'):
        rename.validate_disk_configs(root,mapping)


def test_added_copy_requires_new_audit(copies):
    root,cfg,a,b=copies;mapping=entries(root)
    write(root,'configs/stage_one/grid4/old_fresh.yml',cfg)
    with pytest.raises(ValueError,match='inventory changed'):
        rename.validate_disk_configs(root,mapping)


def test_removed_copy_requires_new_audit(copies):
    root,cfg,a,b=copies;mapping=entries(root);b.unlink()
    with pytest.raises(ValueError,match='inventory changed'):
        rename.validate_disk_configs(root,mapping)


def test_conflict_with_updated_raw_hash_still_rejected(copies):
    root,cfg,a,b=copies;mapping=entries(root)
    cfg['ppo']['ent_coef']=0.002;write(root,b.relative_to(root),cfg)
    mapping[0]['config_copies'][1]['disk_config_sha256']=rename.sha(b.read_text())
    with pytest.raises(ValueError,match='Conflicting config copies'):
        rename.validate_disk_configs(root,mapping)


def test_legacy_single_file_audit(copies):
    root,cfg,a,b=copies;b.unlink();mapping=entries(root)
    del mapping[0]['config_copies']
    assert len(rename.validate_disk_configs(root,mapping))==1


def queues(root,a,b):
    return [write(root,f'configs/search_queues/q{i}.yml',dict(queue_schema_version=1,name=f'q{i}',configs=[p.relative_to(root).as_posix()])) for i,p in enumerate((a,b))]


def test_coverage_merges_identical_paths(copies):
    root,cfg,a,b=copies
    plan,errors=read_plan(root,queues(root,a,b))
    assert not errors and len(plan)==1
    assert plan['old_fresh']['queues']=={'q0','q1'}
    assert len(plan['old_fresh']['paths'])==2


def test_coverage_rejects_conflict(copies):
    root,cfg,a,b=copies;cfg['ppo']['ent_coef']=0.002;write(root,b.relative_to(root),cfg)
    _,errors=read_plan(root,queues(root,a,b))
    assert errors and 'different config' in errors[0]


def test_symlink_copy_rejected(copies):
    root,cfg,a,b=copies;b.unlink();b.symlink_to(a)
    with pytest.raises(ValueError,match='Symlink'):
        audit.collect_configs(root)


class EmptyResult:
    def mappings(self): return []

class EmptyDatabase:
    def scalar(self, query): return 0
    def execute(self, query): return EmptyResult()


def test_plan_renames_all_copies_and_queues_and_recovers(copies,monkeypatch):
    root,cfg,a,b=copies
    child=copy.deepcopy(cfg);child['run']['name']='old_resume'
    child['continuation']={'mode':'resume','source_run':'old_fresh','checkpoint':'final'}
    ca=write(root,'configs/stage_one/grid1/old_resume.yml',child)
    cb=write(root,'configs/stage_one/grid3/old_resume.yml',child)
    qs=queues(root,a,b)
    for q,ch in zip(qs,(ca,cb)):
        obj=yaml.safe_load(q.read_text());obj['configs'].append(ch.relative_to(root).as_posix());q.write_text(yaml.safe_dump(obj))
    mapping=entries(root)
    snapshot=dict(audit_version=audit.AUDIT_VERSION,name_format=audit.NAME_FORMAT,errors=[],mapping=mapping,
                  counts={'checkpoints':0,'evaluations':0},metadata_references=[])
    monkeypatch.setattr(rename,'run_rows',lambda c:{})
    plan=rename.build_plan(EmptyDatabase(),root,snapshot)
    assert len(plan['mapping'])==2 and not plan['runs']
    rename.finish_files(root,plan,'after')
    rename.finish_files(root,plan,'after')
    for e in mapping:
        for item in e['config_copies']:
            assert not (root/item['old_config_path']).exists()
            cfg=yaml.safe_load((root/item['new_config_path']).read_text())
            assert cfg['run']['name']==e['new_name']
            if e['old_name']=='old_resume':
                assert cfg['continuation']['source_run']==mapping[0]['new_name']
    for q in qs:
        assert all((root/p).is_file() for p in yaml.safe_load(q.read_text())['configs'])
    rename.finish_files(root,plan,'before')
    assert a.exists() and b.exists() and ca.exists() and cb.exists()
