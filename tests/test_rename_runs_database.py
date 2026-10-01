"""Real PostgreSQL rename and recovery, using an isolated test schema."""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
import pytest
import yaml
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from tests.test_clean_training import cleanup_database
from train_and_eval.database.models import Run, Checkpoint, TrainingMetric
from train_and_eval.run_config import RunConfig, normalize_config
from extra_tools import rename_runs as tool
from extra_tools.audit_run_names import proposed_names


@pytest.fixture
def case(cleanup_database, tmp_path):
    engine=cleanup_database
    cfg=yaml.safe_load((Path(__file__).parent/'fixtures/temporal_run.yml').read_text())
    entries=[];refs=[];cpid=None
    with Session(engine) as s:
        for index,name in enumerate(['old_fresh','old_resume']):
            c=copy.deepcopy(cfg);c['run']['name']=name
            c['continuation']={'mode':'fresh'} if index==0 else {'mode':'resume','source_run':'old_fresh','checkpoint':'final'}
            raw=yaml.safe_dump(c,sort_keys=False);norm=normalize_config(RunConfig.model_validate(c));now=datetime.now(timezone.utc)
            r=Run(name=name,continuation_mode='fresh' if index==0 else 'resume',source_checkpoint_id=cpid,
                git_commit='a'*40,git_branch='master',config_schema_version=1,seed=1,
                config_sha256=tool.sha(raw),normalized_config_sha256=tool.sha(norm),raw_config_yaml=raw,normalized_config_json=json.loads(norm),
                data_path='data/source.parquet',data_sha256='b'*64,duration_unit='data_epochs',duration_amount=1,
                split_index=10,train_rows=10,validation_rows=10,steps_per_data_epoch=10,training_steps_requested=10,
                training_steps_completed=10,data_epochs_completed=1,status='completed',started_at=now,finished_at=now)
            s.add(r);s.flush()
            cp=Checkpoint(run_id=r.id,run_step=10,model_step=(index+1)*10,save_reason='final',
                          relative_path=f'artifacts/runs/{r.id:08d}/checkpoints/model.zip',sha256='c'*64,size_bytes=7)
            s.add(cp);s.flush();cpid=cp.id
            s.add(TrainingMetric(run_id=r.id,run_step=1,model_step=1,rollout_number=1))
            p=tmp_path/cp.relative_path;p.parent.mkdir(parents=True);p.write_text('weights')
            old=f'configs/stage_one/{name}.yml';newname=f'{"f" if index==0 else "r"}_v{index}_test'
            p=tmp_path/old;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(raw)
            entries.append(dict(run_id=r.id,old_name=name,new_name=newname,old_config_path=old,
                new_config_path=f'configs/stage_one/{newname}.yml',old_source_run=None if index==0 else 'old_fresh',
                source_checkpoint_id=r.source_checkpoint_id,config_sha256=r.config_sha256,
                normalized_config_sha256=r.normalized_config_sha256,disk_config_sha256=tool.sha(raw)))
            refs.append({'path':old,'sha256':tool.sha(raw)})
            rel=f'artifacts/runs/{r.id:08d}/reports/summary.json';p=tmp_path/rel;p.parent.mkdir()
            data=tool.dump({'run_id':r.id,'run_name':name,'git_commit':'a'*40});p.write_text(data);refs.append({'path':rel,'sha256':tool.sha(data)})
        s.commit()
    q='configs/search_queues/test.yml';p=tmp_path/q;p.parent.mkdir(parents=True)
    raw=yaml.safe_dump({'queue_schema_version':1,'name':'test','configs':[e['old_config_path'] for e in entries]});p.write_text(raw);refs.append({'path':q,'sha256':tool.sha(raw)})
    configs = {e['old_name']: json.loads(normalize_config(RunConfig.model_validate(yaml.safe_load((tmp_path/e['old_config_path']).read_text())))) for e in entries}
    names = proposed_names(configs)
    for e in entries:
        e['new_name'] = names[e['old_name']]['new_name']
        e['new_config_path'] = str(Path(e['old_config_path']).with_name(e['new_name']+'.yml'))
    audit={'audit_version':2,'lr_decimal_places':9,'errors':[],'mapping':entries,'counts':{'checkpoints':2,'evaluations':0},'metadata_references':refs}
    return engine,tmp_path,audit


def test_preview_cancel_and_full_migration(case):
    engine,root,audit=case
    with engine.connect() as c:
        original=tool.run_rows(c)
        checkpoints=list(c.execute(select(Checkpoint.__table__)).mappings())
        metrics=list(c.execute(select(TrainingMetric.__table__)).mappings())
    tool.migrate(engine,root,audit)
    tool.migrate(engine,root,audit,apply=True,confirm=lambda _: 'NO')
    assert not tool.pending_path(root).exists()
    tool.migrate(engine,root,audit,apply=True,confirm=lambda _: 'YES')
    with engine.connect() as c:
        changed=tool.run_rows(c)
        assert list(c.execute(select(Checkpoint.__table__)).mappings())==checkpoints
        assert list(c.execute(select(TrainingMetric.__table__)).mappings())==metrics
    for e in audit['mapping']:
        old=original[e['run_id']];new=changed[e['run_id']]
        assert {k:v for k,v in old.items() if k not in tool.FIELDS}=={k:v for k,v in new.items() if k not in tool.FIELDS}
        assert new['name']==e['new_name']
        assert tool.sha(new['raw_config_yaml'])==new['config_sha256']
        assert not (root/e['old_config_path']).exists()
        assert (root/e['new_config_path']).exists()
        assert (root/f'artifacts/runs/{e["run_id"]:08d}/checkpoints/model.zip').read_text()=='weights'
    tool.recover(engine,root)
    assert not tool.pending_path(root).exists()


@pytest.mark.parametrize('phase',['before_commit','after_commit'])
def test_recover_interrupted_migration(case,monkeypatch,phase):
    engine,root,audit=case
    original_apply=tool.apply_database;original_finish=tool.finish_files
    if phase=='before_commit':
        def crash(c,plan):
            original_apply(c,plan)
            raise RuntimeError('injected crash')
        monkeypatch.setattr(tool,'apply_database',crash)
    else:
        def crash(root,plan,side):
            f=next(f for f in plan['files'] if f['after'] is not None)
            tool.write_atomic(tool.safe(root,f['path']),f['after'])
            raise RuntimeError('injected crash')
        monkeypatch.setattr(tool,'finish_files',crash)
    with pytest.raises(RuntimeError,match='injected crash'):
        tool.migrate(engine,root,audit,apply=True,confirm=lambda _: 'YES')
    assert tool.pending_path(root).exists()
    monkeypatch.setattr(tool,'apply_database',original_apply);monkeypatch.setattr(tool,'finish_files',original_finish)
    tool.recover(engine,root)
    with engine.connect() as c:rows=tool.run_rows(c)
    for e in audit['mapping']:
        expected=e['old_name'] if phase=='before_commit' else e['new_name']
        assert rows[e['run_id']]['name']==expected
        assert (root/e['old_config_path']).exists()==(phase=='before_commit')
        assert (root/e['new_config_path']).exists()==(phase=='after_commit')
    assert not tool.pending_path(root).exists()


def test_refuse_config_changed_since_audit(case):
    engine,root,audit=case
    p=root/audit['mapping'][0]['old_config_path'];p.write_text(p.read_text()+'\n# external edit\n')
    with pytest.raises(ValueError,match='changed since audit'):
        tool.migrate(engine,root,audit,apply=True,confirm=lambda _: 'YES')
    assert not tool.pending_path(root).exists()


def test_pending_configs_follow_renamed_parent(case):
    engine, root, audit = case
    parent = audit['mapping'][-1]
    raw = (root/parent['old_config_path']).read_text()
    cfg = yaml.safe_load(raw)
    cfg['run']['name'] = 'pending_resume'
    cfg['continuation']['source_run'] = parent['old_name']
    raw = yaml.safe_dump(cfg, sort_keys=False)
    old_path = 'configs/stage_one/pending_resume.yml'
    (root/old_path).write_text(raw)
    configs = {e['old_name']: yaml.safe_load((root/e['old_config_path']).read_text()) for e in audit['mapping']}
    configs['pending_resume'] = cfg
    name = proposed_names(configs)['pending_resume']['new_name']
    audit['mapping'].append(dict(run_id=None, old_name='pending_resume', new_name=name,
        old_config_path=old_path, new_config_path=f'configs/stage_one/{name}.yml',
        old_source_run=parent['old_name'], disk_config_sha256=tool.sha(raw)))
    tool.migrate(engine, root, audit, apply=True, confirm=lambda _: 'YES')
    changed = yaml.safe_load((root/audit['mapping'][-1]['new_config_path']).read_text())
    assert changed['continuation']['source_run'] == parent['new_name']
    with engine.connect() as connection:
        assert len(tool.run_rows(connection)) == 2


def test_database_lr_mismatch_blocks_all_changes(case):
    engine, root, audit = case
    entry = audit['mapping'][0]
    with engine.begin() as connection:
        row = tool.run_rows(connection)[entry['run_id']]
        cfg = yaml.safe_load(row['raw_config_yaml'])
        cfg['ppo']['learning_rate'] *= 2
        raw = yaml.safe_dump(cfg)
        norm = normalize_config(RunConfig.model_validate(cfg))
        connection.execute(update(Run).where(Run.id == row['id']).values(
            raw_config_yaml=raw, normalized_config_json=json.loads(norm),
            config_sha256=tool.sha(raw), normalized_config_sha256=tool.sha(norm)))
    entry['config_sha256'] = tool.sha(raw)
    entry['normalized_config_sha256'] = tool.sha(norm)
    with pytest.raises(ValueError, match='Disk/database config mismatch'):
        tool.migrate(engine, root, audit, apply=True, confirm=lambda _: 'YES')
    assert (root/entry['old_config_path']).exists()
    assert not tool.pending_path(root).exists()
