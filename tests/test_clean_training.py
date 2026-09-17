from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from sqlalchemy import select, func, update, delete
from sqlalchemy.orm import Session
from train_and_eval.database.models import Base, Run, Checkpoint, Evaluation, TrainingMetric, WalkForwardStudy, WalkForwardCycle
from extra_tools import clean_training as tool
from tests.database_support import isolated_test_schema, resolve_test_database_url


@pytest.fixture
def cleanup_database():
    root = Path(__file__).resolve().parents[1]
    url = resolve_test_database_url(root)
    with isolated_test_schema(url) as (engine, scoped):
        Base.metadata.create_all(engine)
        yield engine


def seed(engine, root):
 with Session(engine) as s:
  def run(name, source=None, cycle=None):
   r=Run(name=name, continuation_mode='resume' if source else 'fresh', source_checkpoint_id=source,
     cycle_id=cycle, stage_role='candidate' if cycle else None, git_commit='a'*40,git_branch='master',config_schema_version=1,
     seed=1,config_sha256='a'*64,normalized_config_sha256='a'*64,raw_config_yaml='{}',normalized_config_json={},
     data_path='data/source.parquet',data_sha256='a'*64,duration_unit='data_epochs',duration_amount=1,
     split_index=10,train_rows=10,validation_rows=10,steps_per_data_epoch=10,training_steps_requested=10)
   s.add(r);s.flush()
   s.add(TrainingMetric(run_id=r.id,run_step=1,model_step=1,rollout_number=1))
   return r.id
  def cp(r):
   c=Checkpoint(run_id=r,run_step=0,model_step=0,save_reason='initial',relative_path=f'artifacts/runs/{r:08d}/checkpoints/source.zip',sha256='a'*64,size_bytes=1)
   s.add(c);s.flush();return c.id
  def evaluation(c):
   now=datetime.now(timezone.utc)
   e=Evaluation(checkpoint_id=c,trigger='final',policy_mode='deterministic_argmax',seed=1,data_scope='custom_range',
     data_path='data/source.parquet',data_sha256='a'*64,data_rows=20,evaluation_start_index=1,evaluation_end_index=10,
     evaluation_start_at=now,evaluation_end_at=now+timedelta(days=1),lookback_rows=1,steps_expected=9,git_commit='a'*40,git_branch='master')
   s.add(e);s.flush();return e.id
  first=run('first');first_cp=cp(first);first_ev=evaluation(first_cp)
  study=WalkForwardStudy(name='study',protocol_sha256='a'*64,protocol={'source_checkpoint_id':first_cp},plan={'stage_one_source':{'run_id':first,'ancestors':[{'run_id':first}]}},git_commit='a'*40,git_branch='master',status='paused')
  s.add(study);s.flush()
  cycle=WalkForwardCycle(study_id=study.id,number=1,plan={},status='pending',source_checkpoint_id=first_cp)
  s.add(cycle);s.flush()
  second=run('second',first_cp,cycle.id);second_cp=cp(second);ref=evaluation(first_cp);test=evaluation(second_cp)
  cycle.reference_evaluation_id=ref;cycle.test_evaluation_id=test;cycle.selected_checkpoint_id=second_cp
  other=run('other');cp(other)
  s.commit()
  ids=dict(first=first,second=second,other=other,first_cp=first_cp,first_ev=first_ev,ref=ref,study=study.id)
 for r in (first,second,other):
  p=root/f'artifacts/runs/{r:08d}/checkpoints/source.zip';p.parent.mkdir(parents=True);p.write_text('weights')
 for e in (first_ev,ref):
  p=root/f'artifacts/runs/{first:08d}/evaluations/{e:08d}/trajectory.parquet';p.parent.mkdir(parents=True);p.write_text('trajectory')
 p=root/f'artifacts/walk_forward/{study.id:08d}/report.html';p.parent.mkdir(parents=True);p.write_text('report')
 return ids


def count(engine, model):
 with engine.connect() as c:return c.scalar(select(func.count()).select_from(model))


def test_real_database_cleanup_and_recovery(tmp_path, monkeypatch, cleanup_database):
 engine=cleanup_database
 data=tmp_path/'data/source.parquet';data.parent.mkdir();data.write_text('market')
 ids=seed(engine,tmp_path)
 assert tool.cleanup(engine,tmp_path,'clean-all',dry_run=True) is False
 assert count(engine,Run)==3
 for answer in ('no', '', 'sure', 'Sure', ' SURE', 'SURE ', 'DELETE clean-all'):
  assert not tool.cleanup(engine,tmp_path,'clean-all',confirm=lambda _, answer=answer: answer)
  assert count(engine,Run)==3
 with pytest.raises(ValueError,match='depends|ancestry'):
  tool.cleanup(engine,tmp_path,'clean-first-stage')
 with engine.begin() as c:
  c.execute(update(WalkForwardStudy).values(status='running'))
 with pytest.raises(ValueError,match='Running'):
  tool.cleanup(engine,tmp_path,'clean-all')
 with engine.begin() as c:c.execute(update(WalkForwardStudy).values(status='paused'))
 with Session(engine) as session:
  shared=WalkForwardStudy(name='shared',protocol_sha256='b'*64,protocol={'source_checkpoint_id':ids['first_cp']},plan={},git_commit='a'*40,git_branch='master',status='paused')
  session.add(shared);session.flush();shared_id=shared.id
  session.add(WalkForwardCycle(study_id=shared_id,number=1,plan={},status='pending',reference_evaluation_id=ids['ref']))
  session.commit()
 assert tool.cleanup(engine,tmp_path,'clean-second-stage',run_ids=[ids['second']],confirm=lambda _: 'SURE')
 assert count(engine,Run)==2 and count(engine,WalkForwardStudy)==1
 with engine.connect() as c:
  assert c.scalar(select(Evaluation.id).where(Evaluation.id==ids['ref'])) == ids['ref']
  assert c.scalar(select(Evaluation.id).where(Evaluation.id==ids['first_ev']))==ids['first_ev']
 assert (tmp_path/f"artifacts/runs/{ids['first']:08d}/evaluations/{ids['ref']:08d}").exists()
 assert tool.cleanup(engine,tmp_path,'clean-second-stage',study_ids=[shared_id],confirm=lambda _: 'SURE')
 assert not (tmp_path/f"artifacts/runs/{ids['first']:08d}/evaluations/{ids['ref']:08d}").exists()
 assert (tmp_path/f"artifacts/runs/{ids['first']:08d}/checkpoints/source.zip").read_text()=='weights'
 assert tool.cleanup(engine,tmp_path,'clean-first-stage',run_ids=[ids['first']],confirm=lambda _: 'SURE')
 assert count(engine,Run)==1
 assert (tmp_path/f"artifacts/runs/{ids['other']:08d}/checkpoints/source.zip").exists()
 # Mixed database + orphan cleanup must restore both on rollback.
 mixed_orphan=tmp_path/'artifacts/runs/99999990'
 mixed_orphan.mkdir();(mixed_orphan/'marker').write_text('orphan')
 # Crash before commit: DB rolls back, files can be restored on the next call.
 original=tool.delete_rows
 def crash(*args):
  original(*args)
  raise RuntimeError('injected before commit')
 monkeypatch.setattr(tool,'delete_rows',crash)
 with pytest.raises(RuntimeError,match='injected'):
  tool.cleanup(engine,tmp_path,'clean-all',confirm=lambda _: 'SURE')
 assert count(engine,Run)==1 and tool.journal_path(tmp_path).exists()
 monkeypatch.setattr(tool,'delete_rows',original)
 answers=iter(['RECOVER','no'])
 assert not tool.cleanup(engine,tmp_path,'clean-all',confirm=lambda _: next(answers))
 assert not tool.journal_path(tmp_path).exists()
 assert (mixed_orphan/'marker').read_text()=='orphan'
 assert (tmp_path/f"artifacts/runs/{ids['other']:08d}/checkpoints/source.zip").exists()
 assert tool.cleanup(engine,tmp_path,'clean-all',confirm=lambda _: 'SURE')
 # Full deletion includes the circular run/checkpoint/cycle references.
 next_ids=seed(engine,tmp_path)
 assert next_ids['first'] > ids['other']  # ID sequences were not reset.
 # Crash after commit: retry must purge, never restore deleted experiments.
 finish=tool.finish_pending
 def crash_after(*args):raise RuntimeError('injected after commit')
 monkeypatch.setattr(tool,'finish_pending',crash_after)
 with pytest.raises(RuntimeError,match='after commit'):
  tool.cleanup(engine,tmp_path,'clean-all',confirm=lambda _: 'SURE')
 assert count(engine,Run)==0
 monkeypatch.setattr(tool,'finish_pending',finish)
 tool.cleanup(engine,tmp_path,'clean-all',confirm=lambda _: 'RECOVER')
 for model in (Run,Checkpoint,Evaluation,TrainingMetric,WalkForwardCycle,WalkForwardStudy):assert count(engine,model)==0
 assert not list((tmp_path/'artifacts/runs').iterdir())
 assert data.read_text()=='market'
 assert not tool.journal_path(tmp_path).exists()
 # Orphan-only cleanup is useful even when the database is entirely empty.
 orphans=[tmp_path/'artifacts/runs/99999991',tmp_path/'artifacts/walk_forward/99999992']
 for directory in orphans:
  directory.mkdir();(directory/'marker').write_text('orphan')
 preserved=[tmp_path/'artifacts/README.md',tmp_path/'artifacts/.gitignore',tmp_path/'artifacts/runs/notes.txt']
 for path in preserved:path.write_text('keep')
 with engine.connect() as connection:
  preview=tool.make_plan(connection,tmp_path,'clean-all')
 assert preview['run_ids']==preview['study_ids']==[]
 assert set(preview['orphan_paths'])=={str(path.relative_to(tmp_path)) for path in orphans}
 assert not tool.cleanup(engine,tmp_path,'clean-all',dry_run=True)
 assert not tool.cleanup(engine,tmp_path,'clean-all',confirm=lambda _: 'sure')
 for mode in ('clean-first-stage','clean-second-stage'):
  assert not tool.cleanup(engine,tmp_path,mode,confirm=lambda _: 'SURE')
 assert all(path.exists() for path in orphans)
 # Crash between renames: recovery must finish both moved and unmoved orphans.
 rename=Path.rename
 def interrupt_rename(path,target):
  if path==orphans[1]:raise RuntimeError('injected between renames')
  return rename(path,target)
 monkeypatch.setattr(Path,'rename',interrupt_rename)
 with pytest.raises(RuntimeError,match='between renames'):
  tool.cleanup(engine,tmp_path,'clean-all',confirm=lambda _: 'SURE')
 monkeypatch.setattr(Path,'rename',rename)
 assert not orphans[0].exists() and orphans[1].exists()
 with Session(engine) as session:
  session.add(WalkForwardStudy(id=99999992,name='new_owner',protocol_sha256='c'*64,
   protocol={},plan={},git_commit='a'*40,git_branch='master',status='paused'))
  session.commit()
 with pytest.raises(ValueError,match='acquired a database owner'):
  tool.cleanup(engine,tmp_path,'clean-all',confirm=lambda _: 'RECOVER')
 assert orphans[1].exists() and tool.journal_path(tmp_path).exists()
 assert list((tmp_path/'artifacts').glob('.cleanup-trash-*/**/marker'))
 with engine.begin() as connection:
  connection.execute(delete(WalkForwardStudy).where(WalkForwardStudy.id==99999992))
 tool.cleanup(engine,tmp_path,'clean-all',confirm=lambda _: 'RECOVER')
 assert all(not path.exists() for path in orphans)
 assert all(path.read_text()=='keep' for path in preserved)
 assert data.read_text()=='market'
 assert not tool.journal_path(tmp_path).exists()
 # Normal orphan-only deletion also completes without a recovery invocation.
 orphans[0].mkdir()
 assert tool.cleanup(engine,tmp_path,'clean-all',confirm=lambda _: 'SURE')
 assert not orphans[0].exists()
 engine.dispose()


@pytest.mark.parametrize('path',['../data','data/file','artifacts','/tmp/file','artifacts/../data/file'])
def test_paths_cannot_escape_artifacts(tmp_path,path):
 with pytest.raises(ValueError):tool.safe_path(tmp_path,path)


def test_symlink_is_rejected(tmp_path):
 (tmp_path/'artifacts').symlink_to(tmp_path/'data')
 with pytest.raises(ValueError,match='symlink'):tool.safe_path(tmp_path,'artifacts/runs/00000001')


def test_orphan_scan_uses_canonical_ids_and_preserves_other_entries(tmp_path):
 for category in ('runs','walk_forward'):
  parent=tmp_path/'artifacts'/category
  parent.mkdir(parents=True)
  for name in ('00000001','00000002','00000000','000000001','notes','7'):
   (parent/name).mkdir()
  (parent/'00000003').write_text('not a directory')
 assert tool.orphan_directories(tmp_path,{1},{2}) == [
  'artifacts/runs/00000002','artifacts/walk_forward/00000001']


@pytest.mark.parametrize('parent_link',[False,True])
def test_orphan_scan_rejects_symlinks(tmp_path,parent_link):
 target=tmp_path/'external';target.mkdir()
 parent=tmp_path/'artifacts/runs';parent.parent.mkdir()
 if parent_link:parent.symlink_to(target)
 else:
  parent.mkdir();(parent/'00000001').symlink_to(target)
 with pytest.raises(ValueError,match='symlink'):
  tool.orphan_directories(tmp_path,[],[])


@pytest.mark.parametrize('pending',[False,True])
def test_cli_only_mentions_recovery_when_journal_exists(tmp_path,monkeypatch,capsys,pending):
 from types import SimpleNamespace
 import sys
 monkeypatch.setattr(sys,'argv',['clean_training.py','clean-all','--project-root',str(tmp_path)])
 monkeypatch.setattr(tool,'get_database_url',lambda **kwargs:'unused')
 monkeypatch.setattr(tool,'create_database_engine',lambda *_:SimpleNamespace(dispose=lambda:None))
 def fail(*args):raise ValueError('missing study')
 monkeypatch.setattr(tool,'cleanup',fail)
 if pending:
  path=tmp_path/'artifacts/.cleanup_pending.json';path.parent.mkdir();path.write_text('{}')
 assert tool.main()==1
 error=capsys.readouterr().err
 assert 'missing study' in error
 assert ('Rerun this tool to recover' in error)==pending
