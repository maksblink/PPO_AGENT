"""Verify a completed run replay against its reference: metrics and final model payloads."""
import argparse
import copy
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))


def normalized(value):
    value=json.loads(value) if isinstance(value,str) else copy.deepcopy(value)
    value['run'].pop('name',None)
    # Selector text may be pinned to an ID; actual checkpoint IDs are checked separately.
    value['continuation'].pop('checkpoint',None)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',type=int,required=True);args=p.parse_args()
    from sqlalchemy import text
    from train_and_eval.database.session import create_database_engine
    from train_and_eval.training.critic_experiment_plan import sha
    directory=ROOT/'artifacts/runs'/f'{args.run_id:08d}'
    plan=json.loads((directory/'critic_experiment.json').read_text())
    if plan['kind']!='run_replay' or plan['run_id']!=args.run_id:
        raise ValueError('A checkpoint continuation has no exact historical replay reference')
    engine=create_database_engine()
    try:
        with engine.connect() as c:
            c.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY'))
            runs=[];metrics=[];checkpoints=[]
            for rid in (plan['reference_run_id'],args.run_id):
                run=c.execute(text('SELECT to_jsonb(r) FROM runs r WHERE id=:id'),{'id':rid}).scalar_one()
                if run['status']!='completed':raise ValueError('Run not completed')
                runs.append(run)
                rows=c.execute(text("SELECT to_jsonb(m) - 'id' - 'run_id' - 'created_at' FROM training_metrics m WHERE run_id=:id ORDER BY run_step,id"),{'id':rid}).scalars().all()
                metrics.append(rows)
                ck=c.execute(text("SELECT relative_path,sha256 FROM checkpoints WHERE run_id=:id AND save_reason='final'"),{'id':rid}).mappings().one()
                checkpoints.append(ck)
            if runs[0]['source_checkpoint_id']!=runs[1]['source_checkpoint_id']:
                raise ValueError('Source checkpoint mismatch')
            if normalized(runs[0]['normalized_config_json'])!=normalized(runs[1]['normalized_config_json']):
                raise ValueError('Configuration mismatch')
            if not metrics[0] or metrics[0]!=metrics[1]:raise ValueError('Training metrics differ')
            print('PASS: saved configurations, source checkpoint and all training metric records match.')
            payloads=[]
            for ck in checkpoints:
                candidates={(ROOT/ck['relative_path']).resolve(),(ROOT/'artifacts'/ck['relative_path']).resolve()}
                found=[f for f in candidates if f.is_file()]
                if len(found)!=1 or sha(found[0])!=ck['sha256']:raise ValueError('Checkpoint integrity mismatch')
                with zipfile.ZipFile(found[0]) as z:
                    payloads.append({n:z.read(n) for n in ('policy.pth','policy.optimizer.pth')})
            if payloads[0]!=payloads[1]:raise ValueError('Final policy or optimizer payload differs')
            print('PASS: final policy and optimizer payloads are byte-identical.')
            print('Evaluations/trajectory files are not compared by this command.')
            c.rollback()
    finally:engine.dispose()
    print('Read-only. Nothing changed.')

if __name__=='__main__':main()
