"""Prepare an explicit diagnostic replay of a run, or continuation of a checkpoint."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def select_steps(records, explicit, top, neighbors):
    available = sorted({r['run_step'] for r in records})
    if top and not records: raise ValueError('No metrics available for automatic selection')
    ranked = []
    for r in records:
        vals = [r.get(k) for k in ('value_error_std', 'value_error_mean', 'post_train_value_mse')]
        if any(v is None or not math.isfinite(v) for v in vals): continue
        pre = vals[0]**2 + vals[1]**2
        if pre > 0: ranked.append((vals[2]/pre, r['run_step']))
    centers = [s for _, s in sorted(ranked, reverse=True)[:top]]
    selected = set(explicit) | set(centers)
    for center in centers:
        i = available.index(center)
        selected.update(available[max(0,i-neighbors):i+neighbors+1])
    if not selected or any(type(s) is not int or s <= 0 for s in selected):
        raise ValueError('Select positive local steps using --steps or --top-mse-increases')
    if records and not selected <= set(available):
        raise ValueError('Selected step has no saved metric record in the source run')
    return sorted(selected), centers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-id', type=int)
    source.add_argument('--checkpoint-id', type=int)
    p.add_argument('--config', help='Required for checkpoint mode: explicit continuation duration and settings')
    p.add_argument('--name', required=True, help='New unique run and queue name')
    p.add_argument('--steps', type=int, nargs='+', default=[])
    p.add_argument('--top-mse-increases', type=int, default=0)
    p.add_argument('--neighbors', type=int, default=0, help='Neighbor records around automatically selected steps')
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    if args.top_mse_increases < 0 or args.neighbors < 0: p.error('Counts cannot be negative')
    if args.checkpoint_id is not None and (not args.config or args.top_mse_increases):
        p.error('Checkpoint mode requires --config and explicit --steps')
    if args.run_id is not None and args.config: p.error('Run replay takes its saved configuration')
    import yaml
    from sqlalchemy import text
    from train_and_eval.database.session import create_database_engine
    from train_and_eval.run_config import RunConfig, validate_resume_compatibility
    from train_and_eval.training.critic_experiment_plan import sha, sidecar
    from extra_tools.maintenance.maintenance_guard import activity_guard, maintenance_guard
    def decoded(value): return json.loads(value) if isinstance(value, str) else copy.deepcopy(value)
    # Preparation never mutates the DB, and excludes active managed queues/maintenance.
    with activity_guard(ROOT, exclusive=True), maintenance_guard(ROOT, exclude_managed=True):
        engine = create_database_engine()
        try:
            with engine.connect() as c:
                c.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY'))
                if c.execute(text('SELECT count(*) FROM runs WHERE name=:name'), {'name':args.name}).scalar_one():
                    raise ValueError('Run name already exists; choose a new experiment name')
                records = []; cp_id = args.checkpoint_id
                if args.run_id is not None:
                    run = c.execute(text('SELECT to_jsonb(r) FROM runs r WHERE id=:id'), {'id':args.run_id}).scalar_one()
                    if run['status'] != 'completed' or run.get('cycle_id') is not None or run.get('stage_role') is not None:
                        raise ValueError('Replay requires a completed standalone stage-one run')
                    cfg = decoded(run['normalized_config_json']); cp_id = run['source_checkpoint_id']
                    if cfg['run']['name'] != run['name']: raise ValueError('Source identity mismatch')
                    if (cfg['continuation']['mode']=='fresh') != (cp_id is None): raise ValueError('Source ancestry mismatch')
                    records = c.execute(text('''SELECT run_step,value_error_std,value_error_mean,post_train_value_mse
                        FROM training_metrics WHERE run_id=:id ORDER BY run_step'''), {'id':args.run_id}).mappings().all()
                else:
                    cfg = yaml.safe_load(Path(args.config).read_text())
                cfg['run']['name'] = args.name
                checkpoint_hash = None
                if cp_id is not None:
                    cp = c.execute(text('''SELECT ck.*,r.name AS source_name,r.normalized_config_json AS source_config
                        FROM checkpoints ck JOIN runs r ON r.id=ck.run_id WHERE ck.id=:id'''), {'id':cp_id}).mappings().one()
                    if args.run_id is not None and cfg['continuation']['source_run'] != cp['source_name']:
                        raise ValueError('Saved parent name differs from checkpoint owner')
                    cfg['continuation'] = {'mode':'resume','source_run':cp['source_name'],'checkpoint':str(cp_id)}
                    candidates = {(ROOT/cp['relative_path']).resolve(),(ROOT/'artifacts'/cp['relative_path']).resolve()}
                    found = [f for f in candidates if f.is_file()]
                    if len(found)!=1: raise ValueError('Source checkpoint is missing or ambiguous')
                    checkpoint_hash = sha(found[0])
                    if checkpoint_hash != cp['sha256']: raise ValueError('Source checkpoint hash differs from database')
                    validate_resume_compatibility(RunConfig.model_validate(cfg),RunConfig.model_validate(decoded(cp['source_config'])))
                RunConfig.model_validate(cfg)
                steps, centers = select_steps(records,args.steps,args.top_mse_increases,args.neighbors)
                c.rollback()
        finally: engine.dispose()
        config_path = ROOT/'configs/diagnostics/critic'/f'{args.name}.yml'
        queue_path = ROOT/'configs/search_queues/diagnostics/critic'/f'{args.name}.yml'
        config_text = yaml.safe_dump(cfg,sort_keys=False)
        import hashlib
        plan = {'format':1,'name':args.name,'kind':'run_replay' if args.run_id is not None else 'checkpoint_continuation',
                'reference_run_id':args.run_id,'source_checkpoint_id':cp_id,'source_checkpoint_sha256':checkpoint_hash,
                'config_sha256':hashlib.sha256(config_text.encode()).hexdigest(),
                'capture_steps':steps,'detail_steps':steps,'selection':{'top_mse_centers':centers,'neighbors':args.neighbors}}
        files = {config_path:config_text, sidecar(config_path):json.dumps(plan,indent=2)+'\n',
                 queue_path:yaml.safe_dump({'queue_schema_version':1,'name':args.name,
                                            'configs':[config_path.relative_to(ROOT).as_posix()]},sort_keys=False)}
        # Duplicate queue stems are forbidden by the scheduler, including outside this folder.
        for f in (ROOT/'configs/search_queues').rglob('*'):
            if f.suffix in {'.yml','.yaml'} and f.stem==args.name and f != queue_path:
                raise ValueError(f'Queue name collision: {f}')
        for f, content in files.items():
            if f.is_symlink() or (f.exists() and f.read_text()!=content): raise ValueError(f'Refusing overwrite: {f}')
        print(json.dumps(plan,indent=2))
        for f, content in files.items():
            print(('READY: ' if args.apply else 'PREVIEW: ')+str(f.relative_to(ROOT)))
            if args.apply and not f.exists():
                f.parent.mkdir(parents=True,exist_ok=True)
                with f.open('x') as stream: stream.write(content)
        print('No database changes or training. Use the managed queue after reviewing this plan.')


if __name__ == '__main__': main()
