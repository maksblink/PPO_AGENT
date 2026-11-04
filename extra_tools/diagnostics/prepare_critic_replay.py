"""Prepare one diagnostic replay; read-only DB, explicit --apply for config writes."""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml
from sqlalchemy import text
from stable_baselines3 import PPO
from train_and_eval.database.session import create_database_engine
from train_and_eval.training.critic_diagnostics import EXPECTED_TRAIN_HASH, REPLAY_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--capture-multi', action='store_true', help='Capture 16 preselected rollouts in a new replay')
    modes.add_argument('--capture', action='store_true', help='Prepare a new replay capturing the rollout at step 404480')
    args = parser.parse_args()
    replay_name = 'diagnostic_capture_3732' if args.capture else REPLAY_NAME
    queue_name = 'critic_capture_3732' if args.capture else 'critic_repeat_3732'
    if args.capture_multi:
        replay_name = 'diagnostic_capture_multi_3732'
        queue_name = 'critic_capture_multi_3732'
    if hashlib.sha256(inspect.getsource(PPO.train).encode()).hexdigest() != EXPECTED_TRAIN_HASH:
        raise SystemExit('STOP: PPO.train differs from reviewed source')
    engine = create_database_engine()
    try:
        with engine.connect() as c:
            c.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY'))
            run = c.execute(text('SELECT to_jsonb(r) FROM runs r WHERE id=3732')).scalar_one()
            parent = c.execute(text('SELECT c.run_id, c.relative_path, c.save_reason, r.name, r.status FROM checkpoints c JOIN runs r ON r.id=c.run_id WHERE c.id=:id'), {'id':run['source_checkpoint_id']}).mappings().one()
            if run['status'] != 'completed' or parent['status'] != 'completed' or parent['run_id'] != 3730 or parent['save_reason'] != 'final':
                raise SystemExit('STOP: original/parent identity or status mismatch')
            if c.execute(text('SELECT count(*) FROM runs WHERE name=:name'), {'name':replay_name}).scalar_one():
                raise SystemExit('STOP: diagnostic run already exists; inspect it before retrying')
            cfg = run['normalized_config_json']
            cfg = json.loads(cfg) if isinstance(cfg, str) else copy.deepcopy(cfg)
            if cfg['run']['name'] != run['name'] or cfg['continuation']['source_run'] != parent['name']:
                raise SystemExit('STOP: saved config/ancestry mismatch')
            ppo=cfg['ppo']
            expected={'device':'cuda','hidden_sizes':[768]*4,'n_epochs':9,'learning_rate':0.00004,'gamma':0.9,'gae_lambda':0.95,'ent_coef':0.0002}
            if any(ppo.get(k)!=v for k,v in expected.items()) or cfg['evaluation']['device']!='cpu':
                raise SystemExit('STOP: original scientific configuration differs from expected run')
            candidates={ (ROOT/parent['relative_path']).resolve(), (ROOT/'artifacts'/parent['relative_path']).resolve() }
            found=[p for p in candidates if p.is_file()]
            if len(found)!=1:
                raise SystemExit('STOP: parent checkpoint not uniquely located')
            digest=hashlib.sha256(found[0].read_bytes()).hexdigest()
            c.rollback()
    finally:
        engine.dispose()
    cfg['run']['name']=replay_name
    from train_and_eval.run_config import RunConfig
    RunConfig.model_validate(cfg)
    config=Path(f'configs/diagnostics/{queue_name}.yml')
    queue=Path(f'configs/search_queues/diagnostics/{queue_name}.yml')
    files={config:yaml.safe_dump(cfg,sort_keys=False),queue:yaml.safe_dump({'queue_schema_version':1,'name':queue_name,'configs':[str(config)]},sort_keys=False)}
    for relative, content in files.items():
        path=ROOT/relative
        if path.exists() and path.read_text()!=content:
            raise SystemExit(f'STOP: refusing to overwrite differing file: {relative}')
    print(f'Original: #3732 {run["name"]}')
    print(f'Parent: #3730; checkpoint #{run["source_checkpoint_id"]}; SHA256={digest}')
    print(f'Replay: {replay_name}; only run.name changes')
    for relative, content in files.items():
        print(('READY: ' if args.apply else 'PREVIEW: ')+str(relative))
        if args.apply:
            path=ROOT/relative;path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                with path.open('x') as f:f.write(content)
    print('Database/checkpoints unchanged. No training started.')
    if not args.apply: print('Use --apply to create config and queue.')

if __name__=='__main__': main()
