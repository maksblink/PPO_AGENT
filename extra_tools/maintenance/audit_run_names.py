#!/usr/bin/env python3
"""Read-only run rename preflight. Writes only a new local audit directory."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


LR_DECIMAL_PLACES = 9


def learning_rate_token(value) -> str:
    rate = Decimal(str(value))
    if not rate.is_finite() or not 0 < rate < 1:
        raise ValueError(f"LR must be finite and between 0 and 1: {value}")
    rendered = format(rate, f'.{LR_DECIMAL_PLACES}f')
    if Decimal(rendered) != rate:
        raise ValueError(f"LR requires more than {LR_DECIMAL_PLACES} decimal places: {value}")
    return rendered.replace('.', 'p')


def proposed_names(configs: dict) -> dict:
    result, visiting = {}, set()

    def visit(old):
        if old in result:
            return result[old]
        if old in visiting:
            raise ValueError(f"Continuation cycle at {old}")
        if old not in configs:
            raise ValueError(f"Source run has no stage-one config: {old}")
        visiting.add(old)
        c = configs[old]
        continuation = c['continuation']
        mode = continuation['mode']
        if mode == 'fresh':
            phase = 0
        elif mode == 'resume':
            source = continuation['source_run']
            parent = visit(source)
            phase = parent['phase'] + 1
            if continuation['checkpoint'] != 'final':
                raise ValueError(f"Non-final continuation: {old}")
            for field in ('seed',):
                if c['run'][field] != configs[source]['run'][field]:
                    raise ValueError(f"Different {field}: {old} <- {source}")
            for field in ('hidden_sizes', 'learning_rate', 'gamma', 'n_epochs'):
                if c['ppo'][field] != configs[source]['ppo'][field]:
                    raise ValueError(f"Different {field}: {old} <- {source}")
        else:
            raise ValueError(f"Unsupported continuation mode: {mode}")
        widths = c['ppo']['hidden_sizes']
        if not widths or len(set(widths)) != 1:
            raise ValueError(f"Non-uniform hidden layers: {old}")
        gamma = Decimal(str(c['ppo']['gamma'])) * 100
        if gamma != gamma.to_integral_value():
            raise ValueError(f"Gamma cannot be represented by gNNN: {old}")
        learning_rate = learning_rate_token(c['ppo']['learning_rate'])
        prefix = 'f' if mode == 'fresh' else 'r'
        name = (f"{prefix}_v{phase}_w{widths[0]}x{len(widths)}_g{int(gamma):03d}"
                f"_lr{learning_rate}_s{c['run']['seed']}_ne{c['ppo']['n_epochs']}")
        result[old] = {'new_name': name, 'phase': phase}
        visiting.remove(old)
        return result[old]

    for name in configs:
        visit(name)
    values = [r['new_name'] for r in result.values()]
    if len(values) != len(set(values)):
        raise ValueError('Proposed names are not unique')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, help='New audit directory (must not exist)')
    args = parser.parse_args()
    root = args.project.expanduser().resolve()
    if not (root / 'train_and_eval/run_config.py').is_file():
        parser.error('Project root does not contain train_and_eval/run_config.py')
    sys.path.insert(0, str(root))
    from sqlalchemy import select, text
    from train_and_eval.run_config import load_run_config, normalize_config, RunConfig
    from train_and_eval.database.models import Run, Checkpoint, Evaluation, WalkForwardStudy, WalkForwardCycle
    from train_and_eval.database.session import create_database_engine

    errors, configs, files, loaded = [], {}, {}, {}
    for path in sorted((root / 'configs/stage_one').rglob('*')):
        if path.suffix not in ('.yml', '.yaml'):
            continue
        if path.is_symlink():
            raise ValueError(f'Symlink config: {path}')
        obj = load_run_config(path, verify_data=False)
        name = obj.config.run.name
        if name in configs:
            raise ValueError(f'Duplicate config name: {name}')
        configs[name] = json.loads(obj.normalized_json)
        files[name] = path.relative_to(root).as_posix()
        loaded[name] = obj
    if not configs:
        raise ValueError('No stage-one configs found')
    names = proposed_names(configs)
    engine = create_database_engine()
    try:
        with engine.connect() as connection:
            with connection.begin():
                connection.execute(text('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY'))
                database = connection.scalar(text('SELECT current_database()'))
                records = list(connection.execute(select(Run.__table__)).mappings())
                checkpoints = list(connection.execute(select(Checkpoint.__table__)).mappings())
                evaluations = list(connection.execute(select(Evaluation.__table__)).mappings())
                studies = list(connection.execute(select(WalkForwardStudy.__table__)).mappings())
                cycles = list(connection.execute(select(WalkForwardCycle.__table__)).mappings())
    finally:
        engine.dispose()
    by_name = {r['name']: r for r in records}
    by_id = {r['id']: r for r in records}
    cp_by_id = {c['id']: c for c in checkpoints}
    if len(by_name) != len(records):
        errors.append('Duplicate database run names')
    outside = [{'id':r['id'],'name':r['name']} for r in records if r['name'] not in names]
    if outside:
        errors.append(f'{len(outside)} database runs have no stage-one config; review dependents')
    if studies or cycles:
        errors.append('Walk-forward records exist; their embedded names and config hashes require review')
    mapping=[]
    for old, plan in names.items():
        source = configs[old]['continuation'].get('source_run')
        path=Path(files[old]); new_path=path.with_name(plan['new_name']+path.suffix).as_posix()
        r=by_name.get(old)
        entry={'run_id':r['id'] if r else None,'old_name':old,'new_name':plan['new_name'],
               'old_config_path':files[old],'new_config_path':new_path,
               'old_source_run':source,'new_source_run':names[source]['new_name'] if source else None,
               'phase':plan['phase'],'learning_rate':format(Decimal(str(configs[old]['ppo']['learning_rate'])), 'f'),
               'disk_config_sha256':loaded[old].sha256}
        if new_path != files[old] and (root/new_path).exists():
            errors.append(f'Target config already exists: {new_path}')
        collision=by_name.get(plan['new_name'])
        if collision and collision is not r:
            errors.append(f'Target run name already exists: {plan["new_name"]}')
        if r is None:
            entry['status'] = 'not_started'
        else:
            status=getattr(r['status'],'value',r['status'])
            entry.update(status=status,source_checkpoint_id=r['source_checkpoint_id'],
                         config_sha256=r['config_sha256'],normalized_config_sha256=r['normalized_config_sha256'])
            if Decimal(str(r['normalized_config_json']['ppo']['learning_rate'])) != Decimal(entry['learning_rate']):
                errors.append(f'LR differs between database and config: #{r["id"]}')
            if status != 'completed':errors.append(f'Run #{r["id"]} is {status}')
            if r['normalized_config_json'] != configs[old]:errors.append(f'Disk/database config mismatch: #{r["id"]}')
            if digest(r['raw_config_yaml'].encode()) != r['config_sha256']:errors.append(f'Raw config hash mismatch: #{r["id"]}')
            normalized=normalize_config(RunConfig.model_validate(r['normalized_config_json']))
            if digest(normalized.encode()) != r['normalized_config_sha256']:errors.append(f'Normalized config hash mismatch: #{r["id"]}')
            import yaml
            raw_normalized=normalize_config(RunConfig.model_validate(yaml.safe_load(r['raw_config_yaml'])))
            if raw_normalized != normalized:errors.append(f'Raw/normalized config mismatch: #{r["id"]}')
            if source:
                cp=cp_by_id.get(r['source_checkpoint_id']);parent=by_id.get(cp['run_id']) if cp else None
                if not parent or parent['name']!=source or getattr(cp['save_reason'],'value',cp['save_reason'])!='final':
                    errors.append(f'Actual predecessor is not expected final: #{r["id"]}')
            elif r['source_checkpoint_id'] is not None:errors.append(f'Fresh run has source checkpoint: #{r["id"]}')
        mapping.append(entry)
    metadata_files=[];checkpoint_problems=[]
    for cp in checkpoints:
        rel=Path(cp['relative_path'])
        if rel.is_absolute() or '..' in rel.parts:
            checkpoint_problems.append({'id':cp['id'],'problem':'unsafe relative_path'});continue
        path=root/rel
        if not path.is_file():checkpoint_problems.append({'id':cp['id'],'problem':'missing'})
        elif path.stat().st_size != cp['size_bytes']:checkpoint_problems.append({'id':cp['id'],'problem':'size mismatch'})
    if checkpoint_problems:errors.append(f'{len(checkpoint_problems)} checkpoint path/size problems')
    pattern=re.compile('|'.join(re.escape(k) for k in sorted(names,key=len,reverse=True)))
    for dirname in ['artifacts','configs','extra_tools']:
        for path in sorted((root/dirname).rglob('*')):
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {'.json','.yaml','.yml','.csv','.html','.txt','.md'}:
                continue
            rel=path.relative_to(root).as_posix()
            if rel.startswith('extra_tools/maintenance/run_rename/'):
                continue
            if path.stat().st_size>32*1024*1024:
                metadata_files.append({'path':rel,'scan':'skipped_over_32MiB'});continue
            raw=path.read_bytes()
            try: content=raw.decode('utf-8')
            except UnicodeDecodeError:
                metadata_files.append({'path':rel,'scan':'not_utf8'});continue
            hits=sorted(set(pattern.findall(content)))
            if hits:metadata_files.append({'path':rel,'sha256':digest(raw),'run_name_matches':hits})
    git=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True)
    report={'audit_version':2,'lr_decimal_places':LR_DECIMAL_PLACES,'created_at':datetime.now(timezone.utc).isoformat(),'project':str(root),
            'git_commit':git.stdout.strip() if git.returncode==0 else None,'database':database,
            'counts':{'configs':len(configs),'runs':len(records),'checkpoints':len(checkpoints),
                      'evaluations':len(evaluations),'studies':len(studies),'cycles':len(cycles)},
            'run_statuses':dict(Counter(getattr(r['status'],'value',r['status']) for r in records)),
            'mapping':mapping,'outside_runs':outside,'errors':errors,'checkpoint_problems':checkpoint_problems,
            'metadata_references':metadata_files,
            'notes':['Database transaction was read-only; no configs or artifacts changed.',
                     'Checkpoint existence and byte sizes checked; contents were not rehashed.',
                     'Text references are inventory candidates, not an automatic replacement plan.',
                     'Training must remain stopped for migration; this audit does not lock future writers.']}
    target = (args.output.expanduser().resolve() if args.output else
              root/'extra_tools/maintenance/run_rename'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    target.mkdir(parents=True,exist_ok=False)
    (target/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (target/'mapping.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['run_id','old_name','new_name','old_config_path','new_config_path','old_source_run','new_source_run','status','learning_rate'])
        writer.writeheader();writer.writerows({k:r.get(k) for k in writer.fieldnames} for r in mapping)
    print(json.dumps(report['counts'],indent=2));print(f'Issues requiring review: {len(errors)}')
    for error in errors:print('REVIEW:',error)
    print(f'Audit: {target / "audit.json"}');print(f'Mapping: {target / "mapping.csv"}')
    print('No run, config, checkpoint, evaluation or existing artifact was modified.')
    return 1 if errors else 0


if __name__=='__main__':
    raise SystemExit(main())
