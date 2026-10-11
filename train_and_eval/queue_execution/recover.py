"""Recover an owned abandoned attempt; retain its checkpoints and artifacts."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from sqlalchemy import text

from train_and_eval.database.session import create_database_engine
from train_and_eval.queue_execution.core import CONTROL, claim, mark_abandoned, try_lock


def validate_receipt(receipt, row, proc_root=Path('/proc')):
    if row['name'] != receipt['name'] or row['config_sha256'] != receipt['config_sha256']:
        raise ValueError('Receipt does not match persisted run identity')
    if row['created_at'] < datetime.fromisoformat(receipt['created_not_before']):
        raise ValueError('Run predates this attempt; preserving it')
    if row['status'] not in {'pending', 'running'}:
        raise ValueError('Only an abandoned pending/running attempt can be recovered')
    try:
        current = (proc_root / str(receipt['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return
    if receipt.get('pid_start_ticks') is None or current == receipt['pid_start_ticks']:
        raise ValueError('Worker PID still exists; preserving active work')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True, help='events.jsonl or a legacy attempt receipt')
    parser.add_argument('--attempt-id', help='Attempt identifier from events.jsonl')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    path = args.attempt.resolve(strict=True)
    storage = (root / 'artifacts').resolve()
    storage.relative_to(root)
    path.relative_to((storage / 'search_queues').resolve())
    if path.name == 'events.jsonl':
        if not args.attempt_id:
            raise ValueError('--attempt-id is required with events.jsonl')
        matches = []
        for line in path.read_text().splitlines():
            try: entry = json.loads(line)
            except ValueError: continue  # A crash can leave a partial final line.
            if entry.get('event') == 'attempt_registered' and entry.get('attempt') == args.attempt_id:
                matches.append(entry['receipt'])
        if len(matches) != 1:
            raise ValueError('Expected exactly one matching attempt registration')
        receipt = matches[0]
    elif path.name.endswith('.attempt.json'):
        receipt = json.loads(path.read_text())
    else:
        raise ValueError('Expected events.jsonl or legacy scheduler receipt')
    engine = create_database_engine()
    control = None
    try:
        control = engine.connect().execution_options(isolation_level='AUTOCOMMIT')
        if not try_lock(control, CONTROL, 1):
            raise RuntimeError('A scheduler is active; recovery did not change anything')
        with claim(engine, receipt['name'], 2) as connection:
            if connection is None:
                raise RuntimeError('Worker/name/capacity still claimed; recovery did not change anything')
            row = connection.execute(text('SELECT name, status, config_sha256, created_at FROM runs WHERE name = :name'),
                                     {'name': receipt['name']}).mappings().one()
            validate_receipt(receipt, row)
            mark_abandoned(connection, receipt['name'], 'Recovered abandoned owned queue attempt; artifacts retained')
            print(f'{receipt["name"]}: marked failed; checkpoints and artifacts retained. Restart uses the existing queue preview.')
    finally:
        if control is not None:
            control.invalidate()
            control.close()
        engine.dispose()


if __name__ == '__main__':
    main()
