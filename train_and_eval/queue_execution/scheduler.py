"""Bounded subprocess scheduling with database claims and thermal admission."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from sqlalchemy import text
from train_and_eval.database.session import create_database_engine
from train_and_eval.queue_execution.core import (
    CONTROL, RUN, claim, cpu_temperature, lock_available, mark_abandoned,
    occupied_slots, plan_jobs, run_key, runnable, ThermalGate, try_lock,
)
from train_and_eval.queue_execution.telemetry import Recorder, temperatures
from train_and_eval.queue_execution.affinity import Allocator, discover_groups


def other_local_queue(root):
    """Old queue versions do not understand claims: refuse to overlap them."""
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if int(path.parent.name) == os.getpid():
            continue
        try:
            arguments = path.read_bytes().split(b'\0')
            cwd = (path.parent / 'cwd').resolve()
        except OSError:
            continue
        if cwd != root.resolve():
            continue
        if any(Path(os.fsdecode(arg)).name == 'run_search_queue.py' for arg in arguments):
            return int(path.parent.name)
    return None


def database_states(connection, names):
    if not names:
        return {}
    rows = connection.execute(text('SELECT name, status FROM runs WHERE name = ANY(:names)'),
                              {'names': list(names)})
    return dict(rows.tuples().all())


@dataclass
class Active:
    job: object
    process: object
    log: object
    started: float
    attempt: str
    receipt: dict | None = None


def reconcile_attempt(engine, item, reason):
    if item.receipt is None:
        return
    from train_and_eval.queue_execution.recover import validate_receipt
    with claim(engine, item.job.name, 2) as owned:
        if owned is None:
            raise RuntimeError('Attempt still claimed; no status was changed')
        row = owned.execute(text('SELECT name, status, config_sha256, created_at FROM runs WHERE name = :name'),
                            {'name': item.job.name}).mappings().one_or_none()
        if row is None or row['status'] not in {'pending', 'running'}:
            return
        validate_receipt(item.receipt, row)
        mark_abandoned(owned, item.job.name, reason)


def stop_children(active, timeout=30):
    for item in active.values():
        if item.process.poll() is None:
            try: os.killpg(item.process.pid, signal.SIGINT)
            except ProcessLookupError: pass
    deadline = time.monotonic() + timeout
    while any(item.process.poll() is None for item in active.values()) and time.monotonic() < deadline:
        time.sleep(.2)
    for item in active.values():
        if item.process.poll() is None:
            try: os.killpg(item.process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        item.process.wait()


def run(selected, args, root):
    jobs = plan_jobs(selected)
    import yaml
    for job in jobs:
        cfg = yaml.safe_load(job.path.read_text())
        if cfg.get('ppo', {}).get('device') != 'cuda' or cfg.get('evaluation', {}).get('device') != 'cpu':
            raise ValueError(f'{job.name}: queue requires explicit training CUDA and evaluation CPU')
    gate = ThermalGate(args.temperature_limit, args.temperature_resume)
    legacy = other_local_queue(root)
    if legacy is not None:
        raise RuntimeError(f'Another queue process is still active: PID {legacy}. No training started.')
    engine = create_database_engine()
    control = None
    recorder = None
    events = None
    groups = discover_groups() if getattr(args, "cpu_affinity", "none") == "split" else []
    allocator = Allocator(groups) if groups else None
    active = {}
    records = []
    interrupted = False
    thermal_abort = False
    control_cpu_max_c = None
    failed = False
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    def interrupt(signum, frame):
        raise KeyboardInterrupt('Scheduler received SIGTERM')
    signal.signal(signal.SIGTERM, interrupt)
    started = time.monotonic()
    artifacts_directory = 'artifacts'
    directory = root / artifacts_directory / 'search_queues' / args.queue / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    try:
        control = engine.connect().execution_options(isolation_level='AUTOCOMMIT')
        control.execute(text("SET statement_timeout = '5s'"))
        if not try_lock(control, CONTROL, 1):
            raise RuntimeError('Another scheduler owns this PostgreSQL database')
        # Active pre-upgrade runs have no claim. Never reinterpret or cancel them.
        running = control.execute(text("SELECT name FROM runs WHERE status = 'running'")).scalars().all()
        for name in running:
            if lock_available(control, RUN, run_key(name)):
                raise RuntimeError(f'Unclaimed active/stale run {name}; preserving it. Resolve it before scheduling.')
        names = {job.name for job in jobs} | {job.source for job in jobs if job.source}
        states = database_states(control, names)
        for job in jobs:
            if job.source and job.source not in {j.name for j in jobs} and states.get(job.source) != 'completed':
                raise RuntimeError(f'External source {job.source} must already be completed')
        # Keep the existing explicit preview/confirmation for failed or cancelled
        # runs, before starting ANY child. Never automatically delete artifacts.
        skipped = set()
        for position, config in selected:
            state = states.get(config.name)
            if state in {'pending', 'failed', 'cancelled'}:
                if not lock_available(control, RUN, run_key(config.name)):
                    continue
                from run_search_queue import handle_incomplete_run
                run_id = control.scalar(text('SELECT id FROM runs WHERE name = :name'), {'name': config.name})
                decision = handle_incomplete_run(config, run_id)
                if decision == 'STOP':
                    return 0
                if decision == 'SKIP':
                    skipped.add(config.name)
        directory.mkdir(parents=True)
        recorder = Recorder(directory / 'telemetry.csv', groups, args.flush_seconds)
        events = (directory/'events.jsonl').open('x', buffering=1)
        def event(kind, **fields):
            events.write(json.dumps(dict(event=kind, timestamp=time.time(), **fields))+'\n')
        event('queue_start', groups=groups)

        (directory / 'plan.json').write_text(json.dumps({'workers': args.workers,
            'serial_evaluation': getattr(args, 'serial_evaluation', False),
            'threads': args.threads, 'temperature_limit': gate.limit, 'temperature_resume': gate.resume,
            'abort_temperature': args.abort_temperature,
            'jobs': [{'name': j.name, 'source': j.source, 'sha256': j.sha256,
                      'position': j.position} for j in jobs]}, indent=2))
        pending = [job for job in jobs if states.get(job.name) != 'completed' and job.name not in skipped]
        next_sample = 0
        next_control = 0
        sensors = {}
        temperature = None
        phase_cache = {}
        last_message = None
        last_message_at = 0
        while pending or active:
            now = time.monotonic()
            states = database_states(control, names)
            # Validate actual database completion; exit code alone is insufficient.
            for name, item in list(active.items()):
                code = item.process.poll()
                if code is None:
                    continue
                item.log.close()
                event('worker_exit', name=name, pid=item.process.pid, code=code)
                del active[name]
                elapsed = now - item.started
                if code in (75, 76):
                    pending.append(item.job)
                    continue
                ok = code == 0 and states.get(name) == 'completed'
                if not ok:
                    reconcile_attempt(engine, item, f'Queue child PID {item.process.pid} exited {code}')
                    failed = True
                records.append({'name': name, 'attempt': item.attempt, 'pid': item.process.pid,
                    'wall_seconds': elapsed, 'exit_code': code, 'completed': ok})
                print(f'{name}: {"completed" if ok else "failed"}, {elapsed:.1f}s', flush=True)
            if now >= next_control:
                sensors = temperatures()
                temperature = max(sensors.values()) if sensors else None
                next_control = now + args.control_seconds
            if temperature is not None:
                control_cpu_max_c = max(control_cpu_max_c or temperature, temperature)
            if active and temperature is not None and temperature >= args.abort_temperature:
                thermal_abort = True
                interrupted = True
                event('thermal_abort', cpu_c=temperature)
                raise RuntimeError(f'CPU reached {temperature:.1f} C; interrupting owned workers')
            if active and temperature is None:
                raise RuntimeError('CPU sensor disappeared; thermal supervision unavailable')
            admit = gate.admit(temperature)
            message = None if admit else 'Thermal admission paused'
            if message != last_message:
                event('admission_paused' if message else 'admission_resumed', cpu_c=temperature)
                print(f'{message or "Thermal admission resumed"}: CPU={temperature}', flush=True)
            last_message = message
            if now >= next_sample:
                workers=[]
                for name,item in active.items():
                    try: state=json.loads(item.phase_path.read_text())
                    except (OSError,ValueError): state={'phase':'starting'}
                    # Phase files are tiny atomically replaced snapshots, never growing logs.
                    phase=state.get('phase','unknown')
                    if phase_cache.get(name)!=phase:
                        event('worker_phase',name=name,phase=phase)
                        phase_cache[name]=phase
                    workers.append({'name':name,'pid':item.process.pid,'phase':phase,'cpus':item.cpus})
                recorder.record(workers,sensors)
                next_sample = now + args.sample_seconds
            recorder.flush_due()
            # On first failure drain already-running work and start nothing else.
            if failed:
                if not active:
                    break
                time.sleep(.5)
                continue
            pending = [job for job in pending if states.get(job.name) != 'completed']
            launched = False
            while admit and len(active) < args.workers:
                # Existing orphan workers retain global capacity and name claims.
                if occupied_slots(control) >= args.workers:
                    break
                candidate = next((job for job in pending if job.name not in active
                    and states.get(job.name) is None and runnable(job, states)
                    and lock_available(control, RUN, run_key(job.name))), None)
                if candidate is None:
                    break
                pending.remove(candidate)
                attempt = uuid.uuid4().hex
                log = (directory / f'{candidate.position:04d}_{attempt}.log').open('x')
                command = [sys.executable, '-m', 'train_and_eval.queue_execution.worker',
                    '--config', str(candidate.path.relative_to(root)), '--name', candidate.name,
                    '--sha256', candidate.sha256, '--workers', str(args.workers),
                    '--threads', str(args.threads), '--temperature-limit', str(gate.limit)]
                side, cpus = allocator.choose() if allocator else (None, [])
                if cpus:
                    command += ['--cpus', ','.join(map(str,cpus))]
                if getattr(args, 'serial_evaluation', False):
                    command += ['--evaluation-lock', str((root / artifacts_directory / '.evaluation.lock').resolve())]
                if candidate.source:
                    command += ['--source', candidate.source]
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                phase_path=directory/f'{attempt}.phase.json'
                env['PPO_QUEUE_PHASE_PATH'] = str(phase_path.resolve())
                claimed_not_before = control.scalar(text('SELECT clock_timestamp()')).isoformat()
                try:
                    process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                active[candidate.name] = Active(candidate, process, log, time.monotonic(), attempt)
                active[candidate.name].phase_path=phase_path
                active[candidate.name].cpus=cpus
                if allocator: allocator.attach(side, process)
                event('worker_start',name=candidate.name,pid=process.pid,cpus=cpus,source=candidate.source)
                receipt = {'name': candidate.name, 'config_sha256': candidate.sha256,
                    'pid': process.pid, 'config_path': str(candidate.path.relative_to(root)),
                    'created_not_before': claimed_not_before}
                try:
                    receipt['pid_start_ticks'] = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
                except OSError:
                    receipt['pid_start_ticks'] = None
                active[candidate.name].receipt = receipt
                with (directory / f'{candidate.position:04d}_{attempt}.attempt.json').open('x') as receipt_stream:
                    json.dump(receipt, receipt_stream)
                    receipt_stream.flush()
                    os.fsync(receipt_stream.fileno())
                launched = True
                print(f'Start {candidate.name}: PID {process.pid}, log {log.name}', flush=True)
            if pending and not active and not launched and occupied_slots(control) == 0:
                eligible = any(states.get(job.name) is None and runnable(job, states) for job in pending)
                if not eligible:
                    raise RuntimeError('Remaining work is blocked by incomplete or skipped dependencies: ' +
                                       ', '.join(job.name for job in pending[:5]))
            time.sleep(.5)
        return 1 if failed else 0
    except (KeyboardInterrupt, EOFError):
        interrupted = True
        print('Queue interrupted; signalling owned children to persist their interrupted state.', flush=True)
        return 130
    except Exception:
        failed = True
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if active:
            stop_children(active)
            for name, item in active.items():
                item.log.close()
                record = {'name': name, 'attempt': item.attempt, 'pid': item.process.pid,
                    'wall_seconds': time.monotonic() - item.started,
                    'exit_code': item.process.returncode, 'completed': False}
                records.append(record)
                try:
                    reconcile_attempt(engine, item, 'Queue stopped; worker was reaped')
                    with engine.connect() as final_connection:
                        record['completed'] = (item.process.returncode == 0 and
                            database_states(final_connection, [name]).get(name) == 'completed')
                except Exception as error:
                    print(f'Could not reconcile {name}; preserve and inspect database state: {error}', flush=True)
        if recorder:
            try: recorder.close()
            except Exception as error:
                print(f'Telemetry finalization failed; preserve raw file: {error}', flush=True)
            total = time.monotonic() - started
            completed = sum(record['completed'] for record in records)
            (directory / 'summary.json').write_text(json.dumps({'workers': args.workers,
                'wall_seconds': total, 'completed_new_runs': completed,
                'runs_per_hour': completed * 3600 / total if total else None,
                'interrupted': interrupted, 'failed': failed,
                'thermal_abort': thermal_abort, 'control_cpu_max_c': control_cpu_max_c, 'runs': records}, indent=2))
            print(f'Queue measurements: {directory}', flush=True)
        if events:
            events.write(json.dumps({'event':'queue_end','timestamp':time.time(),'failed':failed,'interrupted':interrupted})+'\n')
            events.close()
        if control is not None:
            control.invalidate()
            control.close()
        engine.dispose()
        signal.signal(signal.SIGTERM, previous_sigterm)
