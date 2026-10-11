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
import traceback

from sqlalchemy import text
from train_and_eval.database.session import create_database_engine
from train_and_eval.queue_execution.core import (
    CONTROL, RUN, claim, cpu_temperature, lock_available, mark_abandoned,
    occupied_slots, plan_jobs, run_key, runnable, ThermalGate, try_lock,
)
from train_and_eval.queue_execution.telemetry import Recorder, temperatures
from train_and_eval.queue_execution.affinity import Allocator, discover_groups
from train_and_eval.queue_execution.monitor import OutputReader, Progress, atomic_json, classify


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
    output: object
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
    configs = {}
    import yaml
    for job in jobs:
        cfg = yaml.safe_load(job.path.read_text())
        configs[job.name] = cfg
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
    progress = None
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
    managed_directory = os.environ.get('PPO_QUEUE_SESSION_DIR')
    if managed_directory:
        directory = Path(managed_directory).resolve()
        directory.relative_to((root / 'artifacts' / 'search_queues' / args.queue).resolve())
        if not (directory / 'hardware.json').is_file():
            raise RuntimeError('Managed session is missing hardware metadata')
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
        # Keep the existing explicit preview/confirmation for failed or cancelled
        # runs, before starting ANY child. Never automatically delete artifacts.
        skipped = set()
        decisions = {}
        print(f"Preflight: inspecting {len(jobs)} selected runs before any training starts.", flush=True)
        for position, config in selected:
            state = database_states(control, [config.name]).get(config.name)
            if state in {'pending', 'failed', 'cancelled'}:
                if not lock_available(control, RUN, run_key(config.name)):
                    continue
                from run_search_queue import handle_incomplete_run
                run_id = control.scalar(text('SELECT id FROM runs WHERE name = :name'), {'name': config.name})
                decision = handle_incomplete_run(config, run_id)
                decisions[config.name] = decision
                if decision == 'STOP':
                    return 0
                if decision == 'SKIP':
                    skipped.add(config.name)
        # Restart decisions can remove children too: refresh the entire plan.
        states = database_states(control, names)
        completed_before, blocked, pending = classify(jobs, states, skipped)
        progress = Progress(pending, configs, workers=args.workers)
        print(f'QUEUE PLAN | Selected: {len(jobs)} | Already completed: {len(completed_before)} | To run: {len(pending)} | Skipped: {len(skipped)} | Blocked: {len(blocked)}', flush=True)
        for name in sorted(blocked): print(f'BLOCKED: {name}', flush=True)
        directory.mkdir(parents=True, exist_ok=bool(managed_directory))
        recorder = Recorder(directory / 'telemetry.csv', groups, args.flush_seconds)
        events = (directory/'events.jsonl').open('x', buffering=1)
        def event(kind, **fields):
            durable = fields.pop('_durable', False)
            events.write(json.dumps(dict(event=kind, timestamp=time.time(), **fields))+'\n')
            if durable:
                events.flush()
                os.fsync(events.fileno())

        def consume(item, message):
            if not isinstance(message, dict): return
            progress.accept(item.job.name, message)
            label = message.get('phase', 'UNKNOWN')
            label = {'training': 'TRAIN', 'initialization': 'INITIALIZING',
                     'evaluation_wait': 'WAIT_EVAL', 'evaluation_start': message.get('stage', 'EVAL'),
                     'evaluation_end': 'FINALIZING'}.get(label, label.upper())
            if label == 'TRAIN' and item.job.name in progress.done['TRAIN']:
                label = 'FINALIZING'
            item.state.update(phase=label, updated_at=time.time())
            atomic_json(item.phase_path, item.state)
            event('worker_event', name=item.job.name, attempt=item.attempt, details=message)

        def finish(item, ok, code, interrupted_worker=False):
            item.output.finish(lambda m: consume(item, m))
            label = 'COMPLETED' if ok else ('INTERRUPTED' if interrupted_worker or code == 130 else 'FAILED')
            if code in (75, 76): label = 'DEFERRED'
            item.state.update(phase=label, updated_at=time.time(), exit_code=code)
            atomic_json(item.phase_path, item.state)
            if not ok and code not in (75, 76):
                event('worker_error', name=item.job.name, attempt=item.attempt,
                      exit_code=code, output_tail=''.join(item.output.tail), reader_error=item.output.error)
            elif item.output.error:
                event('output_reader_error', name=item.job.name, error=item.output.error)
            event('worker_exit', name=item.job.name, attempt=item.attempt,
                  pid=item.process.pid, code=code, status=label, _durable=True)
            progress.dirty = True
        event('queue_start', groups=groups)

        (directory / 'plan.json').write_text(json.dumps({'workers': args.workers,
            'serial_evaluation': getattr(args, 'serial_evaluation', False),
            'threads': args.threads, 'temperature_limit': gate.limit, 'temperature_resume': gate.resume,
            'abort_temperature': args.abort_temperature,
            'already_completed': sorted(completed_before), 'skipped': sorted(skipped),
            'blocked': sorted(blocked), 'decisions': decisions, 'stages': progress.counts(),
            'jobs': [{'name': j.name, 'source': j.source, 'sha256': j.sha256,
                      'position': j.position} for j in jobs]}, indent=2))
        progress.display(active, records, force=True)
        next_sample = 0
        next_control = 0
        sensors = {}
        temperature = None
        last_message = None
        last_message_at = 0
        while pending or active:
            now = time.monotonic()
            states = database_states(control, names)
            for item in active.values():
                for message in item.output.drain(): consume(item, message)
            # Validate actual database completion; exit code alone is insufficient.
            for name, item in list(active.items()):
                code = item.process.poll()
                if code is None:
                    continue
                states.update(database_states(control, [name]))
                finish(item, code == 0 and states.get(name) == 'completed', code)
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
                progress.thermal(f'{message or "Thermal admission resumed"}: CPU={temperature}')
            last_message = message
            if now >= next_sample:
                workers=[]
                for name,item in active.items():
                    phase = item.state.get('phase', 'STARTING')
                    workers.append({'name':name,'pid':item.process.pid,'phase':phase,'cpus':item.cpus})
                recorder.record(workers,sensors)
                next_sample = now + args.sample_seconds
            recorder.flush_due()
            progress.display(active, records)
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
                slot = next(i for i in range(1, args.workers+1) if all(a.slot != i for a in active.values()))
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
                phase_path=directory/f'worker_{slot}_last.state.json'
                env.pop('PPO_QUEUE_PHASE_PATH', None)
                env['PPO_QUEUE_EVENTS'] = '1'
                claimed_not_before = control.scalar(text('SELECT clock_timestamp()')).isoformat()
                process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, errors='replace',
                                           start_new_session=True)
                item = Active(candidate, process, OutputReader(process.stdout), time.monotonic(), attempt)
                active[candidate.name] = item
                item.slot = slot
                item.phase_path = phase_path
                item.cpus = cpus
                item.state = {'run':candidate.name, 'attempt':attempt, 'slot':slot,
                              'pid':process.pid, 'cpus':cpus, 'phase':'STARTING', 'updated_at':time.time()}
                atomic_json(phase_path, item.state)
                progress.dirty = True
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
                event('attempt_registered', attempt=attempt, slot=slot, receipt=receipt, _durable=True)
                launched = True
            if pending and not active and not launched and occupied_slots(control) == 0:
                eligible = any(states.get(job.name) is None and runnable(job, states) for job in pending)
                if not eligible:
                    raise RuntimeError('Remaining work is blocked by incomplete or skipped dependencies: ' +
                                       ', '.join(job.name for job in pending[:5]))
            time.sleep(.5)
        return 1 if failed else 0
    except (KeyboardInterrupt, EOFError):
        interrupted = True
        if progress:
            progress.thermal('Queue interrupted; stopping owned workers.')
        else:
            print('Queue interrupted before worker start.', flush=True)
        return 130
    except Exception:
        failed = True
        if events:
            event('queue_error', traceback=traceback.format_exc(), _durable=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if active:
            stop_children(active)
            for name, item in active.items():
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
                    progress.thermal(f'Reconciliation failed for {name}; see events.jsonl')
                    event('reconciliation_error', name=name, traceback=traceback.format_exc())
                finish(item, record['completed'], item.process.returncode, interrupted_worker=interrupted)
        if progress:
            progress.display({}, records, force=True, final=True)
            progress.close()
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
                'thermal_abort': thermal_abort, 'control_cpu_max_c': control_cpu_max_c,
                'stages': progress.counts(), 'completed_evaluations': progress.evaluations,
                'already_completed': len(completed_before), 'skipped': sorted(skipped),
                'blocked': sorted(blocked), 'runs': records}, indent=2))
            print(f'Queue measurements: {directory}', flush=True)
        if events:
            events.write(json.dumps({'event':'queue_end','timestamp':time.time(),'failed':failed,'interrupted':interrupted})+'\n')
            events.close()
        if control is not None:
            control.invalidate()
            control.close()
        engine.dispose()
        signal.signal(signal.SIGTERM, previous_sigterm)
