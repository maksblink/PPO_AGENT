"""One claimed run in one process; configure thread pools before Torch imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import threading
import time
from pathlib import Path

from train_and_eval.database.session import create_database_engine
from train_and_eval.queue_execution.core import claim, status, cpu_temperature

BUSY = 75
HOT = 76


def configure_threads(threads):
    if threads < 1:
        raise ValueError('threads must be positive')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = str(threads)
    os.environ['OMP_DYNAMIC'] = 'FALSE'
    os.environ['MKL_DYNAMIC'] = 'FALSE'


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Worker received signal {signum}')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpus')
    parser.add_argument('--cuda-retried', action='store_true')
    parser.add_argument('--config', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--source')
    parser.add_argument('--evaluation-lock')
    parser.add_argument('--workers', type=int, choices=(1, 2, 4), required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--temperature-limit', type=float, required=True)
    args = parser.parse_args(argv)
    if args.threads != 1:
        raise RuntimeError('Reproducible queue workers require --threads 1')
    if args.cpus:
        cpus = {int(x) for x in args.cpus.split(',')}
        os.sched_setaffinity(0, cpus)
        if os.sched_getaffinity(0) != cpus: raise RuntimeError('Affinity readback failed')
    configure_threads(args.threads)
    os.environ['PPO_QUEUE_THREAD_POLICY'] = '16-to-1'
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    # Retry only before a DB claim/run exists. exec replaces this process, preserving
    # PID, affinity, scheduler capacity and cancellation semantics.
    import torch
    from train_and_eval.queue_execution.runtime import phase
    try:
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        probe = torch.empty(1, device='cuda')
        torch.cuda.synchronize()
        del probe
    except Exception as error:
        if args.cuda_retried:
            phase('cuda_failed', error=str(error))
            return 1
        phase('cuda_retry_wait', seconds=30, error=str(error))
        time.sleep(30)
        import sys
        command = [sys.executable, '-m', 'train_and_eval.queue_execution.worker',
                   *(sys.argv[1:] if argv is None else argv), '--cuda-retried']
        os.execv(sys.executable, command)
    engine = create_database_engine()
    try:
        with claim(engine, args.name, args.workers) as connection:
            if connection is None:
                return BUSY
            existing = status(connection, args.name)
            if existing == 'completed':
                return 0
            if existing is not None:
                raise RuntimeError(f'Existing run {args.name}: {existing}; preserving it')
            if args.source and status(connection, args.source) != 'completed':
                raise RuntimeError(f'Source run {args.source} is not completed')
            if hashlib.sha256(Path(args.config).read_bytes()).hexdigest() != args.sha256:
                raise RuntimeError('Config changed after queue planning')
            temperature = cpu_temperature()
            if temperature is None or temperature >= args.temperature_limit:
                return HOT

            # Dedicated lock session remains owned by this training process.
            from sqlalchemy import text
            connection.execute(text("SET statement_timeout = '5s'"))
            lost = threading.Event()
            stop = threading.Event()

            def watch_connection():
                while not stop.wait(2):
                    try:
                        connection.execute(text('SELECT 1'))
                    except Exception:
                        lost.set()
                        os.kill(os.getpid(), signal.SIGINT)
                        return

            torch.set_num_threads(args.threads)
            torch.set_num_interop_threads(1)
            print(json.dumps({'phase': 'before_ppo_initialization', 'worker_pid': os.getpid(),
                'torch_version': torch.__version__, 'threads': torch.get_num_threads(),
                'interop_threads': torch.get_num_interop_threads(),
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cudnn_benchmark': torch.backends.cudnn.benchmark,
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'inherited'),
                'started_at': time.time()}), flush=True)
            from train_and_eval.training.cli import main as train
            watcher = threading.Thread(target=watch_connection, daemon=True)
            watcher.start()
            try:
                from train_and_eval.queue_execution.evaluation_gate import evaluation_lock_scope
                with evaluation_lock_scope(args.evaluation_lock):
                    code = train(['--config', args.config, '--plain-output', '--traceback'])
            finally:
                stop.set()
                watcher.join(timeout=7)
            print(json.dumps({'phase': 'after_training',
                'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                'cudnn_deterministic': torch.backends.cudnn.deterministic,
                'cudnn_benchmark': torch.backends.cudnn.benchmark}), flush=True)
            if lost.is_set():
                return 1
            if code != 0:
                from train_and_eval.queue_execution.core import mark_abandoned
                mark_abandoned(connection, args.name, f'Worker exited with code {code}')
            elif status(connection, args.name) != 'completed':
                raise RuntimeError('Training returned success without a completed database run')
            return code
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f'QUEUE WORKER FAILED: {error}', flush=True)
        return 1
    finally:
        engine.dispose()


if __name__ == '__main__':
    raise SystemExit(main())
