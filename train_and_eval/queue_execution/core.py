"""Dependency planning, thermal admission and PostgreSQL session claims."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import yaml
from sqlalchemy import text

# Two-key advisory locks, separate from the application's existing lock IDs.
CONTROL = 724019
RUN = 724020
SLOT = 724021


def run_key(name):
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'big') & 0x7fffffff


@dataclass(frozen=True)
class Job:
    position: int
    path: Path
    name: str
    source: str | None
    sha256: str


def plan_jobs(selected):
    jobs = []
    for position, config in selected:
        raw = config.path.read_bytes()
        payload = yaml.safe_load(raw)
        continuation = payload['continuation']
        mode = continuation['mode']
        if mode not in {'fresh', 'resume'}:
            raise ValueError(f'Unsupported continuation mode: {mode}')
        source = continuation['source_run'] if mode == 'resume' else None
        if source is not None and not isinstance(source, str):
            raise ValueError('source_run must be a run name')
        jobs.append(Job(position, config.path, payload['run']['name'], source,
                        hashlib.sha256(raw).hexdigest()))
    by_name = {job.name: job for job in jobs}
    if len(by_name) != len(jobs):
        raise ValueError('Duplicate run names in selected queue')
    for job in jobs:
        seen = {job.name}
        source = job.source
        while source in by_name:
            if source in seen:
                raise ValueError(f'Checkpoint dependency cycle involving {source}')
            seen.add(source)
            source = by_name[source].source
    return jobs


def status(connection, name):
    return connection.scalar(text('SELECT status FROM runs WHERE name = :name'), {'name': name})


def runnable(job, states):
    return job.source is None or states.get(job.source) == 'completed'


def cpu_temperature(root=Path('/sys/class/hwmon')):
    """Read CPU package/core sensors only, never GPU/NVMe temperature."""
    values = []
    for directory in root.glob('hwmon*'):
        try:
            driver = (directory / 'name').read_text().strip()
        except OSError:
            continue
        if driver not in {'k10temp', 'coretemp', 'zenpower'}:
            continue
        for sensor in directory.glob('temp*_input'):
            try:
                value = float(sensor.read_text()) / 1000
                if math.isfinite(value) and 0 < value < 150:
                    values.append(value)
            except (ValueError, OSError):
                continue
    return max(values) if values else None


@dataclass
class ThermalGate:
    limit: float = 90
    resume: float = 85
    paused: bool = False

    def __post_init__(self):
        if not 0 < self.resume < self.limit <= 100:
            raise ValueError('Require 0 < resume temperature < admission limit <= 100')

    def admit(self, value):
        if value is None or not math.isfinite(value):
            self.paused = True
            return False
        if value >= self.limit:
            self.paused = True
        elif self.paused and value <= self.resume:
            self.paused = False
        return not self.paused


def try_lock(connection, namespace, key):
    return connection.scalar(text('SELECT pg_try_advisory_lock(:namespace, :key)'),
                             {'namespace': namespace, 'key': key})


def unlock(connection, namespace, key):
    connection.execute(text('SELECT pg_advisory_unlock(:namespace, :key)'),
                       {'namespace': namespace, 'key': key})


def lock_available(connection, namespace, key):
    available = try_lock(connection, namespace, key)
    if available:
        unlock(connection, namespace, key)
    return available


def occupied_slots(connection):
    return int(connection.scalar(text('''SELECT count(*) FROM pg_locks
        WHERE locktype = 'advisory' AND database = (SELECT oid FROM pg_database WHERE datname = current_database())
        AND classid = :namespace AND objid IN (1, 2, 3, 4) AND objsubid = 2 AND granted'''),
                                 {'namespace': SLOT}))


@contextmanager
def claim(engine, name, workers):
    """Worker owns both locks until training returns, even if its parent dies.

    The gate makes run + capacity acquisition atomic across processes. Session
    locks are deliberately kept on a dedicated connection, never in the pool.
    No timeout can steal work from a slow but live worker.
    """
    if workers not in (1, 2, 4):
        raise ValueError('workers must be 1, 2 or 4')
    connection = engine.connect().execution_options(isolation_level='AUTOCOMMIT')
    acquired = []
    try:
        connection.execute(text('SELECT pg_advisory_lock(:namespace, 2)'), {'namespace': CONTROL})
        try:
            if occupied_slots(connection) < workers and try_lock(connection, RUN, run_key(name)):
                acquired.append((RUN, run_key(name)))
                for slot in (1, 2, 3, 4):
                    if try_lock(connection, SLOT, slot):
                        acquired.append((SLOT, slot))
                        break
        finally:
            unlock(connection, CONTROL, 2)
        yield connection if len(acquired) == 2 else None
    finally:
        try:
            for namespace, key in reversed(acquired):
                unlock(connection, namespace, key)
        finally:
            # Physically close the session: pool reset does not clear advisory locks.
            connection.invalidate()
            connection.close()


def mark_abandoned(connection, name, reason):
    """Only caller holding the run claim may repair its own failed attempt."""
    connection.execute(text('''UPDATE runs SET status = 'failed', finished_at = now(),
        modified_at = now(), error_type = 'QueueWorkerExit', error_message = :reason
        WHERE name = :name AND status IN ('pending', 'running')'''),
                       {'name': name, 'reason': reason})
