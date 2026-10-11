"""Compact session events, bounded worker output and final-stage counters."""
from collections import deque
import json
from pathlib import Path
import queue
import threading
import time

PREFIX = '@@PPO_QUEUE@@'
STAGES = ('TRAIN', 'EVAL_TRAIN', 'EVAL_VAL')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def classify(jobs, states, skipped):
    completed = {j.name for j in jobs if states.get(j.name) == 'completed'}
    blocked = set()
    by_name = {j.name: j for j in jobs}
    changed = True
    while changed:
        changed = False
        for job in jobs:
            if job.name in completed | skipped | blocked or not job.source:
                continue
            if (job.source in skipped | blocked or
                (job.source not in by_name and states.get(job.source) != 'completed')):
                blocked.add(job.name)
                changed = True
    return completed, blocked, [j for j in jobs if j.name not in completed | skipped | blocked]


class OutputReader:
    """Drain output continuously; only structured events and a bounded error tail survive."""
    def __init__(self, stream):
        self.stream = stream
        self.messages = queue.Queue(maxsize=1024)
        self.tail = deque(maxlen=64)  # at most 64 * 4096 characters
        self.error = None
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        try:
            while True:
                line = self.stream.readline(4096)
                if not line:
                    break
                if line.startswith(PREFIX):
                    try:
                        self.messages.put(json.loads(line[len(PREFIX):]))
                        continue
                    except ValueError:
                        pass
                self.tail.append(line)
        except Exception as error:
            self.error = str(error)
        finally:
            self.stream.close()

    def drain(self):
        messages = []
        while True:
            try: messages.append(self.messages.get_nowait())
            except queue.Empty: return messages

    def finish(self, consume):
        # Continue draining while joining, including when the bounded queue is full.
        deadline = time.monotonic() + 10
        while self.thread.is_alive() and time.monotonic() < deadline:
            for message in self.drain(): consume(message)
            self.thread.join(.05)
        for message in self.drain(): consume(message)
        if self.thread.is_alive():
            self.error = 'Worker output did not reach EOF after exit'


class Progress:
    def __init__(self, jobs, configs):
        self.targets = {stage: set() for stage in STAGES}
        for job in jobs:
            self.targets['TRAIN'].add(job.name)
            if configs[job.name].get('evaluation', {}).get('training_mode', 'scheduled') in ('scheduled', 'final_only'):
                self.targets['EVAL_TRAIN'].add(job.name)
                self.targets['EVAL_VAL'].add(job.name)
        self.done = {stage: set() for stage in STAGES}
        self.evaluations = {'EVAL_TRAIN': 0, 'EVAL_VAL': 0}
        self.seen_evaluations = set()
        self.started = time.monotonic()
        self.last_print = 0
        self.dirty = True

    def accept(self, name, message):
        stage = message.get('stage')
        if stage in self.evaluations and message.get('completed'):
            key = (name, stage, message.get('checkpoint_id'), message.get('trigger'))
            if key not in self.seen_evaluations:
                self.seen_evaluations.add(key)
                self.evaluations[stage] += 1
        if stage in self.done and message.get('completed') and (
                stage == 'TRAIN' or message.get('trigger') == 'final'):
            self.done[stage].add(name)
        self.dirty = True

    def counts(self):
        return {s: {'completed': len(self.done[s] & self.targets[s]),
                    'total': len(self.targets[s])} for s in STAGES}

    def display(self, active, records, force=False):
        now = time.monotonic()
        if not force and now-self.last_print < (3 if self.dirty else 30): return
        self.last_print = now; self.dirty = False
        completed = sum(r['completed'] for r in records)
        failures = sum(not r['completed'] and r.get('exit_code') not in (130, -2, -15) for r in records)
        hours = (now-self.started)/3600
        print(f'\nQUEUE STATUS | Elapsed: {hours:.2f} h | Completed runs: {completed}/{len(self.targets["TRAIN"])} | Failed attempts: {failures} | Rate: {completed/hours if hours else 0:.2f} runs/h', flush=True)
        print('Final stages per run | Completed | Active | Remaining', flush=True)
        for stage, counts in self.counts().items():
            busy = sum(i.state.get('phase') == stage for i in active.values())
            remaining = counts['total']-counts['completed']
            print(f'{stage:20} {counts["completed"]:4}/{counts["total"]:<4} {busy:6} {remaining:10}', flush=True)
        print('Remaining includes active work. All completed evaluations (including periodic): '
              + ', '.join(f'{s}={n}' for s,n in self.evaluations.items()), flush=True)
        for item in sorted(active.values(), key=lambda i:i.slot):
            print(f'Worker {item.slot}: {item.job.name} | {item.state.get("phase", "STARTING")}', flush=True)
