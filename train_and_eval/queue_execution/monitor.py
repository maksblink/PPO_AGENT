"""Compact session events, bounded worker output and final-stage counters."""
from collections import deque
import json
from pathlib import Path
import queue
import threading
import time
import sys
import os
import shutil

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


class TerminalBlock:
    """Keep the cursor beneath a fixed-height block; rewrite changed rows only."""
    def __init__(self, stream=None):
        self.stream = stream if stream is not None else sys.stdout
        self.interactive = bool(getattr(self.stream, 'isatty', lambda: False)()) and os.environ.get('TERM') != 'dumb'
        self.previous = []
        self.closed = False

    def render(self, lines, final=False):
        if self.closed: return
        # No escape sequences or periodic table spam in redirected output.
        if not self.interactive:
            if not self.previous or final:
                self.stream.write('\n'.join(lines) + '\n')
                self.stream.flush()
            self.previous = list(lines)
            return
        columns = shutil.get_terminal_size((120, 30)).columns
        width = max(1, columns - 1)
        clean = [''.join(c if c >= ' ' and c != '\x7f' else ' ' for c in line) for line in lines]
        rows = [line if len(line) <= width else line[:max(0, width-3)] + '...'[:width] for line in clean]
        if not self.previous:
            self.stream.write('\n'.join(rows) + '\n')
        else:
            if len(rows) != len(self.previous):
                raise ValueError('Terminal status block height changed')
            for i, (old, new) in enumerate(zip(self.previous, rows)):
                if old == new: continue
                distance = len(rows) - i
                self.stream.write(f'\x1b[{distance}A\r\x1b[2K{new}\x1b[{distance}B\r')
        self.stream.flush()
        self.previous = rows

    def close(self):
        self.closed = True


class Progress:
    def __init__(self, jobs, configs, workers=2):
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
        self.workers = workers
        self.block = TerminalBlock()
        self.lines = []
        self.thermal_text = 'Thermal admission: RUNNING'

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

    def thermal(self, message):
        self.thermal_text = message
        if self.lines:
            self.lines[-1] = message
            self.block.render(self.lines)
        if not self.block.interactive:
            print(message, file=self.block.stream, flush=True)

    def close(self):
        self.block.close()

    def display(self, active, records, force=False, final=False):
        now = time.monotonic()
        if not force and now-self.last_print < 90: return
        self.last_print = now; self.dirty = False
        completed = sum(r['completed'] for r in records)
        failures = sum(not r['completed'] and r.get('exit_code') not in (130, -2, -15) for r in records)
        hours = (now-self.started)/3600
        lines = [f'QUEUE STATUS | Elapsed: {hours:.2f} h | Completed: {completed}/{len(self.targets["TRAIN"])} | Failed: {failures} | Rate: {completed/hours if hours else 0:.2f} runs/h',
                 'Final stages per run | Completed | Active | Remaining']
        for stage, counts in self.counts().items():
            busy = sum(i.state.get('phase') == stage for i in active.values())
            remaining = counts['total']-counts['completed']
            lines.append(f'{stage:20} {counts["completed"]:4}/{counts["total"]:<4} {busy:6} {remaining:10}')
        lines.append('Remaining includes active work. Refresh: 90 s. Periodic + final evaluations:')
        lines.append(', '.join(f'{s}={n}' for s,n in self.evaluations.items()))
        by_slot = {item.slot:item for item in active.values()}
        for slot in range(1, self.workers+1):
            item = by_slot.get(slot)
            lines.append(f'Worker {slot}: {item.job.name} | {item.state.get("phase", "STARTING")}' if item else
                         f'Worker {slot}: - | {"STOPPED" if final else "IDLE"}')
        lines.append(self.thermal_text)
        self.lines = lines
        self.block.render(lines, final=final)
