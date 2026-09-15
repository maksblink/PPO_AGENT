"""Compact study progress; rendering never changes training or selection."""
from __future__ import annotations

from datetime import datetime
import math
import shutil
import statistics
import time


def weeks(bounds):
    return (datetime.fromisoformat(bounds["end"]) -
            datetime.fromisoformat(bounds["start"])).total_seconds() / 604800


class WalkForwardProgress:
    def __init__(self, protocol, plan, *, stream, live=True, max_cycles=None):
        self.stream = stream
        self.interactive = live and stream.isatty()
        self.name = protocol.name
        self.protocol = protocol
        self.cycles = plan["cycles"]
        self.limit = min(max_cycles or len(self.cycles), len(self.cycles))
        self.done = {key: {} for key in ("base", "refit", "validation", "reference", "test")}
        self.results = {"validation": {}, "test": {}}
        self.active = None
        self.phase = "INITIALIZING"
        self.number = 0
        self.started = time.monotonic()
        self.last_render = 0.
        self.lines = 0
        self.total = {key: sum(self.amount(c, key) for c in self.cycles) for key in self.done}

    def amount(self, cycle, key):
        if key == "base":
            epochs = self.protocol.bootstrap_epochs if cycle["number"] == 1 else self.protocol.update_epochs
            return weeks(cycle["update"]) * epochs
        if key == "refit":
            return weeks(cycle["validation"]) * self.protocol.refit_epochs
        return weeks(cycle["test"] if key == "test" else cycle["validation"])

    def begin(self, cycle, key):
        self.number = cycle["number"]
        self.active = (cycle, key)
        self.phase = key.upper()
        self.requested = 0
        self.render(force=True)

    def complete(self, cycle, key, metrics=None, *, render=True):
        self.done[key][cycle["number"]] = self.amount(cycle, key)
        if metrics is not None and key in self.results:
            self.results[key][cycle["number"]] = (cycle[key], metrics)
        if self.active and self.active[0]["number"] == cycle["number"] and self.active[1] == key:
            self.active = None
        if render:
            self.render(force=True)

    def metric_lines(self, key, *, compact=False):
        rows = self.results[key]
        lines = [f"{key.upper()} | completed {'weeks' if key == 'test' else 'windows'}: {len(rows)}"]
        for metric in ("balanced_score", "agent_return", "agent_max_drawdown"):
            values = [(number, bounds, data[metric]) for number, (bounds, data) in rows.items()
                      if data.get(metric) is not None and math.isfinite(data[metric])]
            if not values:
                lines.append(f"  {metric}: n/a")
                continue
            best = max(values, key=lambda v: v[2])
            worst = min(values, key=lambda v: v[2])
            def label(item):
                if compact:
                    return f"{item[2]:+.5f} c{item[0]} {item[1]['start'][:10]}"
                return f"{item[2]:+.5f} c{item[0]} [{item[1]['start'][:10]}, {item[1]['end'][:10]})"
            numbers = [v[2] for v in values]
            lines.append(f"  {metric}: mean {statistics.mean(numbers):+.5f} | median {statistics.median(numbers):+.5f}")
            lines.append(f"    best {label(best)} | worst {label(worst)}")
        return lines

    def render(self, *, force=False):
        now = time.monotonic()
        if not force and (not self.interactive or now - self.last_render < .5):
            return
        # Plain logs get a snapshot per finished cycle and at start/end only.
        if not self.interactive and self.phase not in ("INITIALIZING", "CYCLE COMPLETED", "PAUSED", "COMPLETED", "FAILED"):
            return
        lines = [f"WALK FORWARD | {self.name}",
                 f"Cycle {self.number}/{len(self.cycles)} | invocation through {self.limit} | {self.phase} | session {int(now-self.started)}s",
                 "Week work: done / total / remaining (training includes epochs; overlapping windows repeat)"]
        for key, total in self.total.items():
            done = sum(self.done[key].values())
            lines.append(f"  {key:10s} {done:g} / {total:g} / {max(0., total-done):g}")
        width = max(20, shutil.get_terminal_size((120, 24)).columns - 1)
        compact = self.interactive and width < 120
        lines += self.metric_lines("validation", compact=compact) + self.metric_lines("test", compact=compact)
        if self.active:
            lines.append("Counters: completed operations. Narrow view: extrema show start dates.")
        if self.interactive:
            # Keep one physical row per line; automatic wrapping breaks cursor accounting.
            lines = [line[:width] for line in lines]
            if self.lines:
                self.stream.write(f"\033[{self.lines}A")
            for line in lines:
                self.stream.write("\r\033[2K" + line + "\n")
            for _ in range(max(0, self.lines-len(lines))):
                self.stream.write("\r\033[2K\n")
            self.lines = max(self.lines, len(lines))
        else:
            self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()
        self.last_render = now

    def cycle_completed(self, number):
        self.number = number
        self.phase = "CYCLE COMPLETED"
        self.active = None
        self.render(force=True)

    def close(self, phase):
        self.phase = phase
        self.active = None
        self.render(force=True)

    # Implements the existing TrainingProgressReporter callbacks.
    def preflight(self, snapshot):
        pass

    def start(self, *, requested_steps, **kwargs):
        self.requested = requested_steps

    def training_update(self, *, completed_steps, **kwargs):
        self.phase = f"{self.active[1].upper()} {100*completed_steps/max(1,self.requested):.0f}%"
        self.render()

    def validation_started(self, **kwargs):
        cycle, _ = self.active
        self.complete(cycle, "base", render=False)
        self.begin(cycle, "validation")

    def validation_update(self, *, completed_steps, expected_steps):
        self.evaluation_update(completed_steps, expected_steps)

    def evaluation_update(self, completed_steps, expected_steps):
        self.phase = f"{self.active[1].upper()} {100*completed_steps/max(1,expected_steps):.0f}%"
        self.render()

    def validation_completed(self, metrics):
        cycle, _ = self.active
        self.complete(cycle, "validation", {
            key: getattr(metrics, key) for key in ("balanced_score", "agent_return", "agent_max_drawdown")
        })

    def finish(self, **kwargs):
        pass

    def fail(self, error, **kwargs):
        self.close("FAILED")
