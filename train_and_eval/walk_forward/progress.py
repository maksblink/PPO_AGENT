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
        self.rendered_lines = 0
        self.rendered_size = None
        self.active_fraction = 0.
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
        self.active_fraction = 0.
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

    @staticmethod
    def bar(done, total, width=16):
        fraction = min(1., max(0., done / total)) if total else 0.
        filled = int(width * fraction)
        return "[" + "#" * filled + "-" * (width-filled) + f"] {100*fraction:5.1f}%"

    @staticmethod
    def table(headers, rows):
        widths = [max(len(str(row[i])) for row in [headers, *rows]) for i in range(len(headers))]
        def line(row):
            return " | ".join(str(value).ljust(width) for value, width in zip(row, widths))
        return [line(headers), "-+-".join("-"*width for width in widths), *map(line, rows)]

    def metric_lines(self, key, *, compact=False):
        rows = self.results[key]
        title = f"{key.upper()} | completed windows: {len(rows)}"
        table_rows = []
        for metric, label in (("balanced_score", "balanced_score"),
                              ("agent_return", "return"), ("agent_max_drawdown", "DD")):
            values = [(number, bounds, data[metric]) for number, (bounds, data) in rows.items()
                      if data.get(metric) is not None and math.isfinite(data[metric])]
            if not values:
                table_rows.append([label, "n/a", "n/a", "n/a", "n/a"])
                continue
            best = max(values, key=lambda v: v[2])
            worst = min(values, key=lambda v: v[2])
            def extreme(item):
                result = f"{item[2]:+.5f} c{item[0]}"
                if not compact:
                    result += f" [{item[1]['start'][:10]}, {item[1]['end'][:10]})"
                return result
            numbers = [v[2] for v in values]
            table_rows.append([label, f"{statistics.mean(numbers):+.5f}",
                               f"{statistics.median(numbers):+.5f}", extreme(best), extreme(worst)])
        return [title, *self.table(["Metric", "Mean", "Median", "Best", "Worst"], table_rows)]

    def panel_lines(self, *, compact=False):
        total_cycles = len(self.cycles)
        completed_cycles = len(self.done["test"])
        lines = [f"WALK FORWARD | {self.name}",
                 f"Cycle {self.number}/{total_cycles} | stop after {self.limit} | {self.phase} | session {int(time.monotonic()-self.started)}s",
                 "Cycles " + self.bar(completed_cycles, total_cycles)]
        rows = []
        for key, total in self.total.items():
            done = sum(self.done[key].values())
            rows.append([key, f"{done:g}", f"{total:g}", f"{max(0., total-done):g}", self.bar(done, total)])
        lines += self.table(["Week work", "Done", "Total", "Left", "Progress"], rows)
        active = "Active: " + (self.bar(self.active_fraction, 1) if self.active else "none")
        lines.append(active)
        lines += self.metric_lines("validation", compact=compact) + self.metric_lines("test", compact=compact)
        lines.append("Weeks include repeated work/epochs. DD: closer to 0 is better.")
        return lines

    def render(self, *, force=False):
        now = time.monotonic()
        if not force and (not self.interactive or now - self.last_render < .5):
            return
        size = shutil.get_terminal_size((120, 24))
        terminal_phase = self.phase in ("PAUSED", "COMPLETED", "FAILED")
        # Reserve a row for the cursor after the final newline. Otherwise the
        # first panel row scrolls out of view and cursor-up cannot reach it.
        lines = self.panel_lines(compact=size.columns < 150)[:-1]
        fits = size.columns >= 80 and len(lines) < size.lines
        live_panel = self.interactive and fits and not terminal_phase
        if not live_panel and self.phase not in (
                "INITIALIZING", "CYCLE COMPLETED", "PAUSED", "COMPLETED", "FAILED"):
            return
        if self.interactive:
            self.clear_previous(size)
        if live_panel:
            self.stream.write("\n".join(line[:size.columns-1] for line in lines) + "\n")
            self.rendered_lines = len(lines)
            self.rendered_size = size
        else:
            self.stream.write("\n".join(self.panel_lines()) + "\n")
        self.stream.flush()
        self.last_render = now

    def clear_previous(self, size):
        # After a resize the terminal may have reflowed the old panel. Do not
        # guess cursor offsets and risk overwriting unrelated shell history.
        if self.rendered_lines and size == self.rendered_size:
            self.stream.write(f"\033[{self.rendered_lines}A")
            for _ in range(self.rendered_lines):
                self.stream.write("\r\033[2K\n")
            self.stream.write(f"\033[{self.rendered_lines}A")
        self.rendered_lines = 0
        self.rendered_size = None

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
        self.active_fraction = min(1., max(0., completed_steps/max(1,self.requested)))
        self.phase = f"{self.active[1].upper()} {100*self.active_fraction:.0f}%"
        self.render()

    def validation_started(self, **kwargs):
        cycle, _ = self.active
        self.complete(cycle, "base", render=False)
        self.begin(cycle, "validation")

    def validation_update(self, *, completed_steps, expected_steps):
        self.evaluation_update(completed_steps, expected_steps)

    def evaluation_update(self, completed_steps, expected_steps):
        self.active_fraction = min(1., max(0., completed_steps/max(1,expected_steps)))
        self.phase = f"{self.active[1].upper()} {100*self.active_fraction:.0f}%"
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
