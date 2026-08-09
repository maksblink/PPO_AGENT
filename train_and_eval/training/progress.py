from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass, field
from typing import Mapping, Protocol, TextIO


Number = int | float


@dataclass(frozen=True, slots=True)
class ValidationMetricSnapshot:
    evaluation_id: int
    checkpoint_id: int
    run_step: int
    trigger: str
    balanced_score: float | None
    agent_return: float | None
    always_long_return: float | None
    always_short_return: float | None
    agent_max_drawdown: float | None
    profit_factor: float | None
    win_rate: float | None
    market_exposure: float | None
    round_trips: int | None


class TrainingProgressReporter(Protocol):
    def start(
        self,
        *,
        run_id: int,
        run_name: str,
        requested_steps: int,
        model_steps_before: int,
    ) -> None: ...

    def training_update(
        self,
        *,
        completed_steps: int,
        model_steps: int,
        rollouts_completed: int,
        metrics: Mapping[str, Number],
    ) -> None: ...

    def validation_started(
        self,
        *,
        run_step: int,
        checkpoint_id: int,
        trigger: str,
        expected_steps: int,
    ) -> None: ...

    def validation_update(
        self,
        *,
        completed_steps: int,
        expected_steps: int,
    ) -> None: ...

    def validation_completed(
        self,
        metrics: ValidationMetricSnapshot,
    ) -> None: ...

    def finish(
        self,
        *,
        completed_steps: int,
        stopped_early: bool,
    ) -> None: ...

    def fail(
        self,
        error: BaseException,
        *,
        completed_steps: int,
    ) -> None: ...


@dataclass(slots=True)
class LiveTrainingProgress:
    stream: TextIO
    minimum_refresh_seconds: float = 0.15
    _run_id: int | None = field(default=None, init=False)
    _run_name: str = field(default="", init=False)
    _requested_steps: int = field(default=0, init=False)
    _model_steps_before: int = field(default=0, init=False)
    _completed_steps: int = field(default=0, init=False)
    _model_steps: int = field(default=0, init=False)
    _rollouts_completed: int = field(default=0, init=False)
    _training_metrics: dict[str, Number] = field(default_factory=dict, init=False)
    _validation_metrics: ValidationMetricSnapshot | None = field(default=None, init=False)
    _validation_status: str = field(default="not run yet", init=False)
    _validation_completed_steps: int = field(default=0, init=False)
    _validation_expected_steps: int = field(default=0, init=False)
    _validation_started_at: float | None = field(default=None, init=False)
    _validation_elapsed_total: float = field(default=0.0, init=False)
    _last_validation_duration: float | None = field(default=None, init=False)
    _phase: str = field(default="INITIALIZING", init=False)
    _started_at: float | None = field(default=None, init=False)
    _last_render_at: float = field(default=0.0, init=False)
    _rendered_lines: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)

    def start(
        self,
        *,
        run_id: int,
        run_name: str,
        requested_steps: int,
        model_steps_before: int,
    ) -> None:
        self._run_id = int(run_id)
        self._run_name = str(run_name)
        self._requested_steps = int(requested_steps)
        self._model_steps_before = int(model_steps_before)
        self._model_steps = int(model_steps_before)
        self._completed_steps = 0
        self._rollouts_completed = 0
        self._phase = "TRAINING"
        self._started_at = time.monotonic()
        self._render(force=True)

    def training_update(
        self,
        *,
        completed_steps: int,
        model_steps: int,
        rollouts_completed: int,
        metrics: Mapping[str, Number],
    ) -> None:
        self._completed_steps = int(completed_steps)
        self._model_steps = int(model_steps)
        self._rollouts_completed = int(rollouts_completed)
        self._training_metrics = dict(metrics)
        self._phase = "TRAINING"
        self._render()

    def validation_started(
        self,
        *,
        run_step: int,
        checkpoint_id: int,
        trigger: str,
        expected_steps: int,
    ) -> None:
        self._completed_steps = int(run_step)
        self._validation_completed_steps = 0
        self._validation_expected_steps = max(0, int(expected_steps))
        self._validation_started_at = time.monotonic()
        self._last_validation_duration = None
        self._validation_status = (
            f"running ({trigger}, checkpoint {checkpoint_id})"
        )
        self._phase = "VALIDATION"
        self._render(force=True)

    def validation_update(
        self,
        *,
        completed_steps: int,
        expected_steps: int,
    ) -> None:
        self._validation_completed_steps = max(0, int(completed_steps))
        self._validation_expected_steps = max(0, int(expected_steps))
        self._phase = "VALIDATION"
        self._render()

    def validation_completed(
        self,
        metrics: ValidationMetricSnapshot,
    ) -> None:
        now = time.monotonic()
        if self._validation_started_at is not None:
            duration = max(0.0, now - self._validation_started_at)
            self._validation_elapsed_total += duration
            self._last_validation_duration = duration
            self._validation_started_at = None

        self._validation_completed_steps = self._validation_expected_steps
        self._validation_metrics = metrics
        self._validation_status = (
            f"completed ({metrics.trigger}, evaluation {metrics.evaluation_id})"
        )
        self._phase = (
            "VALIDATION"
            if metrics.trigger == "final"
            else "TRAINING"
        )
        self._render(force=True)

    def finish(
        self,
        *,
        completed_steps: int,
        stopped_early: bool,
    ) -> None:
        self._completed_steps = int(completed_steps)
        self._phase = "STOPPED EARLY" if stopped_early else "COMPLETED"
        self._render(force=True)
        self._closed = True

    def fail(
        self,
        error: BaseException,
        *,
        completed_steps: int,
    ) -> None:
        self._completed_steps = int(completed_steps)
        self._phase = f"FAILED: {type(error).__name__}"
        self._render(force=True)
        self._closed = True

    @staticmethod
    def _format_duration(seconds: float | None) -> str:
        if seconds is None or not math.isfinite(seconds) or seconds < 0:
            return "--:--"

        total = int(round(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)

        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"

        return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def _metric_text(value: Number | None, *, digits: int = 5) -> str:
        if value is None:
            return "n/a"

        numeric = float(value)
        if not math.isfinite(numeric):
            return str(numeric)

        if isinstance(value, int):
            return f"{value:,}"

        if numeric == 0:
            return "0"

        magnitude = abs(numeric)
        if magnitude >= 1000 or magnitude < 0.0001:
            return f"{numeric:.3e}"

        return f"{numeric:.{digits}f}"

    @staticmethod
    def _bar_line(done: int, total: int) -> str:
        resolved_total = max(1, int(total))
        resolved_done = min(max(0, int(done)), resolved_total)
        fraction = resolved_done / resolved_total
        percent = 100.0 * fraction
        columns = shutil.get_terminal_size((110, 24)).columns
        width = min(44, max(18, columns - 68))
        filled = int(round(width * fraction))
        bar = "█" * filled + "░" * (width - filled)
        remaining = max(0, int(total) - int(done))

        return (
            f"[{bar}] {percent:6.2f}%  "
            f"{int(done):,}/{int(total):,}  remaining {remaining:,}"
        )

    def _wall_elapsed(self, now: float) -> float | None:
        if self._started_at is None:
            return None
        return max(0.0, now - self._started_at)

    def _current_validation_elapsed(self, now: float) -> float:
        if self._validation_started_at is None:
            return 0.0
        return max(0.0, now - self._validation_started_at)

    def _training_elapsed(self, now: float) -> float | None:
        wall = self._wall_elapsed(now)
        if wall is None:
            return None
        return max(
            0.0,
            wall
            - self._validation_elapsed_total
            - self._current_validation_elapsed(now),
        )

    def _training_timing_line(self) -> str:
        now = time.monotonic()
        elapsed = self._training_elapsed(now)
        rate = 0.0
        if elapsed is not None and elapsed > 0:
            rate = self._completed_steps / elapsed

        remaining = max(0, self._requested_steps - self._completed_steps)
        eta = None if rate <= 0 else remaining / rate

        return (
            f"Training {self._format_duration(elapsed)}  |  "
            f"ETA {self._format_duration(eta)}  |  "
            f"{rate:,.1f} steps/s  |  "
            f"rollouts {self._rollouts_completed:,}  |  "
            f"model step {self._model_steps:,}"
        )

    def _run_timing_line(self) -> str:
        now = time.monotonic()
        wall = self._wall_elapsed(now)
        validation = (
            self._validation_elapsed_total
            + self._current_validation_elapsed(now)
        )
        return (
            f"Run wall {self._format_duration(wall)}  |  "
            f"validation time {self._format_duration(validation)}"
        )

    def _training_lines(self) -> list[str]:
        m = self._training_metrics
        return [
            "TRAIN — latest PPO update",
            (
                "  ep_reward " + self._metric_text(m.get("ep_rew_mean"))
                + "  ep_len " + self._metric_text(m.get("ep_len_mean"))
                + "  approx_kl " + self._metric_text(m.get("approx_kl"))
                + "  clip_fraction " + self._metric_text(m.get("clip_fraction"))
            ),
            (
                "  entropy_loss " + self._metric_text(m.get("entropy_loss"))
                + "  explained_variance " + self._metric_text(m.get("explained_variance"))
                + "  learning_rate " + self._metric_text(m.get("learning_rate"))
            ),
            (
                "  policy_loss " + self._metric_text(m.get("policy_gradient_loss"))
                + "  value_loss " + self._metric_text(m.get("value_loss"))
                + "  updates " + self._metric_text(m.get("n_updates"), digits=0)
            ),
        ]

    def _validation_lines(self) -> list[str]:
        metrics = self._validation_metrics

        if self._validation_status == "not run yet":
            return [
                "VALIDATION",
                "  not run yet",
                "",
                "  balanced_score n/a  agent_return n/a  max_drawdown n/a  profit_factor n/a",
                "  win_rate n/a  exposure n/a  round_trips n/a  always_long n/a",
            ]

        expected = max(0, self._validation_expected_steps)
        done = min(max(0, self._validation_completed_steps), expected)
        now = time.monotonic()

        if self._validation_started_at is not None:
            elapsed = self._current_validation_elapsed(now)
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = max(0, expected - done)
            eta = None if rate <= 0 else remaining / rate
            timing = (
                f"  Elapsed {self._format_duration(elapsed)}  |  "
                f"ETA {self._format_duration(eta)}  |  "
                f"{rate:,.1f} steps/s"
            )
        else:
            duration = self._last_validation_duration
            rate = (
                expected / duration
                if duration is not None and duration > 0
                else 0.0
            )
            timing = (
                f"  Completed in {self._format_duration(duration)}  |  "
                f"avg {rate:,.1f} steps/s"
            )

        lines = [
            f"VALIDATION — {self._validation_status}",
            "  " + self._bar_line(done, expected),
            timing,
        ]

        if metrics is None:
            lines.append(
                "  balanced_score n/a  agent_return n/a  max_drawdown n/a  profit_factor n/a"
            )
            lines.append(
                "  win_rate n/a  exposure n/a  round_trips n/a  always_long n/a"
            )
            return lines

        lines.append(
            "  balanced_score " + self._metric_text(metrics.balanced_score)
            + "  agent_return " + self._metric_text(metrics.agent_return)
            + "  max_drawdown " + self._metric_text(metrics.agent_max_drawdown)
            + "  profit_factor " + self._metric_text(metrics.profit_factor)
        )
        lines.append(
            "  win_rate " + self._metric_text(metrics.win_rate)
            + "  exposure " + self._metric_text(metrics.market_exposure)
            + "  round_trips " + self._metric_text(metrics.round_trips, digits=0)
            + "  always_long " + self._metric_text(metrics.always_long_return)
        )
        return lines

    def _lines(self) -> list[str]:
        return [
            f"PPO RUN #{self._run_id}  {self._run_name}",
            f"Phase: {self._phase}",
            "",
            "TRAINING",
            self._bar_line(self._completed_steps, self._requested_steps),
            self._training_timing_line(),
            self._run_timing_line(),
            "",
            *self._training_lines(),
            "",
            *self._validation_lines(),
        ]

    def _render(self, *, force: bool = False) -> None:
        if self._closed:
            return

        now = time.monotonic()
        if (
            not force
            and now - self._last_render_at < self.minimum_refresh_seconds
        ):
            return

        lines = self._lines()
        previous_lines = self._rendered_lines

        if previous_lines:
            self.stream.write(f"\x1b[{previous_lines}A")

        for line in lines:
            self.stream.write("\x1b[2K\r")
            self.stream.write(line)
            self.stream.write("\n")

        for _ in range(max(0, previous_lines - len(lines))):
            self.stream.write("\x1b[2K\r\n")

        if previous_lines > len(lines):
            self.stream.write(
                f"\x1b[{previous_lines - len(lines)}A"
            )

        self.stream.flush()
        self._rendered_lines = len(lines)
        self._last_render_at = now
