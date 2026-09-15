from io import StringIO
from types import SimpleNamespace

import pytest

from train_and_eval.walk_forward.progress import WalkForwardProgress


def make_progress(stream=None, live=False):
    cycles = [
        {"number": 1, "update": {"start": "2020-01-06", "end": "2020-01-20"},
         "validation": {"start": "2020-01-20", "end": "2020-02-17"},
         "test": {"start": "2020-02-17", "end": "2020-02-24"}},
        {"number": 2, "update": {"start": "2020-01-20", "end": "2020-01-27"},
         "validation": {"start": "2020-01-27", "end": "2020-02-24"},
         "test": {"start": "2020-02-24", "end": "2020-03-02"}},
    ]
    protocol = SimpleNamespace(name="pilot", bootstrap_epochs=2, update_epochs=3, refit_epochs=2)
    return WalkForwardProgress(protocol, {"cycles": cycles}, stream=stream or StringIO(), live=live, max_cycles=1)


def test_week_work_counts_epochs_and_overlaps_without_duplicate_completion():
    p = make_progress()
    assert p.total == {"base": 7, "refit": 16, "validation": 8, "reference": 8, "test": 2}
    p.complete(p.cycles[0], "base")
    p.complete(p.cycles[0], "base")
    assert sum(p.done["base"].values()) == 4
    assert p.limit == 1  # Pilot does not shrink the full-study total.


def test_statistics_separate_windows_and_tests_and_rank_negative_dd():
    p = make_progress()
    for cycle, dd, ret in zip(p.cycles, [-.4, -.1], [-.2, .1]):
        p.complete(cycle, "test", {"balanced_score": ret+dd, "agent_return": ret, "agent_max_drawdown": dd})
    text = "\n".join(p.metric_lines("test"))
    assert "-0.25000" in text
    assert "-0.10000 c2" in text
    assert "-0.40000 c1" in text
    assert "2020-03-02" in text
    assert "n/a" in "\n".join(p.metric_lines("validation"))


def test_plain_output_has_no_per_step_or_per_stage_spam():
    p = make_progress()
    p.render(force=True)
    initial = p.stream.getvalue()
    p.begin(p.cycles[0], "base")
    p.start(requested_steps=1024)
    for step in range(1, 1025):
        p.training_update(completed_steps=step)
    p.validation_started()
    p.validation_update(completed_steps=20, expected_steps=100)
    assert p.stream.getvalue() == initial
    p.cycle_completed(1)
    assert p.stream.getvalue().count("WALK FORWARD") == 2
    assert "\033" not in p.stream.getvalue()


def test_training_callbacks_update_base_before_validation():
    p = make_progress()
    p.begin(p.cycles[0], "base")
    p.validation_started()
    assert p.done["base"] == {1: 4}
    p.validation_completed(SimpleNamespace(balanced_score=.1, agent_return=.2, agent_max_drawdown=-.1))
    assert p.done["validation"] == {1: 4}
    assert p.active is None


def test_interactive_output_refreshes_in_place(monkeypatch):
    import os
    monkeypatch.setattr("train_and_eval.walk_forward.progress.shutil.get_terminal_size", lambda _: os.terminal_size((80, 24)))
    class Terminal(StringIO):
        def isatty(self):
            return True
    p = make_progress(Terminal(), live=True)
    p.render(force=True)
    p.begin(p.cycles[0], "test")
    p.last_render = 0
    p.evaluation_update(50, 100)
    assert "TEST 50%" in p.stream.getvalue()
    assert "\033[" in p.stream.getvalue()


def test_table_layout_fits_standard_terminal_with_all_metrics():
    p = make_progress()
    for cycle in p.cycles:
        for key in ("test", "validation"):
            p.complete(cycle, key, {"balanced_score": -.2, "agent_return": .1, "agent_max_drawdown": -.3})
    lines = p.panel_lines(compact=True)
    assert len(lines) <= 24
    assert max(map(len, lines)) <= 79
    assert sum("balanced_score" in line for line in lines) == 2
    assert "Best" in "\n".join(lines)
    assert "Worst" in "\n".join(lines)


def test_bars_are_bounded_and_active_progress_does_not_complete_weeks():
    p = make_progress()
    assert p.bar(1, 2, 4) == "[##--]  50.0%"
    assert p.bar(2, 1, 4) == "[####] 100.0%"
    p.begin(p.cycles[0], "base")
    p.start(requested_steps=100)
    p.training_update(completed_steps=50)
    assert p.active_fraction == .5
    assert sum(p.done["base"].values()) == 0


def test_inline_panel_preserves_screen_and_final_snapshot_has_dates(monkeypatch):
    import os
    monkeypatch.setattr("train_and_eval.walk_forward.progress.shutil.get_terminal_size", lambda _: os.terminal_size((80, 24)))
    class Terminal(StringIO):
        def isatty(self):
            return True
    p = make_progress(Terminal(), live=True)
    p.begin(p.cycles[0], "test")
    p.complete(p.cycles[0], "test", {"balanced_score": .1, "agent_return": .2, "agent_max_drawdown": -.1})
    p.close("PAUSED")
    text = p.stream.getvalue()
    assert "\033[?" not in text
    assert "\033[H" not in text
    assert "\033[J" not in text
    assert "\033[23A" in text
    assert "[2020-02-17, 2020-02-24)" in text
    assert p.rendered_lines == 0


def test_small_terminal_uses_complete_plain_tables(monkeypatch):
    import os
    monkeypatch.setattr("train_and_eval.walk_forward.progress.shutil.get_terminal_size", lambda _: os.terminal_size((60, 12)))
    class Terminal(StringIO):
        def isatty(self):
            return True
    p = make_progress(Terminal(), live=True)
    p.render(force=True)
    p.begin(p.cycles[0], "test")
    p.cycle_completed(1)
    text = p.stream.getvalue()
    assert "\033" not in text
    assert text.count("WALK FORWARD") == 2
    assert "Worst" in text


def test_resize_does_not_rewind_into_shell_history(monkeypatch):
    import os
    size = [os.terminal_size((80, 24))]
    monkeypatch.setattr("train_and_eval.walk_forward.progress.shutil.get_terminal_size", lambda _: size[0])
    class Terminal(StringIO):
        def isatty(self):
            return True
    stream = Terminal()
    p = make_progress(stream, live=True)
    p.begin(p.cycles[0], "base")
    assert p.rendered_lines == 23
    assert p.rendered_lines < size[0].lines
    stream.seek(0)
    stream.truncate()
    size[0] = os.terminal_size((100, 30))
    p.render(force=True)
    assert "\033" not in stream.getvalue()
    p.render(force=True)
    assert "\033[23A" in stream.getvalue()
