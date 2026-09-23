from __future__ import annotations

import base64
from dataclasses import fields
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sqlalchemy import select

from train_and_eval.database.models import Checkpoint, Evaluation, Run, WalkForwardCycle, WalkForwardStudy
from train_and_eval.evaluation.metrics import EvaluationMetrics
from train_and_eval.reporting.artifacts import evaluation_artifact_directory
from train_and_eval.reporting.service import generate_run_report


def combine_tests(parts: list[pd.DataFrame], stake: float) -> tuple[pd.DataFrame, dict]:
    """Join zero-based weekly P&L additively; include the original zero peak."""
    if not parts:
        raise ValueError("No completed tests to report")
    offsets = {"agent_equity": 0.0, "always_long_equity": 0.0}
    result = []
    last = None
    for part in parts:
        part = part.copy()
        times = pd.to_datetime(part["timestamp"], utc=True)
        if times.empty or not times.is_monotonic_increasing or times.duplicated().any():
            raise ValueError("Test timestamps must be nonempty, ordered and unique")
        if last is not None and times.iloc[0] <= last:
            raise ValueError("Test trajectories overlap or are out of order")
        last = times.iloc[-1]
        for column in offsets:
            values = part[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite test equity")
            part[column] = values + offsets[column]
            offsets[column] = float(part[column].iloc[-1])
        result.append(part)
    curve = pd.concat(result, ignore_index=True)
    summary = {}
    for column, prefix in (("agent_equity", "agent"), ("always_long_equity", "always_long")):
        values = curve[column].to_numpy()
        peak = np.maximum.accumulate(np.r_[0., values])[1:]
        curve[f"{prefix}_drawdown"] = values - peak
        curve[f"{prefix}_pnl_pln"] = values * stake
        summary[prefix] = {"total_return_on_stake": float(values[-1]),
                           "total_pnl_pln": float(values[-1] * stake),
                           "max_drawdown_on_stake": float(np.min(values-peak)),
                           "max_drawdown_pln": float(np.min(values-peak)*stake)}
    return curve, summary


def _metrics(row):
    values = {}
    for field in fields(EvaluationMetrics):
        value = getattr(row, field.name)
        values[field.name] = getattr(value, "value", value)
    return values


def _stats(values):
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "minimum": float(array.min()), "maximum": float(array.max()),
            "positive_fraction": float((array > 0).mean())}


def generate_report(factory, study_id: int, *, project_root) -> Path:
    root = Path(project_root).resolve()
    directory = root / "artifacts" / "walk_forward" / f"{study_id:08d}"
    directory.mkdir(parents=True, exist_ok=True)
    with factory() as session:
        study = session.get(WalkForwardStudy, study_id)
        if study is None:
            raise ValueError("Study does not exist")
        cycles = list(session.scalars(select(WalkForwardCycle).where(
            WalkForwardCycle.study_id == study_id).order_by(WalkForwardCycle.number)))
        completed = [c for c in cycles if c.status == "completed"]
        if [c.number for c in completed] != list(range(1, len(completed)+1)):
            raise ValueError("Report requires a consecutive completed prefix of cycles")
        runs = list(session.scalars(select(Run).join(WalkForwardCycle, Run.cycle_id == WalkForwardCycle.id).where(
            WalkForwardCycle.study_id == study_id).order_by(Run.id)))
        stages = [{"run_id": r.id, "name": r.name, "cycle_id": r.cycle_id, "role": r.stage_role,
                   "candidate_id": r.candidate_id, "source_checkpoint_id": r.source_checkpoint_id,
                   "status": r.status.value, "window": r.window_metadata, "summary": r.stage_summary} for r in runs]
        validation_rows, test_rows, parts = [], [], []
        stake = float(study.protocol["run"]["environment"]["stake_pln"])
        for cycle in cycles:
            for candidate in (cycle.selection or {}).get("candidates", []):
                evaluation = session.get(Evaluation, candidate["evaluation_id"])
                validation_rows.append({"cycle": cycle.number, **candidate,
                                        "selected": candidate["checkpoint_id"] == cycle.selected_checkpoint_id,
                                        "role": "candidate", **cycle.plan["validation"], **_metrics(evaluation)})
            if cycle.reference_evaluation_id is not None:
                reference = session.get(Evaluation, cycle.reference_evaluation_id)
                validation_rows.append({"cycle": cycle.number, "checkpoint_id": cycle.source_checkpoint_id,
                                        "evaluation_id": reference.id, "selected": False,
                                        "role": "unchanged_reference", **cycle.plan["validation"], **_metrics(reference)})
            if cycle.status != "completed":
                continue
            evaluation = session.get(Evaluation, cycle.test_evaluation_id)
            checkpoint = session.get(Checkpoint, cycle.refit_checkpoint_id)
            if evaluation.checkpoint_id != checkpoint.id:
                raise ValueError("Test checkpoint does not match the recorded refit")
            artifact_dir = evaluation_artifact_directory(root, "artifacts", checkpoint.run_id, evaluation.id)
            trajectory_path = artifact_dir / "trajectory_test.parquet"
            if not trajectory_path.exists():
                # Release the read transaction before nested replay/report sessions.
                session.commit()
                generate_run_report(factory, run_id=checkpoint.run_id, evaluation_id=evaluation.id, project_root=root)
            frame = pd.read_parquet(trajectory_path)
            expected = cycle.plan["test_rows"]
            if (len(frame) != expected["rows"] or
                    list(frame.execution_index) != list(range(expected["start_index"], expected["end_index"]))):
                raise ValueError("Test trajectory indices differ from the frozen calendar plan")
            for column in ("agent_equity", "always_long_equity"):
                field = "agent_return" if column == "agent_equity" else "always_long_return"
                if not np.isclose(frame[column].iloc[-1], getattr(evaluation, field), rtol=1e-10, atol=1e-12):
                    raise ValueError("Test trajectory result differs from database")
            frame["cycle"] = cycle.number
            parts.append(frame)
            # The agreed benchmark opens once per test and force-closes at its end.
            baseline_gross = float(frame.close_price.iloc[-1] / frame.execution_price.iloc[0] - 1)
            baseline_fee = 2 * study.protocol["run"]["environment"]["fee_bps"] / 10000
            baseline_cost = baseline_gross - evaluation.always_long_return
            test_rows.append({"cycle": cycle.number, **cycle.plan["test"],
                              "steps": len(frame), "source_checkpoint_id": cycle.source_checkpoint_id,
                              "selected_checkpoint_id": cycle.selected_checkpoint_id,
                              "refit_checkpoint_id": cycle.refit_checkpoint_id, "evaluation_id": evaluation.id,
                              "always_long_cost_return": baseline_cost,
                              "always_long_fee_return": baseline_fee,
                              "always_long_swap_return": baseline_cost-baseline_fee,
                              **_metrics(evaluation)})
        attempts = [{"cycle": c.number, "status": c.status, "error": c.error,
                     "reference_evaluation_id": c.reference_evaluation_id, "selection": c.selection} for c in cycles
                    if c.selection is not None or c.reference_evaluation_id is not None]
        protocol, plan = study.protocol, study.plan
        identity = {"study_id": study.id, "name": study.name, "status": study.status,
                    "git_commit": study.git_commit, "protocol_sha256": study.protocol_sha256}
    validation = pd.DataFrame(validation_rows)
    tests = pd.DataFrame(test_rows)
    (directory/"attempts.json").write_text(json.dumps(attempts, indent=2, allow_nan=False))
    if not parts:
        summary = {**identity, "completed_tests": 0, "planned_tests": len(plan["cycles"]),
                   "stake_pln": stake, "capitalization": False,
                   "message": "No completed tests; validation attempts are not test results."}
        for name, value in (("summary", summary), ("protocol", protocol), ("plan", plan), ("stages", stages)):
            (directory/f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False))
        validation.to_csv(directory/"validation.csv", index=False)
        pd.DataFrame(columns=["cycle", "start", "end", "agent_return"]).to_csv(directory/"tests.csv", index=False)
        report = (f"<!doctype html><html lang='en'><meta charset='utf-8'><h1>{html.escape(identity['name'])}</h1>"
                  f"<pre>{html.escape(json.dumps(summary, indent=2))}</pre>"
                  f"<h2>Stop / attempt details</h2><pre>{html.escape(json.dumps(attempts, indent=2))}</pre>"
                  f"<h2>Validation</h2>{validation.to_html(index=False)}</html>")
        (directory/"report.html").write_text(report)
        return directory/"report.html"
    curve, summary = combine_tests(parts, stake)
    summary.update(identity)
    summary.update({"completed_tests": len(tests), "planned_tests": len(plan["cycles"]),
                    "stake_pln": stake, "capitalization": False, "test_weeks": protocol["test_weeks"],
                    "exposure_weighting": "available scored bars, not elapsed wall-clock time",
                    "agent_test_window_return": _stats(tests.agent_return),
                    "always_long_test_window_return": _stats(tests.always_long_return),
                    "selected_validation_score": _stats(validation.loc[validation.selected, "balanced_score"]),
                    "agent_exposure": float(np.average(tests.market_exposure, weights=tests.steps)),
                    "agent_round_trips": int(tests.round_trips.sum()),
                    "agent_trade_events": int(tests.trade_events_total.sum()),
                    "agent_fee_pln": float(tests.total_fee_return.sum()*stake),
                    "agent_swap_pln": float(tests.total_swap_return.sum()*stake),
                    "agent_cost_pln": float(tests.total_cost_return.sum()*stake),
                    "always_long_exposure": 1.0, "always_long_round_trips": len(tests),
                    "always_long_cost_pln": float(tests.always_long_cost_return.sum()*stake)})
    validation.to_csv(directory/"validation.csv", index=False)
    tests.to_csv(directory/"tests.csv", index=False)
    curve.to_parquet(directory/"test_trajectory.parquet", index=False)
    for name, value in (("summary", summary), ("protocol", protocol), ("plan", plan), ("stages", stages)):
        (directory/f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False))
    charts = []
    def save(fig, name):
        if name in {"test_pnl", "test_drawdown"}:
            for axis in fig.axes:
                locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
                axis.xaxis.set_major_locator(locator)
                axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.tight_layout()
        path = directory/f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        charts.append((name, base64.b64encode(path.read_bytes()).decode()))
    time = pd.to_datetime(curve.timestamp, utc=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    for prefix, label in (("agent", "PPO"), ("always_long", "Always long (weekly reset)")):
        ax.plot(time, curve[f"{prefix}_pnl_pln"], label=label, linewidth=1)
    ax.set(ylabel="Cumulative P&L (PLN, fixed stake)", title="Consecutive tests only")
    ax.legend(); ax.grid(alpha=.2); save(fig, "test_pnl")
    fig, ax = plt.subplots(figsize=(12, 4))
    for prefix in ("agent", "always_long"):
        ax.plot(time, curve[f"{prefix}_drawdown"]*stake, label=prefix)
    ax.set(ylabel="Drawdown (PLN)", title="Drawdown of the entire test path")
    ax.legend(); ax.grid(alpha=.2); save(fig, "test_drawdown")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(tests.cycle, tests.agent_return*stake, label="PPO")
    axes[0].plot(tests.cycle, tests.always_long_return*stake, label="Always long")
    axes[0].set_ylabel("Test P&L (PLN)"); axes[0].legend()
    axes[1].plot(tests.cycle, tests.market_exposure*100); axes[1].set_ylabel("Exposure (%)")
    axes[2].plot(tests.cycle, tests.total_cost_return*stake, label="Costs PLN")
    axes[2].plot(tests.cycle, tests.round_trips, label="Round trips"); axes[2].legend()
    axes[2].set_xlabel("Test cycle"); save(fig, "weekly_tests")
    fig, ax = plt.subplots(figsize=(12, 4))
    for key, group in validation[validation.role == "candidate"].groupby("candidate_order"):
        ax.plot(group.cycle, group.balanced_score, label=f"Candidate {int(key)}")
    ax.set(xlabel="Cycle", ylabel="Balanced score", title="Validation before refit (overlapping windows)")
    ax.legend(); ax.grid(alpha=.2); save(fig, "validation_scores")
    images = "".join(f'<img alt="{name}" src="data:image/png;base64,{data}">' for name, data in charts)
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>{html.escape(identity['name'])}</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{padding:6px;border:1px solid #ddd}}pre{{white-space:pre-wrap}}</style>
<h1>{html.escape(identity['name'])}</h1><p>{len(tests)} of {len(plan['cycles'])} planned tests. Seed {protocol['seed']}.
Validation evaluates the selected model before refit. Tests evaluate its separate refitted copy.
P&L is additive on a fixed {stake:g} PLN stake; drawdown is measured on that same nominal.
Always-long closes and reopens at each test boundary, with identical fees and swaps.
Exposure is weighted by available bars. Missing candles are not imputed.</p>
<h2>Study status</h2><p>{html.escape(identity["status"])}</p>
<pre>{html.escape(json.dumps([{ "cycle": a["cycle"], "error": a["error"]} for a in attempts if a["error"]], indent=2))}</pre>
<h2>Summary</h2><pre>{html.escape(json.dumps(summary, indent=2))}</pre>{images}
<h2>Weekly tests</h2>{tests[['cycle','start','end','agent_return','always_long_return','agent_max_drawdown','market_exposure','round_trips','total_cost_return']].to_html(index=False)}
<h2>Validation before refit</h2>{validation[['cycle','role','checkpoint_id','selected','balanced_score','agent_return','agent_max_drawdown']].to_html(index=False)}
<p>Full provenance: protocol.json, plan.json, stages.json, attempts.json. Detailed metrics: validation.csv, tests.csv.
The complete non-overlapping path is test_trajectory.parquet. Training checkpoints and individual trajectories remain under artifacts/runs.</p></html>'''
    (directory/"report.html").write_text(report)
    return directory/"report.html"
