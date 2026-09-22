"""Automatic PostgreSQL integration test in a dedicated test database."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from alembic import command
from alembic.config import Config
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import select
from tests.database_support import isolated_test_schema, resolve_test_database_url
import yaml

from train_and_eval.database.models import Checkpoint, Evaluation, Run, WalkForwardCycle
from train_and_eval.database.session import create_session_factory
from train_and_eval.walk_forward.service import execute
from train_and_eval.walk_forward.config import GRID_FIELDS
from train_and_eval.walk_forward.reporting import generate_report


@pytest.fixture
def database(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    url = resolve_test_database_url(root)
    with isolated_test_schema(url) as (engine, scoped):
        monkeypatch.setenv("DATABASE_URL", scoped.render_as_string(hide_password=False))
        command.upgrade(Config(str(root / "alembic.ini")), "head")
        yield create_session_factory(engine), engine


def synthetic_project(tmp_path, end="2026-04-13"):

    root = tmp_path/"project"
    root.mkdir()
    (root/"data").mkdir()
    manifest_dir = root/"train_and_eval/market_data"
    manifest_dir.mkdir(parents=True)
    times = pd.date_range("2026-01-05", end, freq="5min", tz="UTC", inclusive="left")
    times = times[times.weekday < 5]
    # Explicit gaps exercise variable weekly row counts.
    times = times.delete([11000, 11001, 15432])
    price = 100 + np.arange(len(times))*.0001 + .05*np.sin(np.arange(len(times))*.1)
    table = pa.table({
        "Datetime": pa.array(times, type=pa.timestamp("ns", tz="UTC")),
        "Open_NQ": pa.array(price, type=pa.float64()), "High_NQ": pa.array(price+.02, type=pa.float64()),
        "Low_NQ": pa.array(price-.02, type=pa.float64()), "Close_NQ": pa.array(price+.005, type=pa.float64()),
        "Volume_NQ": pa.array(np.ones(len(times), dtype=np.uint64), type=pa.uint64()),
        "symbol": pa.array(["NQ.v.0"]*len(times), type=pa.large_string()),
        "instrument_id": pa.array(np.ones(len(times), dtype=np.uint32), type=pa.uint32()),
    })
    path = root/"data/synthetic.parquet"
    pq.write_table(table, path)
    manifest = {"manifest_version":2,"data_schema_version":1,"dataset_id":"NQ_CONTINUOUS_WEEKDAYS", "release_id":"synthetic_test",
                "status":"published_with_warnings", "build":{"target_end_exclusive_utc":end+"T00:00:00Z"},
                "files":{"5m":{"path":path.name,"interval":"5m","data_schema_version":1,"validation_status":"passed",
                                  "rows":len(times),"size_bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "first_timestamp":times[0].isoformat(),"last_timestamp":times[-1].isoformat()}},
                "validation":{"status":"passed_with_warnings","structural_error_count":0,"warning_event_count":3,
                              "warning_counts":{"missing_minute":3},"warning_acceptance":{"required":True,"accepted":True,"mode":"cli_flag","accepted_at_utc":"2026-04-13T00:00:00Z"}}}
    (manifest_dir/"manifest.json").write_text(json.dumps(manifest))
    config = yaml.safe_load(Path("tests/fixtures/temporal_run.yml").read_text())
    config["data"]["path"] = "data/synthetic.parquet"
    config["ppo"].update(device="cpu", hidden_sizes=[8], n_steps=256, batch_size=64, n_epochs=1)
    config["artifacts"]["training_metrics"]["every_steps"] = 256
    config_path = root/"protocol.yml"
    config_path.write_text(yaml.safe_dump(config))
    (root/".gitignore").write_text("data/\nartifacts/\n__pycache__/\n")
    for args in (["init","-q","-b","master"], ["add","."], ["-c","user.name=Test","-c","user.email=test@local","commit","-qm","synthetic fixture"]):
        subprocess.run(["git","-C",str(root),*args],check=True)
    return root, config_path


@pytest.mark.parametrize("outcome", ["accept", "exhaust_first", "exhaust_second"])
def test_manual_stage_one_then_three_checkpoint_cycles(database, tmp_path, capsys, monkeypatch, outcome):
    from train_and_eval.training.service import train_ppo_run
    from train_and_eval.walk_forward.service import prepare
    factory, engine = database
    root, stage_one_config = synthetic_project(tmp_path, end="2026-05-11")
    base = yaml.safe_load(stage_one_config.read_text())
    base["run"] = {"name": "manual_stage_one", "seed": 1}
    base["data"] = {"path": "data/synthetic.parquet", "alignment": "trim_start",
                    "train_range": {"start": "2026-01-05T00:00:00Z", "end": "2026-03-02T00:00:00Z"},
                    "validation_range": {"start": "2026-03-02T00:00:00Z", "end": "2026-03-09T00:00:00Z"}}
    base["evaluation"]["training_mode"] = "scheduled"
    base_path = tmp_path / "base.yml"
    base_path.write_text(yaml.safe_dump(base))
    trained = train_ppo_run(factory, config_path=base_path, project_root=root,
                            data_directory=root/"data", manifest_path=root/"train_and_eval/market_data/manifest.json")
    checkpoint_id = trained.checkpoint.checkpoint_id
    with factory() as session:
        assert len(list(session.scalars(select(Run)))) == 1  # Stage one never launches stage two.
    protocol_path = tmp_path / "stage_two.yml"
    definition = {"schema_version": 3, "name": "manual_stage_two", "seed": 1,
                  "source_checkpoint_id": checkpoint_id, "bootstrap_epochs": 1,
                  "grid": {name: [base[name.split(".")[0]][name.split(".")[1]]] for name in GRID_FIELDS}}
    definition["grid"].update({"ppo.learning_rate": [.00005, .000075, .0001], "environment.turnover_penalty": [.001]})
    protocol_path.write_text(yaml.safe_dump(definition))
    protocol, plan = prepare(protocol_path, root, factory)
    assert plan["cycles"][0]["update"] == protocol.run.data.validation_range.model_dump()
    assert plan["cycles"][0]["validation"]["start"] == "2026-03-09T00:00:00+00:00"
    # Real training/checkpoint I/O and PostgreSQL persistence; controlled validation
    # scores exercise selection branches independently of stochastic policy quality.
    from train_and_eval.walk_forward import service
    from train_and_eval.database.models import WalkForwardStudy
    original_stage, original_evaluation = service._run_stage, service._evaluation
    def scored_stage(factory, protocol, cycle, role, source, root, progress, candidate_order=0):
        run_id, cp_id = original_stage(factory, protocol, cycle, role, source, root, progress, candidate_order)
        if role == "candidate":
            fail = outcome == "exhaust_first" or (outcome == "exhaust_second" and cycle.number == 2)
            with factory() as session:
                evaluation = session.scalar(select(Evaluation).where(Evaluation.checkpoint_id == cp_id))
                evaluation.balanced_score = 10. if fail or candidate_order == 0 else 11.
                session.commit()
        return run_id, cp_id
    def scored_reference(*args, **kwargs):
        result = original_evaluation(*args, **kwargs)
        from train_and_eval.database.models import EvaluationDataScope
        if args[3] == EvaluationDataScope.CUSTOM_RANGE:
            with factory() as session:
                session.get(Evaluation, result).balanced_score = 10.
                session.commit()
        return result
    monkeypatch.setattr(service, "_run_stage", scored_stage)
    monkeypatch.setattr(service, "_evaluation", scored_reference)
    if outcome == "accept":
        def interrupt_between_attempts(*args, **kwargs):
            if args[-1] == 1:
                raise RuntimeError("simulated interruption between trials")
            return scored_stage(*args, **kwargs)
        monkeypatch.setattr(service, "_run_stage", interrupt_between_attempts)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            execute(factory, protocol_path, project_root=root, max_cycles=1, live=False)
        with factory() as session:
            interrupted = session.scalar(select(WalkForwardCycle).where(WalkForwardCycle.number == 1))
            assert len(interrupted.selection["candidates"]) == 1
            assert interrupted.selected_checkpoint_id is None
        monkeypatch.setattr(service, "_run_stage", scored_stage)
    study_id = execute(factory, protocol_path, project_root=root, max_cycles=1, live=False)
    if outcome != "accept":
        if outcome == "exhaust_second":
            execute(factory, protocol_path, project_root=root, max_cycles=3, live=False)
        with factory() as session:
            failed_number = 1 if outcome == "exhaust_first" else 2
            stopped = session.scalar(select(WalkForwardCycle).where(WalkForwardCycle.number == failed_number))
            assert stopped.status == "stopped"
            assert stopped.selected_checkpoint_id is None
            assert stopped.refit_checkpoint_id is None and stopped.test_evaluation_id is None
            assert len(stopped.selection["candidates"]) == 3
            assert stopped.selection["exhausted"] is True
            assert session.get(WalkForwardStudy, study_id).status == "stopped"
            run_count = len(list(session.scalars(select(Run))))
        assert execute(factory, protocol_path, project_root=root, max_cycles=3, live=False) == study_id
        with factory() as session:
            assert len(list(session.scalars(select(Run)))) == run_count
        report = generate_report(factory, study_id, project_root=root)
        summary = json.loads((report.parent/"summary.json").read_text())
        assert summary["status"] == "stopped"
        assert summary["completed_tests"] == (failed_number - 1)
        attempts = json.loads((report.parent/"attempts.json").read_text())
        assert len(attempts[-1]["selection"]["candidates"]) == 3
        assert len(pd.read_csv(report.parent/"validation.csv")) == (4 if failed_number == 1 else 7)
        assert "Grid exhausted" in capsys.readouterr().out
        return
    with factory() as session:
        first = session.scalar(select(WalkForwardCycle).where(WalkForwardCycle.number == 1))
        first_test_id = first.test_evaluation_id
        first_test_return = session.get(Evaluation, first_test_id).agent_return
    assert execute(factory, protocol_path, project_root=root, max_cycles=3, live=False) == study_id
    with factory() as session:
        cycles = list(session.scalars(select(WalkForwardCycle).order_by(WalkForwardCycle.number)))
        completed = [c for c in cycles if c.status == "completed"]
        assert len(completed) == 3
        runs = list(session.scalars(select(Run)))
        assert len(runs) == 10
        for i, cycle in enumerate(completed):
            expected = checkpoint_id if i == 0 else completed[i-1].selected_checkpoint_id
            assert cycle.source_checkpoint_id == expected
            candidates = [r for r in runs if r.cycle_id == cycle.id and r.stage_role == "candidate"]
            assert len(candidates) == 2
            assert cycle.selection["winner"]["candidate_order"] == 1
            assert cycle.selection["reference_score"] == 10.
            assert all(r.source_checkpoint_id == expected for r in candidates)
            assert len({r.stage_summary["initial_policy_sha256"] for r in candidates}) == 1
            assert all(r.window_metadata["alignment"] == "prepend" for r in candidates)
            refit = next(r for r in runs if r.cycle_id == cycle.id and r.stage_role == "refit")
            assert refit.source_checkpoint_id == cycle.selected_checkpoint_id
            assert refit.window_metadata["train"]["end_index"] <= cycle.plan["test_rows"]["start_index"]
            assert refit.validation_rows == 0
            assert refit.stage_summary["optimizer_steps"] > 0
            assert refit.normalized_config_json["ppo"]["learning_rate"] == .000075
            assert candidates[0].normalized_config_json["ppo"]["learning_rate"] == .00005
            test = session.get(Evaluation, cycle.test_evaluation_id)
            assert test.checkpoint_id == cycle.refit_checkpoint_id
        assert session.get(Evaluation, first_test_id).agent_return == first_test_return
        before_count = len(list(session.scalars(select(Evaluation))))
        forbidden = completed[0].refit_checkpoint_id
    assert execute(factory, protocol_path, project_root=root, max_cycles=3, live=False) == study_id
    with factory() as session:
        assert len(list(session.scalars(select(Run)))) == 10
        assert len(list(session.scalars(select(Evaluation)))) == before_count
    # The aggregate report must reproduce a missing TEST trajectory using its own bounds.
    from train_and_eval.reporting.artifacts import evaluation_artifact_directory
    with factory() as session:
        test = session.get(Evaluation, first_test_id)
        cp = session.get(Checkpoint, test.checkpoint_id)
        trajectory = evaluation_artifact_directory(root,"artifacts",cp.run_id,test.id)/"trajectory.parquet"
    trajectory.unlink()
    report = generate_report(factory, study_id, project_root=root)
    assert report.is_file() and trajectory.is_file()
    summary = json.loads((report.parent/"summary.json").read_text())
    assert summary["completed_tests"] == 3
    assert summary["capitalization"] is False
    assert summary["agent"]["total_pnl_pln"] == pytest.approx(sum(pd.read_csv(report.parent/"tests.csv").agent_return)*1000)
    assert json.loads((report.parent/"plan.json").read_text())["stage_one_source"]["checkpoint_id"] == checkpoint_id
    from train_and_eval.walk_forward.progress import WalkForwardProgress
    from train_and_eval.walk_forward.service import _restore_progress
    from io import StringIO
    restored = WalkForwardProgress(protocol, plan, stream=StringIO(), live=False)
    with factory() as session:
        saved_cycles = list(session.scalars(select(WalkForwardCycle).order_by(WalkForwardCycle.number)))
    _restore_progress(factory, restored, saved_cycles)
    assert sum(restored.done["base"].values()) == 6
    assert sum(restored.done["refit"].values()) == 12
    assert sum(restored.done["validation"].values()) == 24
    assert sum(restored.done["reference"].values()) == 12
    assert sum(restored.done["test"].values()) == 3
    assert len(restored.results["test"]) == 3
    assert len(restored.results["validation"]) == 3
    output = capsys.readouterr().out
    assert "CYCLE COMPLETED" in output
    assert "Metric" in output and "DD" in output
    assert "PPO TRAINING RESOLUTION" not in output
    assert "not run yet" not in output

    changed_path = tmp_path / "changed.yml"
    changed_path.write_text(yaml.safe_dump({**definition, "grid": {**definition["grid"], "ppo.learning_rate": [.0001]}}))
    with pytest.raises(RuntimeError, match="changed"):
        execute(factory, changed_path, project_root=root, max_cycles=3, live=False)
    protocol_path.write_text(yaml.safe_dump({**definition,"source_checkpoint_id":forbidden}))
    with pytest.raises(ValueError, match="Stage one|Stage-one"):
        prepare(protocol_path, root, factory)
