"""Opt-in PostgreSQL test. Uses and removes only its own randomly named schema.

PPO_WALK_FORWARD_TEST_DATABASE_URL must point to a disposable test database.
No connection to the application DATABASE_URL is made unless explicitly supplied
through that separate variable. The ordinary suite skips this test.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from alembic import command
from alembic.config import Config
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url
import yaml

from train_and_eval.database.models import Checkpoint, Evaluation, Run, WalkForwardCycle
from train_and_eval.database.session import create_session_factory
from train_and_eval.walk_forward.service import execute
from train_and_eval.walk_forward.reporting import generate_report


@pytest.fixture
def database(monkeypatch):
    url = os.environ.get("PPO_WALK_FORWARD_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set PPO_WALK_FORWARD_TEST_DATABASE_URL for the isolated PostgreSQL integration test")
    schema = "wf_test_" + uuid.uuid4().hex
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    # Startup search_path confines the migration and every stage to this test schema.
    scoped = make_url(url).update_query_dict({"options": f"-csearch_path={schema}"}).render_as_string(hide_password=False)
    engine = create_engine(scoped, pool_pre_ping=True)
    monkeypatch.setenv("DATABASE_URL", scoped)
    try:
        command.upgrade(Config("alembic.ini"), "head")
        yield create_session_factory(engine), engine
    finally:
        engine.dispose()
        with admin.connect() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def synthetic_project(tmp_path):
    root = tmp_path/"project"
    root.mkdir()
    (root/"data").mkdir()
    manifest_dir = root/"train_and_eval/market_data"
    manifest_dir.mkdir(parents=True)
    times = pd.date_range("2026-01-05", "2026-04-13", freq="5min", tz="UTC", inclusive="left")
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
                "status":"published_with_warnings", "build":{"target_end_exclusive_utc":"2026-04-13T00:00:00Z"},
                "files":{"5m":{"path":path.name,"interval":"5m","data_schema_version":1,"validation_status":"passed",
                                  "rows":len(times),"size_bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "first_timestamp":times[0].isoformat(),"last_timestamp":times[-1].isoformat()}},
                "validation":{"status":"passed_with_warnings","structural_error_count":0,"warning_event_count":3,
                              "warning_counts":{"missing_minute":3},"warning_acceptance":{"required":True,"accepted":True,"mode":"cli_flag","accepted_at_utc":"2026-04-13T00:00:00Z"}}}
    (manifest_dir/"manifest.json").write_text(json.dumps(manifest))
    config = yaml.safe_load(Path("configs/walk_forward/nq5m_v1_seed1.yml").read_text())
    config["name"] = "synthetic_walk_forward"
    config["run"]["data"]["path"] = "data/synthetic.parquet"
    config["run"]["ppo"].update(device="cpu", hidden_sizes=[8], n_steps=256, batch_size=64, n_epochs=1)
    config["run"]["artifacts"]["training_metrics"]["every_steps"] = 256
    config_path = root/"protocol.yml"
    config_path.write_text(yaml.safe_dump(config))
    (root/".gitignore").write_text("data/\nartifacts/\n__pycache__/\n")
    for args in (["init","-q","-b","master"], ["add","."], ["-c","user.name=Test","-c","user.email=test@local","commit","-qm","synthetic fixture"]):
        subprocess.run(["git","-C",str(root),*args],check=True)
    return root, config_path


def test_three_cycles_migrations_lineage_resume_and_replay(database, tmp_path):
    factory, engine = database
    root, config = synthetic_project(tmp_path)
    study_id = execute(factory, config, project_root=root, max_cycles=1, live=False)
    with factory() as session:
        first = session.scalar(select(WalkForwardCycle).where(WalkForwardCycle.number == 1))
        first_test_id = first.test_evaluation_id
        first_test_return = session.get(Evaluation, first_test_id).agent_return
    assert execute(factory, config, project_root=root, max_cycles=3, live=False) == study_id
    with factory() as session:
        cycles = list(session.scalars(select(WalkForwardCycle).order_by(WalkForwardCycle.number)))
        runs = list(session.scalars(select(Run)))
        assert len(cycles) == 3 and all(c.status == "completed" for c in cycles)
        assert len(runs) == 12
        for i, cycle in enumerate(cycles):
            candidates = [r for r in runs if r.cycle_id == cycle.id and r.stage_role == "candidate"]
            assert len(candidates) == 3
            assert len({r.stage_summary["initial_policy_sha256"] for r in candidates}) == 1
            previous = None if i == 0 else cycles[i-1].selected_checkpoint_id
            assert cycle.source_checkpoint_id == previous
            assert all(r.source_checkpoint_id == previous for r in candidates)
            refit = next(r for r in runs if r.cycle_id == cycle.id and r.stage_role == "refit")
            assert refit.source_checkpoint_id == cycle.selected_checkpoint_id
            assert refit.validation_rows == 0
            assert refit.stage_summary["optimizer_steps"] > 0
            test = session.get(Evaluation, cycle.test_evaluation_id)
            assert test.checkpoint_id == cycle.refit_checkpoint_id
            assert refit.window_metadata["train"]["end_index"] <= test.evaluation_start_index
        assert session.get(Evaluation, first_test_id).agent_return == first_test_return
        before_count = len(list(session.scalars(select(Evaluation))))
    assert execute(factory, config, project_root=root, max_cycles=3, live=False) == study_id
    with factory() as session:
        assert len(list(session.scalars(select(Run)))) == 12
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
    changed = yaml.safe_load(config.read_text())
    changed["refit_learning_rate"] = .0001
    changed_path = tmp_path/"changed.yml"
    changed_path.write_text(yaml.safe_dump(changed))
    with pytest.raises(RuntimeError, match="changed"):
        execute(factory, changed_path, project_root=root, max_cycles=3, live=False)
