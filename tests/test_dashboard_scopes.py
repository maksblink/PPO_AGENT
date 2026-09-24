from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from train_and_eval.dashboard import data as dashboard


@pytest.fixture
def tables():
    runs = pd.DataFrame([
        {"id": i, "name": f"run_{i}", "status": "completed", "seed": i,
         "config": {"ppo": {"gamma": .9, "n_epochs": 3, "learning_rate": .001}}}
        for i in (1, 2)
    ])
    checkpoints = pd.DataFrame([{"id": i, "run_id": i} for i in (1, 2)])
    evaluations = []
    for scope, scale in (("run_validation", 1), ("run_training", -1)):
        for i in (1, 2):
            evaluations.append(dict(id=len(evaluations)+1, checkpoint_id=i,
                data_scope=scope, status="completed", trigger="final",
                balanced_score=scale*i/10, agent_return=scale*i/10,
                agent_max_drawdown=-.1, market_exposure=.5, round_trips=10,
                profit_factor=1., win_rate=.5))
    return dict(runs=runs, checkpoints=checkpoints,
                evaluations=pd.DataFrame(evaluations), training_metrics=pd.DataFrame())


@pytest.fixture
def mock_database(monkeypatch, tables):
    monkeypatch.setattr(dashboard, "create_database_engine",
                        lambda: SimpleNamespace(dispose=lambda: None))
    monkeypatch.setattr(dashboard, "_read_table", lambda engine, name: tables[name].copy())
    return tables


def test_loader_switches_all_evaluations_and_explorer(mock_database):
    for scope, expected in (("run_validation", [.1, .2]), ("run_training", [-.1, -.2])):
        data = dashboard.load_dashboard_data(scope)
        assert set(data.evaluations.data_scope) == {scope}
        assert data.explorer["eval.agent_return"].tolist() == expected
        assert data.explorer.run_id.tolist() == [1, 2]


def test_missing_final_never_falls_back_to_scheduled_failed_or_other_scope(mock_database):
    evaluations = mock_database["evaluations"]
    evaluations.loc[evaluations.data_scope == "run_training", "trigger"] = "scheduled"
    evaluations.loc[evaluations.id == 4, ["trigger", "status"]] = ["final", "failed"]
    data = dashboard.load_dashboard_data("run_training")
    assert "eval.agent_return" not in data.explorer or data.explorer["eval.agent_return"].isna().all()
    assert data.explorer.run_id.tolist() == [1, 2]
    assert len(data.evaluations) == 2  # still available in Run Detail


def test_empty_database(mock_database):
    for name, frame in mock_database.items():
        mock_database[name] = frame.iloc[:0]
    assert dashboard.load_dashboard_data("run_training").explorer.empty


def test_dashboard_rejects_unknown_scope():
    with pytest.raises(ValueError):
        dashboard.load_dashboard_data("other")


def test_global_switch_updates_metrics_and_run_detail(mock_database):
    from streamlit.testing.v1 import AppTest
    app = Path(__file__).resolve().parents[1] / "train_and_eval/dashboard/app.py"
    at = AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception
    assert next(m for m in at.metric if m.label == "Best return").value == "+20.00%"
    at.segmented_control[0].set_value("TRAIN").run()
    assert not at.exception
    assert next(m for m in at.metric if m.label == "Best return").value == "-10.00%"
    assert not any(s.label == "Evaluation scope" for s in at.selectbox)
    for frame in at.dataframe:
        if "data_scope" in frame.value:
            assert set(frame.value.data_scope) == {"run_training"}
    at.segmented_control[0].set_value("VAL").run()
    assert not at.exception
    assert next(m for m in at.metric if m.label == "Best return").value == "+20.00%"


def test_newer_scheduled_or_failed_evaluation_does_not_replace_final(mock_database):
    frame = mock_database["evaluations"]
    scheduled = frame.iloc[[2]].assign(id=100, trigger="scheduled", agent_return=100.)
    failed = frame.iloc[[2]].assign(id=101, status="failed", agent_return=200.)
    mock_database["evaluations"] = pd.concat([frame, scheduled, failed], ignore_index=True)
    data = dashboard.load_dashboard_data("run_training")
    assert data.explorer["eval.agent_return"].tolist() == [-.1, -.2]
    assert len(data.evaluations) == 4


def test_dashboard_with_no_train_evaluations(mock_database):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    st.cache_data.clear()
    frame = mock_database["evaluations"]
    mock_database["evaluations"] = frame.loc[frame.data_scope == "run_validation"]
    app = Path(__file__).resolve().parents[1] / "train_and_eval/dashboard/app.py"
    at = AppTest.from_file(str(app), default_timeout=30).run()
    at.segmented_control[0].set_value("TRAIN").run()
    assert not at.exception
    assert next(m for m in at.metric if m.label == "Best return").value == "—"
    st.cache_data.clear()
