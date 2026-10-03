from streamlit.testing.v1 import AppTest


def test_database_and_config_seed_filters_are_distinct():
    app = AppTest.from_string('''
import pandas as pd
from train_and_eval.dashboard.analysis import render_grid_analysis
frame = pd.DataFrame({
    "run_id": [1, 2], "run.name": ["first", "second"],
    "run.seed": [1, 2], "run.status": ["completed", "completed"],
    "run.architecture": ["384×3", "384×3"],
    "cfg.normalized_config_json.run.seed": [1, 2],
    "cfg.normalized_config_json.ppo.gamma": [.9, .95],
    "path.root_run_id": [1, 2], "path.parent_run_id": [None, None],
    "path.position": [1, 1], "path.status": ["Complete", "Complete"],
    "eval.id": [1, 2], "eval.agent_return": [.1, .2],
})
render_grid_analysis(frame, "VAL")
''').run()
    assert not app.exception
    fields = app.multiselect(key='ga_filter_fields')
    assert len(fields.options) == len(set(fields.options))
    assert 'run.seed' in fields.value
    assert 'cfg.normalized_config_json.run.seed' in fields.value
    app.multiselect(key='ga_values_run.seed').set_value([1]).run()
    assert not app.exception
    assert app.metric[0].value == '1'
    app.radio(key='ga_kind_cfg.normalized_config_json.run.seed').set_value('Range').run()
    assert not app.exception
