import pytest
from pydantic import ValidationError
from pathlib import Path
import yaml
from train_and_eval.run_config import RunConfig
from train_and_eval.walk_forward.config import CheckpointWalkForwardConfig, build_plan, stage_config
from tests.test_walk_forward_windows import frame_between
from tests.test_stage_two_grid import complete_grid


def test_checkpoint_plan_starts_after_stage_one_validation():
    base = RunConfig.model_validate(yaml.safe_load(Path('tests/fixtures/temporal_run.yml').read_text()))
    protocol = CheckpointWalkForwardConfig(name='stage_two',seed=1,source_checkpoint_id=17, grid=complete_grid() | {'ppo.batch_size': [64, 256]}).model_copy(update={'run':base})
    frame = frame_between('2010-06-07','2026-09-09')
    frame.attrs['source_path'] = base.data.path
    manifest = {'files': {'5m': {'sha256': 'a'*64,'first_timestamp':frame.DT.iloc[0].isoformat(),'last_timestamp':frame.DT.iloc[-1].isoformat()}},
                'release_id':'test','build':{'target_end_exclusive_utc':'2026-09-09T00:00:00Z'}}
    source = {'checkpoint_id':17,'data_sha256':'a'*64,'train_range':base.data.train_range.model_dump(),
              'validation_range':base.data.validation_range.model_dump()}
    plan = build_plan(protocol, frame, manifest, source)
    first = plan['cycles'][0]
    assert first['update'] == source['validation_range']
    assert first['validation']['start'] == '2019-07-01T00:00:00+00:00'
    assert first['test']['start'] == '2019-07-29T00:00:00+00:00'
    assert len(plan['cycles']) == 371
    windows = plan['cycles'][1]['grid_windows']
    assert len(windows) == 2
    assert windows[0]['candidate_window']['validation'] == windows[1]['candidate_window']['validation']
    assert windows[0]['candidate_window']['prepended_steps'] == 32
    assert windows[1]['candidate_window']['prepended_steps'] == 96
    assert first['candidate_window']['alignment'] == 'prepend'
    for prev, current in zip(plan['cycles'],plan['cycles'][1:]):
        assert current['update']['start'] == prev['validation']['start']
        assert current['test']['start'] == prev['test']['end']
    cfg = stage_config(protocol,first,role='candidate',source=('stage_one',17))
    assert cfg.continuation.checkpoint == '17'
    assert cfg.training.duration_amount == protocol.bootstrap_epochs
    with pytest.raises(ValueError,match='identity'):
        build_plan(protocol,frame,manifest,{**source,'data_sha256':'b'*64})


def test_stage_two_requires_checkpoint_and_forbids_percentages_or_template():
    raw = dict(name='stage_two',seed=1,source_checkpoint_id=17,grid=complete_grid())
    for changes in ({'source_checkpoint_id':None},{'initial_time_fraction':.5},{'run':None}):
        with pytest.raises(ValidationError):
            CheckpointWalkForwardConfig(**(raw | changes))
