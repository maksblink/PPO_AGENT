from pathlib import Path
import pandas as pd
import pytest
import yaml
from set_stage_one_ranges import calculate_ranges, ceil_week, confirm_and_write, replace_ranges


def test_calendar_rounds_both_windows_up():
    bounds = calculate_ranges('2010-06-07T00:00:00Z', '2026-09-09T00:00:00Z')
    assert bounds['train_range']['end'] == '2018-07-30T00:00:00+00:00'
    assert bounds['validation_range']['end'] == '2019-07-01T00:00:00+00:00'
    assert ceil_week('2018-07-30T00:00:00Z') == pd.Timestamp('2018-07-30T00:00:00Z')
    assert ceil_week('2018-07-30T00:00:01Z') == pd.Timestamp('2018-08-06T00:00:00Z')


@pytest.mark.parametrize('fraction,split', [(0,.9), (.5,1), ('nan',.9), (.99,.1)])
def test_invalid_or_oversized_ranges_rejected(fraction, split):
    with pytest.raises(ValueError):
        calculate_ranges('2026-01-05T00:00:00Z','2026-04-13T00:00:00Z',fraction,split)


def test_confirm_changes_only_ranges_and_decline_preserves_file(tmp_path):
    original = Path('tests/fixtures/temporal_run.yml').read_text()
    original = original.replace('alignment: trim_start', '# Keep this comment\n  alignment: trim_start')
    path = tmp_path / 'config.yml'
    path.write_text(original)
    ranges = calculate_ranges('2010-06-07T00:00:00Z','2026-09-09T00:00:00Z',.4,.9)
    updated = replace_ranges(original, ranges)
    expected = yaml.safe_load(original)
    expected['data'].update(ranges)
    assert yaml.safe_load(updated) == expected
    assert '# Keep this comment' in updated
    assert not confirm_and_write(path, original, updated, prompt=lambda _: '')
    assert path.read_text() == original
    assert confirm_and_write(path, original, updated, prompt=lambda _: 'tak')
    assert path.read_text() == updated


def test_concurrent_edit_is_not_overwritten(tmp_path):
    path = tmp_path/'config.yml'
    path.write_text('changed')
    with pytest.raises(RuntimeError, match='changed'):
        confirm_and_write(path, 'original', 'replacement', prompt=lambda _: 'yes')
    assert path.read_text() == 'changed'


def test_cli_previews_verified_windows_without_starting_training(monkeypatch, tmp_path, capsys):
    import set_stage_one_ranges as script
    from tests.test_walk_forward_windows import frame_between
    frame = frame_between('2026-01-05','2026-05-11')
    frame.attrs['interval'] = '5m'
    manifest = {'release_id':'preview', 'files':{'5m':{'first_timestamp':frame.DT.iloc[0].isoformat()}},
                'build':{'target_end_exclusive_utc':'2026-05-11T00:00:00Z'}}
    path = tmp_path/'config.yml'
    original = Path('tests/fixtures/temporal_run.yml').read_text()
    path.write_text(original)
    monkeypatch.setattr(script,'load_manifest',lambda _:manifest)
    monkeypatch.setattr(script,'load_market_data',lambda *args,**kwargs:frame)
    writer = script.confirm_and_write
    monkeypatch.setattr(script,'confirm_and_write',lambda p,o,u:writer(p,o,u,prompt=lambda _: 'no'))
    script.main(['--config',str(path),'--project-root',str(tmp_path)])
    assert path.read_text() == original
    output = capsys.readouterr().out
    assert 'Train: 9 weeks; validation: 1 weeks' in output
    assert 'No changes written.' in output
