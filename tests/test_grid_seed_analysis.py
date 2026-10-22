"""Pure analysis tests: no database, market data, checkpoints or hardware access."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "extra_tools" / "ppo_analyze_grid.py"
spec = importlib.util.spec_from_file_location("grid_analysis_under_test", SCRIPT)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def run(seed, stage, value, parent=None):
    name = f"run_seed{seed}_stage{stage}"
    metrics = dict(agent_return=value, balanced_score=value-.1,
                   agent_max_drawdown=-.1, market_exposure=.5, round_trips=100,
                   profit_factor=1.1, win_rate=.55, always_long_return=.2,
                   excess_return=value-.2, signature={})
    return dict(id=seed*100+stage, name=name, seed=seed, stage=stage, source=parent,
                cohort=f"seed{seed}", seed_comparison_cohort="same_conditions",
                architecture="8x8", learning_rate=.001, gamma=.9, n_epochs=3,
                continuation_options={} if stage == 0 else {"checkpoint": "final"},
                steps_completed=1024, stopped_early=False, seconds=10,
                TRAIN=copy.deepcopy(metrics), VAL=copy.deepcopy(metrics))


def dataset():
    rows = []
    for seed, vals in ((1, [.2, .4, .3]), (2, [.25, .35, .5])):
        for stage, value in enumerate(vals):
            rows.append(run(seed, stage, value,
                            f"run_seed{seed}_stage{stage-1}" if stage else None))
    return rows


class GridSeedAnalysisTests(unittest.TestCase):
    def compare(self, rows=None, **kwargs):
        return g.compare_seeds(rows or dataset(), {}, {}, 1, 2, **kwargs)

    def test_matching_without_name_pattern_and_full_paths(self):
        rows = dataset()
        mapping = {r['name']: f"arbitrary_{i}" for i, r in enumerate(rows)}
        for r in rows:
            r['source'] = mapping.get(r['source'])
            r['name'] = mapping[r['name']]
        result = self.compare(rows)
        self.assertEqual(result['counts']['matched_stages'], 3)
        self.assertEqual(result['counts']['complete_chains'], 1)

    def test_locked_reference_stage_is_not_replay_optimum(self):
        c = self.compare()['chains'][0]['best_stage']['agent_return']
        self.assertEqual(c['reference_best_stages'], [1])
        self.assertEqual(c['replay_at_reference_best'], [dict(stage=1, value=.35)])
        self.assertEqual(c['replay_best_stages'], [2])

    def test_broken_parent_excludes_descendants(self):
        rows = dataset()
        rows[4]['source'] = rows[0]['name']
        r = self.compare(rows)
        self.assertEqual(r['counts']['matched_stages'], 1)
        self.assertEqual(r['readiness'], 'INCOMPLETE')

    def test_duplicate_scientific_cells_not_arbitrarily_chosen(self):
        rows = dataset()
        duplicate = copy.deepcopy(rows[0]); duplicate['name'] = 'another'
        rows.append(duplicate)
        r = self.compare(rows)
        self.assertEqual(r['counts']['matched_stages'], 0)

    def test_data_conditions_mismatch_excluded(self):
        rows = dataset()
        for r in rows[3:]:
            r['seed_comparison_cohort'] = 'different_data'
        self.assertEqual(self.compare(rows)['counts']['matched_stages'], 0)

    def test_baseline_mismatch_excluded(self):
        rows = dataset()
        rows[3]['VAL']['always_long_return'] = .99
        self.assertEqual(self.compare(rows)['counts']['matched_stages'], 0)

    def test_missing_final_stage_not_complete(self):
        r = self.compare(dataset()[:-1])
        self.assertEqual(r['counts']['matched_stages'], 2)
        self.assertEqual(r['counts']['complete_chains'], 0)

    def test_expected_completely_absent_run_is_reported(self):
        r = g.compare_seeds(dataset(), {}, {'not_in_database': {}}, 1, 2)
        self.assertEqual(r['readiness'], 'INCOMPLETE')
        self.assertEqual(r['counts']['matched_stages'], 0)
        self.assertIn('missing', r['issues'][0]['reason'])

    def test_top_percent_rounds_up_retains_cutoff_ties(self):
        rows = [run(1, i, float(i)) for i in range(101)]
        rows[-2]['VAL']['agent_return'] = 100
        picked, info = g.select_top_percent(rows, 'VAL', 'agent_return', 1)
        self.assertEqual(info['target_count'], 2)
        self.assertEqual(len(picked), 2)
        rows[-3]['VAL']['agent_return'] = 100
        self.assertEqual(len(g.select_top_percent(rows, 'VAL', 'agent_return', 1)[0]), 3)

    def test_signed_drawdown_delta(self):
        rows = dataset()
        rows[3]['VAL']['agent_max_drawdown'] = -.05
        result = self.compare(rows)
        first = result['pairs'][0]
        self.assertAlmostEqual(first['deltas']['VAL']['agent_max_drawdown'], .05)

    def test_exact_best_stage_ties_preserved(self):
        rows = dataset()
        rows[2]['VAL']['agent_return'] = .4
        best = self.compare(rows)['chains'][0]['best_stage']['agent_return']
        self.assertEqual(best['reference_best_stages'], [1, 2])

    def test_training_length_difference_exposed(self):
        rows = dataset(); rows[3]['steps_completed'] = 512
        self.assertTrue(self.compare(rows)['pairs'][0]['different_realized_training_length'])

    def test_html_and_json_with_undefined_profit_factor(self):
        rows = dataset(); rows[3]['VAL']['profit_factor'] = None
        result = self.compare(rows)
        text = g.seed_comparison_html(result)
        self.assertIn('Original candidates at their original stage', text)
        self.assertIn('run_seed2_stage2', text)
        json.dumps(result, allow_nan=False)

    def test_frozen_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            path.write_text(json.dumps({'candidate_selection': [
                {'shortlist': [{'name': 'run_seed1_stage0'}]}]}))
            result = self.compare(selection_report=path)
            self.assertEqual(result['counts']['original_candidates'], 1)
            self.assertEqual(result['counts']['matched_candidates'], 1)
            self.assertIn('sha256', result['selection'])

    def test_end_to_end_snapshot_to_reports_without_database(self):
        from contextlib import redirect_stdout
        from datetime import datetime, timedelta, timezone
        from io import StringIO
        import os
        from unittest.mock import patch
        source = dataset()
        runs, cps, evs, configs = [], [], [], {}
        now = datetime.now(timezone.utc)
        ids = {r['name']: r['id'] for r in source}
        for r in source:
            c = dict(
                ppo=dict(hidden_sizes=[8, 8], learning_rate=.001, gamma=.9,
                         n_epochs=3, device='cuda'),
                run=dict(name=r['name'], seed=r['seed']),
                data=dict(path='unused.parquet'), environment={}, training={},
                evaluation=dict(device='cpu'),
                continuation=(dict(mode='resume', source_run=r['source'], checkpoint='final')
                              if r['source'] else dict(mode='fresh')))
            configs[r['name']] = c
            parent_id = ids.get(r['source'])
            runs.append(dict(id=r['id'], name=r['name'], status='completed',
                parent_name=r['source'], parent_id=parent_id, source_checkpoint_id=parent_id,
                normalized_config_json=c, cycle_id=None, stopped_early=False,
                training_steps_completed=1024, data_sha256='same-data',
                started_at=now, finished_at=now+timedelta(seconds=10)))
            cps.append(dict(id=r['id'], run_id=r['id'], save_reason='final'))
            for number, (scope, data_scope) in enumerate(g.SCOPES.items()):
                evs.append(dict(r[scope], id=r['id']*2+number, run_id=r['id'],
                    checkpoint_id=r['id'], data_scope=data_scope, status='completed',
                    checkpoint_run_step=1024, steps_completed=100, steps_expected=100,
                    data_sha256='same-data', seed=r['seed']))
        expected = {n: c for n, c in configs.items() if c['run']['seed'] == 2}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'run_search_queue.py').write_text('')
            old = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(g, 'snapshot', return_value=(runs, cps, evs, {}, str(now))), \
                     patch.object(g, 'device_metadata', return_value=('GPU', [])), \
                     patch.object(g, 'load_queue', return_value=({}, expected, {})), \
                     patch.object(g.sys, 'argv', ['tool', '--all-runs', '--compare-seeds', '1', '2',
                                                '--comparison-queue', 'example']), \
                     redirect_stdout(StringIO()):
                    g.main()
                reports = list((root/'artifacts/queue_analysis').glob('*/report.json'))
                self.assertEqual(len(reports), 1)
                report = json.loads(reports[0].read_text())
                comparison = report['seed_comparison']
                self.assertEqual(comparison['readiness'], 'PASS')
                self.assertEqual(comparison['counts']['matched_stages'], 3)
                self.assertEqual(len(report['candidate_selection']), 2)
                self.assertTrue(reports[0].with_suffix('.html').is_file())
            finally:
                os.chdir(old)

    def test_recursive_queue_discovery_and_duplicate_stems(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); queues = root/'configs/search_queues'
            (queues/'seed2').mkdir(parents=True)
            manifest = queues/'seed2/example.yml'
            manifest.write_text('configs: []')
            self.assertEqual(g.resolve_queue(root, 'example'), manifest)
            (queues/'example.yaml').write_text('configs: []')
            with self.assertRaisesRegex(ValueError, 'found 2'):
                g.resolve_queue(root, 'example')


if __name__ == '__main__':
    unittest.main()
