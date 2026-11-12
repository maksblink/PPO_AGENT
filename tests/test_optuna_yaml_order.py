"""Presentation-only ordering must never alter run settings."""
from copy import deepcopy
import unittest
import yaml
from train_and_eval.optuna.yaml_order import FIELD_ORDER, ordered_run_config


class TestOptunaYamlOrder(unittest.TestCase):
    def test_all_sections_and_nested_fields_follow_grid_order(self):
        config = {}
        for path, keys in FIELD_ORDER.items():
            target = config
            for part in path:
                target = target.setdefault(part, {})
            for key in reversed(keys):
                target.setdefault(key, {})
        ordered = ordered_run_config(config)
        for path, keys in FIELD_ORDER.items():
            target = ordered
            for part in path:
                target = target[part]
            self.assertEqual(list(target), list(keys))

    def test_values_and_input_are_unchanged(self):
        config = {'ppo': {'hidden_sizes': [222, 193, 242], 'learning_rate': .000119228,
                          'normalize_advantage': True, 'target_kl': None},
                  'data': {'train_range': {'end': '2020-01-01', 'start': '2010-01-01'},
                           'train_ratio': None}, 'config_schema_version': 1}
        before = deepcopy(config)
        output = ordered_run_config(config)
        self.assertEqual(output, before)
        self.assertEqual(list(output['data']['train_range']), ['start', 'end'])
        self.assertEqual(yaml.safe_load(yaml.safe_dump(output, sort_keys=False)), before)
        output['ppo']['hidden_sizes'][0] = 999
        self.assertEqual(config, before)
        self.assertEqual(list(config), ['ppo', 'data', 'config_schema_version'])

    def test_unknown_fields_are_preserved(self):
        config = {'future': {'b': [1, 2], 'a': None}, 'ppo': {'future_ppo': 7, 'gamma': .9},
                  'config_schema_version': 1}
        result = ordered_run_config(config)
        self.assertEqual(result, config)
        self.assertEqual(list(result), ['config_schema_version', 'ppo', 'future'])
        self.assertEqual(list(result['ppo']), ['gamma', 'future_ppo'])

    def test_fresh_does_not_gain_resume_fields(self):
        self.assertEqual(ordered_run_config({'continuation': {'mode': 'fresh'}}),
                         {'continuation': {'mode': 'fresh'}})

    def test_resume_order_and_idempotence(self):
        config = {'continuation': {'checkpoint': '123', 'source_run': 'parent', 'mode': 'resume'}}
        result = ordered_run_config(config)
        self.assertEqual(list(result['continuation']), ['mode', 'source_run', 'checkpoint'])
        self.assertEqual(yaml.safe_dump(result, sort_keys=False),
                         yaml.safe_dump(ordered_run_config(result), sort_keys=False))


if __name__ == '__main__':
    unittest.main()
