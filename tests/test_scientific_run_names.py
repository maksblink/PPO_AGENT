"""Naming, immutable checkpoints and cross-operation guards; no database required."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from extra_tools.maintenance.audit_run_names import proposed_names, scientific_token, historical_reference
from extra_tools.maintenance.rename_runs import verify_checkpoints
from extra_tools.maintenance.maintenance_guard import maintenance_guard, activity_guard


class ScientificNames(unittest.TestCase):
    def config(self):
        return {'run': {'name': 'old', 'seed': 1}, 'continuation': {'mode': 'fresh'},
                'ppo': {'hidden_sizes': [768]*4, 'learning_rate': 0.0000025,
                        'gamma': .9, 'gae_lambda': .85, 'ent_coef': .0002, 'n_epochs': 3}}

    def test_exact_tokens(self):
        for value, token in [('0.0000025','2p50000em6'),('.85','8p50000em1'),
                             ('0','0p00000ep0'),('1','1p00000ep0'),('0.123456','1p23456em1')]:
            self.assertEqual(scientific_token(value), token)

    def test_reject_precision_and_range(self):
        for value in ['0.1234567','1e-10','1e10','NaN','Infinity','-0.1']:
            with self.assertRaises(ValueError): scientific_token(value)

    def test_names_and_no_input_mutation(self):
        configs={'old': self.config()}; child=copy.deepcopy(configs['old'])
        child['run']['name']='child';child['continuation']={'mode':'resume','source_run':'old','checkpoint':'final'}
        configs['child']=child; before=copy.deepcopy(configs)
        names=proposed_names(configs)
        tail='_gae8p50000em1_ent2p00000em4_g9p00000em1_w768x4_ne03_s01'
        self.assertEqual(names['old']['new_name'],'lr2p50000em6_f_v00'+tail)
        self.assertEqual(names['child']['new_name'],'lr2p50000em6_r_v01'+tail)
        self.assertEqual(configs,before)

    def test_collision(self):
        with self.assertRaisesRegex(ValueError,'not unique'):
            proposed_names({'a':self.config(),'b':self.config()})

    def test_cycle(self):
        cfg=self.config();cfg['continuation']={'mode':'resume','source_run':'a','checkpoint':'final'}
        with self.assertRaisesRegex(ValueError,'cycle'):proposed_names({'a':cfg})

    def test_entropy_and_gae_chain_mismatch(self):
        for field in ['ent_coef','gae_lambda']:
            a=self.config();b=copy.deepcopy(a)
            b['continuation']={'mode':'resume','source_run':'a','checkpoint':'final'}
            b['ppo'][field]=.1
            with self.assertRaisesRegex(ValueError,'Different '+field):proposed_names({'a':a,'b':b})

    def test_fixed_width_bounds(self):
        for field in ['seed','n_epochs']:
            cfg=self.config();cfg['run' if field=='seed' else 'ppo'][field]=100
            with self.assertRaises(ValueError):proposed_names({'old':cfg})

    def test_checkpoint_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model').write_bytes(b'abc')
            plan={'checkpoint_files':[{'path':'model','sha256':hashlib.sha256(b'abc').hexdigest()}]}
            verify_checkpoints(root,plan)
            (root/'model').write_bytes(b'abd')
            with self.assertRaisesRegex(ValueError,'bytes changed'):verify_checkpoints(root,plan)

    def test_maintenance_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            with maintenance_guard(tmp):
                with self.assertRaises(RuntimeError):
                    with maintenance_guard(tmp): pass

    def test_queue_rename_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            with activity_guard(tmp):
                with activity_guard(tmp): pass
                with self.assertRaises(RuntimeError):
                    with activity_guard(tmp,exclusive=True):pass
            with activity_guard(tmp,exclusive=True):
                with self.assertRaises(RuntimeError):
                    with activity_guard(tmp):pass

    def test_pending_rename_blocks_queue_but_not_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'extra_tools/maintenance/run_rename';p.mkdir(parents=True);(p/'pending.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'Pending rename'):
                with activity_guard(tmp):pass
            with activity_guard(tmp,exclusive=True):pass

    def test_historical_scope(self):
        self.assertTrue(historical_reference('artifacts/search_queues/q/events.jsonl'))
        self.assertFalse(historical_reference('artifacts/runs/00000001/reports/summary.json'))

if __name__=='__main__': unittest.main()
