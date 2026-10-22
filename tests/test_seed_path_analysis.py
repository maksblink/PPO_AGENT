import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('seed_paths', Path(__file__).resolve().parents[1]/'extra_tools/analysis/ppo_analyze_grid.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def sample():
    frame = pd.DataFrame(dict(timestamp=pd.date_range('2020-01-29', periods=4, freq='D', tz='UTC'),
        execution_index=[10,11,12,13], position=[1,1,0,1],
        agent_equity=[-.01,.02,.01,.03], always_long_equity=[.01,.02,.03,.04],
        fee_cost=[0.,0.,0.,0.], swap_cost=[0.,0.,0.,0.], trade_cost=[0.,0.,0.,0.]))
    events = pd.DataFrame(dict(event=['CLOSE_LONG','CLOSE_LONG'],
        timestamp=[frame.timestamp.iloc[1],frame.timestamp.iloc[3]],
        net_return=[-.01,.04], gross_return=[-.01,.04], entry_fee=[0.,0.],
        close_fee=[0.,0.], swap_cost=[0.,0.], bars_held=[1,2]))
    expected = dict(agent_return=.03, always_long_return=.04,
        agent_max_drawdown=-.01, market_exposure=.75, round_trips=2)
    return frame, events, expected


class SeedPathTests(unittest.TestCase):
    def test_first_bar_and_calendar_sums(self):
        d,_ = m.path_temporal_summary(*sample(), True)
        self.assertAlmostEqual(sum(p['agent_return'] for p in d['periods']['monthly']), .03)
        self.assertAlmostEqual(d['periods']['monthly'][0]['agent_return'], .01)
        self.assertEqual(d['drawdown']['peak'], 'before first evaluated bar')
        self.assertAlmostEqual(d['drawdown']['value'], -.01)

    def test_concentration_denominator_is_positive_pnl(self):
        d,_ = m.path_temporal_summary(*sample(), True)
        c=d['trade_summary']['concentration'][0]
        self.assertAlmostEqual(c['share_of_positive_trade_pnl'], 1.)
        self.assertAlmostEqual(c['total_minus_contribution'], -.01)

    def test_incomplete_trade_accounting_rejected(self):
        f,e,x=sample(); e.loc[1,['net_return','gross_return']]=.05
        with self.assertRaisesRegex(ValueError,'closed-trade sum'):
            m.path_temporal_summary(f,e,x,True)

    def test_trade_count_rejected(self):
        f,e,x=sample(); x['round_trips']=3
        with self.assertRaisesRegex(ValueError,'trade count'):
            m.path_temporal_summary(f,e,x,True)

    def test_cost_reconciliation_rejected(self):
        f,e,x=sample(); f.loc[0,'fee_cost']=.001
        with self.assertRaisesRegex(ValueError,'Per-bar trade costs'):
            m.path_temporal_summary(f,e,x,True)

    def test_index_gaps_rejected(self):
        f,e,x=sample(); f.loc[3,'execution_index']=15
        with self.assertRaisesRegex(ValueError,'Noncontiguous'):
            m.path_temporal_summary(f,e,x,True)

    def test_no_trades_and_flat_equity(self):
        f,e,x=sample(); f['position']=0; f['agent_equity']=0.
        x.update(agent_return=0.,agent_max_drawdown=0.,market_exposure=0.,round_trips=0)
        d,_=m.path_temporal_summary(f,e.iloc[:0],x,True)
        self.assertIsNone(d['trade_summary']['median_bars_held'])

    def test_seed_alignment_and_behavior(self):
        _,a=m.path_temporal_summary(*sample(),True)
        b=a.copy(); b['position']=0
        result=m.path_compare_positions(a,b)
        self.assertEqual(result['position_agreement'],.25)
        self.assertEqual(result['only_reference_exposed'],.75)
        b.loc[0,'execution_index']=99
        with self.assertRaisesRegex(ValueError,'timelines differ'):
            m.path_compare_positions(a,b)

    def test_offline_chart(self):
        _,a=m.path_temporal_summary(*sample(),True)
        self.assertIn('data:image/png;base64,',m.path_chart(a,a))

    def test_nonfinite_rejected(self):
        f,e,x=sample(); f.loc[1,'agent_equity']=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            m.path_temporal_summary(f,e,x,True)


if __name__=='__main__':
    unittest.main()
