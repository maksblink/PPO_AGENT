import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name):
 spec=importlib.util.spec_from_file_location(name, ROOT/'extra_tools'/f'{name}.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

g=load('ppo_analyze_grid');p=g

class IntegrationTests(unittest.TestCase):
 def test_local_selection_validation(self):
  with tempfile.TemporaryDirectory() as tmp:
   file=Path(tmp)/'selection.json'
   file.write_text(json.dumps({'reference_run_ids':[5,7,5]}))
   self.assertEqual(g.load_detail_ids(file),[5,7])
   for payload in ({'reference_run_ids':[True]}, {'reference_run_ids':[]}, {'ids':[5]}):
    file.write_text(json.dumps(payload))
    with self.assertRaises(ValueError):g.load_detail_ids(file)

 def test_tabs_preserve_grid_paths_and_pairs(self):
  grid='<html><style></style><h1>Grid</h1><section id="s1">Integrity</section></html>'
  details='<html><section>all stages</section><section>pair trajectories</section></html>'
  d={'notes':['known VAL'], 'detailed_pairs':[{'reference':{'id':1},'replay':{'id':2}}]}
  rendered=g.tabbed_report(grid,d,details)
  for text in ('all stages','pair trajectories','Integrity','Full paths','VAL #1 / #2','role="tablist"','ArrowRight','noscript'):
   self.assertIn(text,rendered)
  self.assertEqual(rendered.count('<html>'),1)

 def test_no_details_keeps_explicit_instruction(self):
  result=g.tabbed_report('<html><h1>Grid</h1></html>')
  self.assertIn('--details-config',result)

 def test_build_rejects_missing_pair_before_database(self):
  source={'seed_comparison':{'readiness':'PASS','pairs':[]}}
  with patch.object(p,'path_read_snapshot') as db:
   with self.assertRaisesRegex(ValueError,'Expected one pair'):
    p.path_build_analysis(ROOT,source,[100])
   db.assert_not_called()

 def test_integrated_detail_builder_without_writes(self):
  spec=importlib.util.spec_from_file_location('fixture_sample',ROOT/'tests/test_seed_path_analysis.py')
  fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
  sample=fixture.sample
  metrics=dict(agent_return=.03,always_long_return=.04,agent_max_drawdown=-.01,
               market_exposure=.75,round_trips=2,balanced_score=.02)
  a=dict(id=1,name='fresh_seed1',seed=1,stage=0,TRAIN=metrics,VAL=metrics)
  b=dict(a,id=2,name='fresh_seed2',seed=2)
  comp=dict(readiness='PASS',base_seed=1,replay_seed=2,pairs=[{'reference':a,'replay':b}],
     chains=[dict(reference_root=a['name'],reference_run_names=[a['name']],replay_run_names=[b['name']])])
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   d,f=p.path_temporal_summary(*sample(),True)
   with patch.object(p,'path_read_snapshot',return_value={1:{},2:{}}), patch.object(p,'path_read_artifacts',return_value=(d,f)):
    result,html=p.path_build_analysis(root,{'seed_comparison':comp},[1])
   self.assertEqual(list(root.iterdir()),[])
   self.assertEqual(len(result['chain_runs']),2)
   self.assertEqual(len(result['detailed_pairs']),1)
   self.assertIn('attribution',result['detailed_pairs'][0]['details']['reference'])
   self.assertIn('VAL #1 / #2',g.tabbed_report('<html><h1>Grid</h1></html>',result,html))
   json.dumps(result,allow_nan=False)

class ModeTests(unittest.TestCase):
 def test_full_requires_selection(self):
  with patch('sys.argv',['tool','--mode','full','--all-runs','--compare-seeds','1','2']):
   with self.assertRaisesRegex(ValueError,'Full mode requires'):
    g.main()

 def test_grid_rejects_trajectory_selection(self):
  with patch('sys.argv',['tool','--mode','grid','--all-runs','--detail-run-id','1']):
   with self.assertRaisesRegex(ValueError,'Grid mode excludes'):
    g.main()

 def test_paths_mode_reads_source_and_saves_without_grid_snapshot(self):
  import os
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'run_search_queue.py').touch()
   source=root/'source.json';source.write_text('{}')
   details={'notes':[], 'detailed_pairs':[{'reference':{'id':1},'replay':{'id':2}}]}
   rendered='<html><section>path</section><section>pair</section></html>'
   old=Path.cwd()
   try:
    os.chdir(root)
    with patch('sys.argv',['tool','--mode','paths','--report',str(source),'--detail-run-id','1']), patch.object(g,'path_build_analysis',return_value=(details,rendered)) as build, patch.object(g,'snapshot') as snapshot:
     g.main();snapshot.assert_not_called();build.assert_called_once()
    outputs=list((root/'artifacts/queue_analysis').glob('*/report.json'))
    self.assertEqual(len(outputs),1)
    self.assertEqual(json.loads(outputs[0].read_text())['metadata']['mode'],'paths')
   finally:os.chdir(old)

if __name__=='__main__':unittest.main()
