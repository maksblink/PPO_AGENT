"""Queue observability checks without CUDA or database services."""
import ast
import contextlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from train_and_eval.queue_execution.monitor import Progress, OutputReader, classify, atomic_json, PREFIX
from train_and_eval.queue_execution.evaluation_gate import evaluation_lock_scope, serialize_evaluation
from train_and_eval.queue_execution import runtime


class MonitorTests(unittest.TestCase):
    def test_skip_blocks_descendants_but_not_completed(self):
        jobs=[NS(name='a',source=None),NS(name='b',source='a'),NS(name='c',source='b'),NS(name='d',source=None)]
        completed,blocked,pending=classify(jobs,{'d':'completed'},{'a'})
        self.assertEqual(completed,{'d'});self.assertEqual(blocked,{'b','c'});self.assertEqual(pending,[])

    def test_periodic_final_and_duplicate_events(self):
        p=Progress([NS(name='a')],{'a':{'evaluation':{'training_mode':'scheduled'}}})
        for trigger,checkpoint in [('scheduled',1),('final',2),('final',2)]:
            p.accept('a',dict(stage='EVAL_VAL',completed=True,trigger=trigger,checkpoint_id=checkpoint))
        self.assertEqual(p.counts()['EVAL_VAL'],{'completed':1,'total':1})
        self.assertEqual(p.evaluations['EVAL_VAL'],2)
        self.assertEqual(p.counts()['TRAIN']['completed'],0)

    def test_no_evaluation_mode_has_zero_targets(self):
        p=Progress([NS(name='a')],{'a':{'evaluation':{'training_mode':'none'}}})
        self.assertEqual(p.counts()['EVAL_VAL']['total'],0)

    def test_reader_drains_real_process_and_bounds_discarded_output(self):
        script="import sys; print('x'*400000); print('@@PPO_QUEUE@@'+ '{\"phase\":\"TRAIN\"}'); print('Traceback: example failure',file=sys.stderr)"
        process=subprocess.Popen([sys.executable,'-c',script],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        reader=OutputReader(process.stdout)
        self.assertEqual(process.wait(timeout=10),0)
        messages=[];reader.finish(messages.append)
        self.assertEqual(messages,[{'phase':'TRAIN'}])
        self.assertLessEqual(sum(map(len,reader.tail)),64*4096)
        self.assertIn('Traceback', ''.join(reader.tail))

    def test_last_state_is_replaced_and_retained(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'worker_1_last.state.json'
            atomic_json(path,{'phase':'TRAIN','run':'a'})
            atomic_json(path,{'phase':'INTERRUPTED','run':'b'})
            self.assertEqual(json.loads(path.read_text()),{'phase':'INTERRUPTED','run':'b'})
            self.assertEqual(len(list(Path(d).iterdir())),1)

    def test_evaluation_counts_only_success_and_preserves_scope(self):
        def evaluate_run_validation_checkpoint(**kwargs):
            if kwargs.get('fail'):raise ValueError('failure')
            return 42
        wrapped=serialize_evaluation(evaluate_run_validation_checkpoint)
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'PPO_QUEUE_EVENTS':'1'}), patch.object(runtime,'phase') as phase:
            with evaluation_lock_scope(Path(d)/'lock'):
                self.assertEqual(wrapped(data_scope='run_training',trigger='final',checkpoint_id=9),42)
                with self.assertRaises(ValueError):wrapped(data_scope='run_validation',trigger='final',checkpoint_id=10,fail=True)
            done=[call for call in phase.call_args_list if call.kwargs.get('completed')]
            self.assertEqual(len(done),1)
            self.assertEqual(done[0].args[0],'EVAL_TRAIN')

    def test_scheduler_preflight_then_two_workers_and_compact_files(self):
        self.check_scheduler(0)

    def test_scheduler_failure_retains_error_tail(self):
        self.check_scheduler(1)

    def test_scheduler_worker_interrupt_retains_last_state(self):
        self.check_scheduler(130)

    def check_scheduler(self, exit_code):
        # Execute the actual scheduler orchestration with fake DB/process boundaries.
        import dataclasses, uuid, yaml, traceback
        source=Path(__file__).parents[1]/'train_and_eval/queue_execution/scheduler.py'
        tree=ast.parse(source.read_text())
        definitions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('Active','database_states','run')]
        actions=[];states={'a':'failed','b':'cancelled'}
        class Result:
            def __init__(self,rows):self.rows=rows
            def scalars(self):return self
            def tuples(self):return self
            def all(self):return self.rows
        class Connection:
            def execution_options(self,**kw):return self
            def execute(self,sql,params=None):
                if 'SELECT name, status' in sql:return Result([(n,states[n]) for n in params['names'] if n in states])
                return Result([])
            def scalar(self,sql,params=None):return datetime.now(timezone.utc) if 'clock_timestamp' in sql else 1
            def invalidate(self):pass
            def close(self):pass
        engine=NS(connect=lambda:Connection(),dispose=lambda:None)
        def restart(config,run_id):
            actions.append('ask:'+config.name);states.pop(config.name);return 'YES'
        class Recorder:
            def __init__(self,path,*args):self.path=path;path.write_text('timestamp,cpu_c\n')
            def record(self,*args):pass
            def flush_due(self):pass
            def close(self):self.path.with_name('telemetry_summary.json').write_text('{}')
        def launch(command,**kwargs):
            name=command[command.index('--name')+1];actions.append('start:'+name)
            states[name]='completed' if exit_code == 0 else 'cancelled'
            return NS(pid=99999999,stdout=io.StringIO(PREFIX+json.dumps({'phase':'TRAIN','stage':'TRAIN','completed':True})+'\n'),poll=lambda:exit_code,returncode=exit_code)
        ns=dict(dataclass=dataclasses.dataclass,Path=Path,json=json,os=os,signal=signal,subprocess=NS(Popen=launch,PIPE=subprocess.PIPE,STDOUT=subprocess.STDOUT),sys=sys,time=NS(monotonic=time.monotonic,time=time.time,sleep=lambda s:None),uuid=uuid,datetime=datetime,timezone=timezone,traceback=traceback,
            text=lambda s:s,create_database_engine=lambda:engine,CONTROL=1,RUN=2,run_key=lambda n:n,try_lock=lambda *a:True,lock_available=lambda *a:True,occupied_slots=lambda *a:0,
            other_local_queue=lambda root:None,Recorder=Recorder,temperatures=lambda:{'cpu':60},ThermalGate=lambda a,b:NS(limit=a,resume=b,admit=lambda t:True),
            Progress=Progress,OutputReader=OutputReader,atomic_json=atomic_json,classify=classify,runnable=lambda job,states:True,
            stop_children=lambda a:None,reconcile_attempt=lambda *a:None)
        exec(compile(ast.Module(body=definitions,type_ignores=[]),str(source),'exec',dont_inherit=True),ns)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);jobs=[];selected=[]
            for i,name in enumerate(('a','b'),1):
                path=root/(name+'.yml');path.write_text('ppo: {device: cuda}\nevaluation: {device: cpu, training_mode: final_only}\n')
                jobs.append(NS(name=name,source=None,path=path,position=i,sha256='hash'))
                selected.append((i,NS(name=name)))
            ns['plan_jobs']=lambda selected:jobs
            args=NS(workers=2,queue='q',temperature_limit=91.5,temperature_resume=84.5,abort_temperature=97.5,flush_seconds=30,control_seconds=3,sample_seconds=6,threads=1,serial_evaluation=True)
            with patch.dict(sys.modules,{'run_search_queue':NS(handle_incomplete_run=restart)}),patch.dict(os.environ,{},clear=True),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ns['run'](selected,args,root),0 if exit_code == 0 else 1)
            self.assertEqual(actions,['ask:a','ask:b','start:a','start:b'])
            directory=next((root/'artifacts/search_queues/q').iterdir())
            files={p.name for p in directory.iterdir()}
            self.assertEqual(files,{'plan.json','events.jsonl','telemetry.csv','telemetry_summary.json','summary.json','worker_1_last.state.json','worker_2_last.state.json'})
            summary=json.loads((directory/'summary.json').read_text())
            self.assertEqual(summary['completed_new_runs'],2 if exit_code == 0 else 0)
            self.assertEqual(summary['stages']['TRAIN']['completed'],2)
            events=[json.loads(l) for l in (directory/'events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(e['event']=='attempt_registered' for e in events),2)
            self.assertEqual(json.loads((directory/'worker_1_last.state.json').read_text())['phase'], 'COMPLETED' if exit_code == 0 else ('INTERRUPTED' if exit_code == 130 else 'FAILED'))
            self.assertEqual(sum(e['event']=='worker_error' for e in events), 0 if exit_code == 0 else 2)


if __name__=='__main__':unittest.main()
