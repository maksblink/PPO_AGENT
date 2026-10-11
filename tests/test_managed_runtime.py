"""CPU-only tests of production guards, affinity, telemetry and restoration."""
import contextlib
import gzip
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from train_and_eval.queue_execution import runtime, affinity, telemetry, hardware


class ManagedTests(unittest.TestCase):
    def test_device_policy_rejects_wrong_requests_and_fallback(self):
        runtime.require_training_config(NS(ppo=NS(device='cuda'), evaluation=NS(device='cpu')))
        for train,ev in [('cpu','cpu'),('auto','cpu'),('cuda','cuda')]:
            with self.assertRaises(RuntimeError):
                runtime.require_training_config(NS(ppo=NS(device=train),evaluation=NS(device=ev)))
        model=NS(device=NS(type='cpu'),policy=NS(parameters=lambda:[NS(device=NS(type='cpu'))]))
        runtime.check_model_device(model,'cpu')
        with self.assertRaises(RuntimeError):runtime.check_model_device(model,'cuda')
        with patch.dict('sys.modules',{'torch':NS(cuda=NS(is_available=lambda:False))}):
            with self.assertRaises(RuntimeError):runtime.require_cuda()

    def test_initialization_restores_on_success_and_failure(self):
        for fails in (False,True):
            state={'threads':1}
            fake=NS(get_num_threads=lambda:state['threads'],set_num_threads=lambda n:state.update(threads=n))
            with patch.dict('sys.modules',{'torch':fake}), patch.dict(os.environ,{'PPO_QUEUE_THREAD_POLICY':'16-to-1'}), patch.object(runtime,'phase'):
                try:
                    with runtime.model_initialization():
                        self.assertEqual(state['threads'],16)
                        if fails:raise ValueError('test')
                except ValueError:self.assertTrue(fails)
                self.assertEqual(state['threads'],1)

    def test_affinity_rotation_waits_for_exit(self):
        alloc=affinity.Allocator([[[0,2,4,6],[8,10,12,14]],[[16,18,20,22],[24,26,28,30]]])
        a=NS(poll=lambda:None);b=NS(poll=lambda:None)
        self.assertEqual(alloc.choose(),(0,[0,2,4,6]));alloc.attach(0,a)
        self.assertEqual(alloc.choose(),(1,[16,18,20,22]));alloc.attach(1,b)
        with self.assertRaises(RuntimeError):alloc.choose()
        b.poll=lambda:0
        self.assertEqual(alloc.choose(),(1,[24,26,28,30]))

    def test_topology_uses_one_thread_per_core(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for cpu in range(32):
                p=root/f'cpu{cpu}'
                for rel,value in {'topology/physical_package_id':'0','topology/core_id':str(cpu//2),'cache/index3/level':'3','cache/index3/id':str(cpu//16)}.items():
                    f=p/rel;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(value)
            groups=affinity.discover_groups(root,range(32))
            self.assertEqual(groups[0],[[0,2,4,6],[8,10,12,14]])
            self.assertEqual(groups[1],[[16,18,20,22],[24,26,28,30]])
            with self.assertRaises(RuntimeError):affinity.discover_groups(root,range(16))

    def test_telemetry_flush_retained_csv_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'telemetry.csv'
            with patch.object(telemetry,'cpu_ticks',side_effect=[{'cpu':(100,60)},{'cpu':(200,100)}]),patch.object(telemetry,'memory',return_value=15000),patch.object(telemetry,'gpus',return_value=[['0','50','620','48']]):
                recorder=telemetry.Recorder(path)
                recorder.record([{'name':'a','phase':'training','cpus':[0,2,4,6]}],{'k10temp/Tctl':70})
                recorder.last_flush-=31;recorder.flush_due()
                self.assertIn('training',path.read_text())
                recorder.close()
            self.assertTrue(path.exists())
            self.assertFalse(Path(str(path)+'.gz').exists())
            self.assertIn('70',path.read_text())
            result=json.loads((Path(d)/'telemetry_summary.json').read_text())
            self.assertEqual(result['rows'],1)
            self.assertEqual(result['metrics']['cpu_utilization_pct']['mean'],60)

    def test_hardware_restores_every_setting_after_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sysroot=root/'sys';report=root/'report';report.mkdir()
            values={'devices/platform/legion/powermode':'3','devices/platform/legion/fan_fullspeed':'0',
                    'devices/system/cpu/cpufreq/boost':'1','devices/system/cpu/cpufreq/policy0/scaling_max_freq':'5462711',
                    'devices/system/cpu/cpufreq/policy0/scaling_min_freq':'1480856','devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq':'5462711','class/power_supply/ADP0/online':'1'}
            for rel,v in values.items():
                p=sysroot/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(v)
            def command(argv,**kw):
                self.assertEqual(argv[:3],['sudo','-n','tee'])
                Path(argv[3]).write_text(kw['input'].strip())
            with patch.object(hardware,'SudoSession',return_value=contextlib.nullcontext()),patch.object(hardware.subprocess,'run',side_effect=command),patch.object(hardware.time,'sleep'):
                with self.assertRaisesRegex(ValueError,'workload'):
                    with hardware.thermal_session(report,5000,sysroot):raise ValueError('workload failed')
            for rel,v in values.items():self.assertEqual((sysroot/rel).read_text(),v)
            self.assertEqual(json.loads((report/'hardware.json').read_text())['restore_errors'],[])


    def test_cuda_retry_execs_once_before_database_work(self):
        import ast, argparse, signal, sys, time
        source=Path(__file__).parents[1]/'train_and_eval/queue_execution/worker.py'
        tree=ast.parse(source.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        class Reexec(Exception): pass
        ns={'argparse':argparse,'os':os,'signal':signal,'time':time,
            'configure_threads':lambda n:None,'interrupted':lambda *a:None,
            'create_database_engine':lambda: self.fail('DB must not be opened for CUDA failure')}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        argv=['--config','config.yml','--name','run','--sha256','abc','--workers','2','--threads','1','--temperature-limit','91.5']
        fake=NS(cuda=NS(is_available=lambda:False))
        with patch.dict('sys.modules',{'torch':fake}),patch.dict(os.environ,{},clear=False),patch.object(runtime,'phase'),patch.object(signal,'signal'),patch.object(time,'sleep') as sleep,patch.object(os,'execv',side_effect=Reexec) as execute:
            with self.assertRaises(Reexec):ns['main'](argv)
            sleep.assert_called_once_with(30)
            self.assertEqual(execute.call_args.args[1][-1],'--cuda-retried')
            self.assertEqual(execute.call_args.args[1][1:3],['-m','train_and_eval.queue_execution.worker'])
            sleep.reset_mock();execute.reset_mock()
            self.assertEqual(ns['main'](argv+['--cuda-retried']),1)
            sleep.assert_not_called();execute.assert_not_called()

if __name__=='__main__':unittest.main()
