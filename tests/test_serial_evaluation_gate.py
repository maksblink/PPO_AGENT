"""Real subprocess gate tests, isolated from pytest/Streamlit module imports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from train_and_eval.queue_execution.evaluation_gate import evaluation_lock_scope, serialize_evaluation


PARTICIPANT = r"""
import sys
import time
from pathlib import Path
from train_and_eval.queue_execution.evaluation_gate import evaluation_lock_scope, serialize_evaluation
lock, prefix = sys.argv[1:]
prefix = Path(prefix)
@serialize_evaluation
def evaluate():
    prefix.with_suffix('.entered').touch()
    while not prefix.with_suffix('.release').exists():
        time.sleep(.02)
with evaluation_lock_scope(lock):
    evaluate()
"""


class GateTests(unittest.TestCase):
    def wait_for(self, process, predicate, message, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            self.assertIsNone(process.poll(), 'participant exited before ' + message)
            time.sleep(.02)
        self.fail('Timed out: ' + message)

    def run_pair(self, crash):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / 'evaluation.lock'
            processes = []
            def start(name):
                prefix = root / name
                env = os.environ.copy()
                env['PPO_QUEUE_PHASE_PATH'] = str(prefix.with_suffix('.json'))
                project = Path(__file__).resolve().parents[1]
                env['PYTHONPATH'] = str(project) + os.pathsep + env.get('PYTHONPATH', '')
                process = subprocess.Popen(
                    [sys.executable, '-c', PARTICIPANT, str(lock), str(prefix)],
                    cwd=project, env=env, stdout=subprocess.DEVNULL,
                )
                processes.append(process)
                return process, prefix
            def phase(prefix):
                path = prefix.with_suffix('.json')
                return json.loads(path.read_text()).get('phase') if path.exists() else None
            try:
                a, first = start('first')
                self.wait_for(a, first.with_suffix('.entered').exists, 'first evaluation entered')
                b, second = start('second')
                self.wait_for(b, lambda: phase(second) == 'evaluation_wait', 'second reached lock')
                time.sleep(.3)
                self.assertFalse(second.with_suffix('.entered').exists(), 'evaluations overlap')
                if crash:
                    a.kill()
                else:
                    first.with_suffix('.release').touch()
                a.wait(timeout=10)
                self.assertEqual(a.returncode, -9 if crash else 0)
                self.wait_for(b, second.with_suffix('.entered').exists, 'lock released')
                second.with_suffix('.release').touch()
                self.assertEqual(b.wait(timeout=10), 0)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=10)

    def test_normal_release(self):
        self.run_pair(crash=False)

    def test_process_exclusion_and_crash_release(self):
        self.run_pair(crash=True)

    def test_nested_calls_exception_and_scope_reset(self):
        calls = []
        @serialize_evaluation
        def inner():
            calls.append('inner')
            raise ValueError('expected')
        @serialize_evaluation
        def outer():
            inner()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evaluation.lock'
            with evaluation_lock_scope(path):
                with self.assertRaises(ValueError):
                    outer()
                with self.assertRaises(ValueError):
                    inner()
            # A removed directory demonstrates that the default has no file IO.
        with self.assertRaises(ValueError):
            inner()
        self.assertEqual(calls, ['inner', 'inner', 'inner'])


if __name__ == '__main__':
    unittest.main()
