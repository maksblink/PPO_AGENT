"""Terminal-only behavior: fixed rows, sparse updates and a 90-second cadence."""
import io
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from train_and_eval.queue_execution.monitor import TerminalBlock, Progress


class TTY(io.StringIO):
    def isatty(self): return True


class TerminalTests(unittest.TestCase):
    def test_replaces_only_changed_row_without_newlines(self):
        stream=TTY()
        with patch.dict('os.environ',{'TERM':'xterm'}),patch('shutil.get_terminal_size',return_value=NS(columns=120)):
            block=TerminalBlock(stream)
            block.render(['TABLE','Worker 1: a | TRAIN','Worker 2: - | IDLE','Thermal: RUNNING'])
            stream.seek(0);stream.truncate()
            block.render(['TABLE','Worker 1: a | EVAL_VAL','Worker 2: - | IDLE','Thermal: RUNNING'])
            output=stream.getvalue()
            self.assertNotIn('TABLE',output)
            self.assertNotIn('Worker 2',output)
            self.assertNotIn('\n',output)
            self.assertIn('\x1b[3A\r\x1b[2KWorker 1: a | EVAL_VAL\x1b[3B\r',output)
            stream.seek(0);stream.truncate()
            block.render(['TABLE','Worker 1: a | EVAL_VAL','Worker 2: - | IDLE','Thermal: RUNNING'])
            self.assertEqual(stream.getvalue(),'')

    def test_redirected_output_only_initial_final_and_no_escape_codes(self):
        stream=io.StringIO();block=TerminalBlock(stream)
        block.render(['initial']);block.render(['middle']);block.render(['final'],final=True)
        self.assertEqual(stream.getvalue(),'initial\nfinal\n')

    def test_long_names_do_not_wrap(self):
        stream=TTY()
        with patch.dict('os.environ',{'TERM':'xterm'}),patch('shutil.get_terminal_size',return_value=NS(columns=40)):
            block=TerminalBlock(stream);block.render(['Worker 1: '+'long_name_'*20])
            self.assertEqual(len(stream.getvalue().rstrip('\n')),39)

    def test_ninety_seconds_even_when_dirty_and_immediate_thermal(self):
        stream=TTY()
        with patch.dict('os.environ',{'TERM':'xterm'}),patch('shutil.get_terminal_size',return_value=NS(columns=160)),patch('sys.stdout',stream),patch('time.monotonic',return_value=100) as clock:
            p=Progress([NS(name='a')],{'a':{}},workers=2)
            p.display({},[],force=True)
            stream.seek(0);stream.truncate()
            active={'a':NS(slot=1,job=NS(name='a'),state={'phase':'TRAIN'})}
            p.dirty=True;clock.return_value=189
            p.display(active,[])
            self.assertEqual(stream.getvalue(),'')
            p.thermal('Thermal admission paused: CPU=92')
            self.assertIn('Thermal admission paused',stream.getvalue())
            self.assertNotIn('Worker',stream.getvalue())
            stream.seek(0);stream.truncate();clock.return_value=190
            p.display(active,[])
            self.assertIn('Worker 1: a | TRAIN',stream.getvalue())
            self.assertNotIn('Worker 2:',stream.getvalue())
            self.assertNotIn('\n',stream.getvalue())
            p.close();stream.seek(0);stream.truncate()
            p.display({},[],force=True)
            self.assertEqual(stream.getvalue(),'')


if __name__=='__main__':unittest.main()
