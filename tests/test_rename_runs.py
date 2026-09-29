"""Run rename filesystem recovery never overwrites unrelated edits."""
import tempfile
import unittest
from pathlib import Path
from extra_tools import rename_runs as tool


class RenameFilesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = {'files': [
            {'path':'configs/old.yml','before':'old config','after':None},
            {'path':'configs/new.yml','before':None,'after':'new config'},
            {'path':'reports/summary.json','before':'old summary','after':'new summary'}]}
        for f in self.plan['files']:
            if f['before'] is not None:
                path=self.root/f['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_text(f['before'])

    def test_finish_and_repeat(self):
        tool.finish_files(self.root,self.plan,'after')
        tool.finish_files(self.root,self.plan,'after')
        self.assertFalse((self.root/'configs/old.yml').exists())
        self.assertEqual((self.root/'configs/new.yml').read_text(),'new config')

    def test_recover_partial_application(self):
        (self.root/'configs/new.yml').write_text('new config')
        tool.finish_files(self.root,self.plan,'after')
        self.assertEqual((self.root/'reports/summary.json').read_text(),'new summary')

    def test_rollback_partial_application(self):
        (self.root/'configs/new.yml').write_text('new config')
        (self.root/'configs/old.yml').unlink()
        tool.finish_files(self.root,self.plan,'before')
        self.assertEqual((self.root/'configs/old.yml').read_text(),'old config')
        self.assertFalse((self.root/'configs/new.yml').exists())

    def test_external_edit_refused_before_any_change(self):
        (self.root/'reports/summary.json').write_text('unrelated edit')
        with self.assertRaisesRegex(ValueError,'outside migration'):
            tool.finish_files(self.root,self.plan,'after')
        self.assertTrue((self.root/'configs/old.yml').exists())
        self.assertFalse((self.root/'configs/new.yml').exists())

    def test_path_escape_and_symlink_refused(self):
        for rel in ['/tmp/outside','../outside']:
            with self.assertRaises(ValueError):tool.safe(self.root,rel)
        (self.root/'link').symlink_to(self.root/'configs',target_is_directory=True)
        with self.assertRaises(ValueError):tool.safe(self.root,'link/old.yml')

    def test_atomic_write_can_repeat(self):
        p=self.root/'result.json';tool.write_atomic(p,'one');tool.write_atomic(p,'two')
        self.assertEqual(p.read_text(),'two')
        tool.write_atomic(p,None);tool.write_atomic(p,None)
        self.assertFalse(p.exists())
