"""Real file-lock interoperability, without PostgreSQL, training or hardware writes."""
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cold = load('cold_guard_test', ROOT/'extra_tools/maintenance/cold_storage.py')
guards = load('maintenance_guard_test', ROOT/'extra_tools/maintenance/maintenance_guard.py')


class ColdStorageGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        target = self.root/'extra_tools/maintenance/maintenance_guard.py'
        target.parent.mkdir(parents=True)
        shutil.copy2(ROOT/'extra_tools/maintenance/maintenance_guard.py', target)

    def test_active_queue_blocks_all_live_modes(self):
        with guards.activity_guard(self.root):
            for mode in ['scan', 'sync', 'restore-database']:
                with self.subTest(mode=mode), self.assertRaises(cold.BackupError):
                    with cold.project_operation_guard(self.root, mode):
                        self.fail('Guard admitted a concurrent capture')

    def test_active_maintenance_blocks_capture(self):
        with guards.maintenance_guard(self.root):
            with self.assertRaises(cold.BackupError):
                with cold.project_operation_guard(self.root, 'sync'):pass

    def test_capture_blocks_queue_maintenance_and_legacy_managed(self):
        with patch.object(cold,'no_workers'):
            with cold.project_operation_guard(self.root,'scan'):
                with self.assertRaises(RuntimeError):
                    with guards.activity_guard(self.root):pass
                with self.assertRaises(RuntimeError):
                    with guards.maintenance_guard(self.root):pass
                # A different project still uses the same per-user hardware lock.
                with self.assertRaises(RuntimeError):
                    with guards.maintenance_guard(self.root/'other',exclude_managed=True):pass

    def test_pending_journals_block_capture(self):
        for rel in ['extra_tools/maintenance/run_rename/pending.json', 'artifacts/.cleanup_pending.json']:
            p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
            try:
                with self.assertRaises(cold.BackupError):
                    with cold.project_operation_guard(self.root,'sync'):pass
            finally:p.unlink()

    def test_exception_releases_all_locks(self):
        with patch.object(cold,'no_workers'):
            with self.assertRaisesRegex(ValueError,'test failure'):
                with cold.project_operation_guard(self.root,'sync'):
                    raise ValueError('test failure')
            with cold.project_operation_guard(self.root,'sync'):pass

    def test_failed_acquisition_releases_maintenance_lock(self):
        with guards.activity_guard(self.root):
            with self.assertRaises(cold.BackupError):
                with cold.project_operation_guard(self.root,'scan'):pass
            with guards.maintenance_guard(self.root):pass

    def test_missing_guard_refused_for_live_mode(self):
        with self.assertRaisesRegex(cold.BackupError,'missing'):
            with cold.project_operation_guard(self.root/'missing','scan'):pass

    def test_offline_modes_do_not_need_project_guard(self):
        for mode in ['restore-files','verify','compare','probe','prune','disks']:
            with cold.project_operation_guard(self.root/'missing',mode):pass

    def test_restore_files_entry_is_standalone(self):
        with patch.object(cold,'read_json',return_value={'project_id':'test'}), \
             patch.object(cold,'destination',return_value=nullcontext()), \
             patch.object(cold,'restore_files') as restore, \
             patch.object(cold,'project_operation_guard',side_effect=AssertionError('Project import forbidden')):
            self.assertEqual(cold._main(['restore-files','--backup',str(self.root/'archive'),
                                         '--target',str(self.root/'restored')]),0)
            restore.assert_called_once()

    def test_managed_process_detected_before_database_insert(self):
        proc=self.root/'fakeproc'/'999999';proc.mkdir(parents=True)
        (proc/'cmdline').write_bytes(b'python\0run_managed_queue.py\0')
        (proc/'cwd').symlink_to(self.root,target_is_directory=True)
        real_path=Path
        with patch.object(cold,'Path',side_effect=lambda p: self.root/'fakeproc' if p=='/proc' else real_path(p)):
            with self.assertRaisesRegex(cold.BackupError,'999999'):
                cold.no_workers(self.root)


if __name__=='__main__':unittest.main()
