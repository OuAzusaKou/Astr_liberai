import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    'restart_infrastructure',
    Path(__file__).resolve().parents[1] / 'scripts/restart_infrastructure.py',
)
restart = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restart)


class InfrastructureRecoveryTests(unittest.TestCase):
    def test_retries_reuse_original_batch_and_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'session_original'
            root.mkdir()
            worker = root / 'workers' / 'w00_hang_mugs_random'
            worker.mkdir(parents=True)
            (root / 'parallel_manifest.json').write_text(json.dumps({
                'workers': [{'task': 'hang_mugs_random', 'worker_id': 'hang_mugs_random',
                             'session': str(worker)}]}))
            failure = dict(task='hang_mugs_random', worker_id='hang_mugs_random:layout_2', layout_id=2)

            child = types.SimpleNamespace(pid=1234)
            def popen(*args, **kwargs):
                kwargs['stdout'].close()
                return child

            with patch.object(restart.subprocess, 'Popen', side_effect=popen):
                first = restart.launch_recovery(root, [failure], [2], 1, recovery_root=root)
                second = restart.launch_recovery(root, [failure], [2], 2, recovery_root=root)

            self.assertEqual(Path(first['recovery_batch']), root.resolve())
            self.assertEqual(Path(second['recovery_batch']), root.resolve())
            self.assertTrue(first['reused_worker_sessions'])
            self.assertEqual(first['workers'][0]['worker_session'], str(worker.resolve()))
            self.assertEqual(second['workers'][0]['worker_session'], str(worker.resolve()))


if __name__ == '__main__':
    unittest.main()
