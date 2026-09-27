import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from astra import run


class Process:
    def __init__(self, pid, returncode=None):
        self.pid, self.returncode = pid, returncode
        self.wait_timeouts = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired('fixture', timeout)
        return self.returncode


class LayoutRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.manifest = dict(initial_smoke_task='task', seed=0, layout_ids=[0, 1, 2], eval_num=3)
        self.manifest_path = self.folder / 'run_manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest))

    def complete(self, layout=1, status='success'):
        episode = self.folder / 'task/seed_0' / f'layout_{layout}_fixture'
        episode.mkdir(parents=True, exist_ok=True)
        value = dict(status=status, official_terminal=status in {'success', 'failed'},
                     official_success=True if status == 'success' else (False if status == 'failed' else None))
        (episode / 'summary.json').write_text(json.dumps(value))
        (self.folder / 'controller_status.json').write_text(json.dumps(dict(
            status='episode_finished' if status == 'success' else 'blocked_or_error', folder=str(episode))))
        return episode / 'summary.json'

    def test_reservation_preserves_plan_rejects_started_and_unplanned_layouts(self):
        original = self.manifest_path.read_bytes()
        with self.assertRaises(ValueError):
            run.reserve_layout(self.folder, self.manifest, 9)
        logs, status, record = run.reserve_layout(self.folder, self.manifest, 1)
        self.assertEqual(record['layout_ids'], [1])
        self.assertEqual(record['eval_num'], 1)
        self.assertEqual(self.manifest_path.read_bytes(), original)
        self.assertEqual(json.loads((self.folder / 'active_run.json').read_text())['status_file'], str(status))
        with self.assertRaises(FileExistsError):
            run.reserve_layout(self.folder, self.manifest, 1)
        (self.folder / 'task/seed_0/layout_0_started').mkdir(parents=True)
        with self.assertRaises(ValueError):
            run.reserve_layout(self.folder, self.manifest, 0)
        self.complete(layout=2)
        with self.assertRaises(ValueError):
            run.reserve_layout(self.folder, self.manifest, 2)

    def test_summary_without_matching_controller_terminal_is_not_teardown_authority(self):
        path = self.complete()
        (self.folder / 'controller_status.json').write_text(json.dumps(dict(status='waiting_for_agent')))
        self.assertIsNone(run.completed_layout_result(self.folder, 'task', 0, 1))
        self.complete(layout=0)
        self.assertIsNone(run.completed_layout_result(self.folder, 'task', 0, 1))
        self.complete()
        self.assertEqual(run.completed_layout_result(self.folder, 'task', 0, 1)['summary_path'], str(path))

    def test_hung_teardown_preserves_success_and_uses_bounded_group_cleanup(self):
        path = self.complete()
        original = path.read_bytes()
        simulator, server = Process(20), Process(21)
        elapsed, signals = [0], []
        def sleep(seconds):
            elapsed[0] += seconds
        def kill(pid, signum):
            signals.append((pid, signum))
            if signum == signal.SIGKILL:
                simulator.returncode = -9
        with patch.object(run.time, 'monotonic', side_effect=lambda: elapsed[0]), \
                patch.object(run.time, 'sleep', side_effect=sleep), patch.object(run.os, 'killpg', side_effect=kill):
            result = run.wait_layout(simulator, server, self.folder, 'task', 0, 1)
        self.assertGreaterEqual(elapsed[0], 30)
        self.assertEqual(signals, [(20, signal.SIGTERM), (20, signal.SIGKILL)])
        self.assertEqual(simulator.wait_timeouts, [15, 15])
        self.assertTrue(result['teardown']['forced'])
        self.assertEqual(result['summary']['status'], 'success')
        self.assertEqual(result['native_returncode'], -9)
        self.assertEqual(path.read_bytes(), original)

    def test_failed_or_blocked_summary_survives_nonzero_simulator_exit(self):
        for status in ('failed', 'blocked', 'infrastructure_error'):
            with self.subTest(status=status):
                self.complete(status=status)
                result = run.wait_layout(Process(20, 1), Process(21), self.folder, 'task', 0, 1)
                self.assertEqual(result['summary']['status'], status)
                self.assertIsNot(result['summary']['official_success'], True)
                self.assertEqual(result['native_returncode'], 1)

    def test_process_exit_without_summary_is_error(self):
        with self.assertRaisesRegex(RuntimeError, 'without a finalized'):
            run.wait_layout(Process(20, 0), Process(21), self.folder, 'task', 0, 1)
        with self.assertRaisesRegex(RuntimeError, 'Policy server exited'):
            run.wait_layout(Process(20), Process(21, 1), self.folder, 'task', 0, 1)

    def test_layout_launch_uses_isolated_logs_environment_and_result_record(self):
        (self.folder / 'demos/task').mkdir(parents=True)
        (self.folder / 'demos/task/demo_context.json').write_text('{"status":"reviewed"}')
        (self.folder / 'model.yaml').write_text('flux2_src_path: /tmp/fixture_flux\n')
        nvrtc = self.folder / 'site/nvidia/cu13/lib/libnvrtc.so.13'
        nvrtc.parent.mkdir(parents=True)
        nvrtc.touch()
        original = self.manifest_path.read_bytes()
        server, simulator = Process(21), Process(20, 1)
        launches = []
        def popen(command, **kwargs):
            launches.append((command, kwargs))
            if len(launches) == 1:
                return server
            self.complete(status='blocked')
            return simulator
        def kill(pid, signum):
            self.assertEqual(pid, server.pid)
            server.returncode = -int(signum)
        env = dict(SLURM_JOB_ID='fixture', SLURM_JOB_NUM_NODES='1', SLURM_JOB_NAME='wangzihang_astra',
                   CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
        argv = ['astra.run', '--session', str(self.folder), '--layout-id', '1', '--gpu', '5', '--port', '40000']
        with patch.dict(os.environ, env), patch('sys.argv', argv), \
                patch('site.getsitepackages', return_value=[str(self.folder / 'site')]), \
                patch.object(run.subprocess, 'Popen', side_effect=popen), \
                patch.object(run.socket, 'create_connection', return_value=MagicMock()), \
                patch.object(run.signal, 'signal'), patch.object(run.os, 'killpg', side_effect=kill):
            run.main()
        self.assertEqual(self.manifest_path.read_bytes(), original)
        for command, options in launches:
            self.assertEqual(options['env']['EVAL_NUM'], '1')
            self.assertEqual(options['env']['ROBODOJO_LAYOUT_IDS'], '1')
            self.assertEqual(options['env']['ASTRA_SESSION'], str(self.folder))
            self.assertIn('_layout1_', options['env']['ROBODOJO_RUN_ID'])
            self.assertEqual(Path(options['stdout'].name).parent, self.folder / 'runs/layout_1')
        self.assertEqual(launches[0][1]['env']['CUDA_VISIBLE_DEVICES'], '5')
        status = json.loads((self.folder / 'runs/layout_1/run_status.json').read_text())
        self.assertEqual(status['status'], 'layout_finished')
        self.assertEqual(status['outcome'], 'blocked')
        self.assertIsNone(status['official_success'])
        self.assertEqual(status['result']['native_returncode'], 1)
        self.assertFalse(status['cleanup_errors'])


if __name__ == '__main__':
    unittest.main()
