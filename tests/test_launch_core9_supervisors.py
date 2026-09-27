import importlib.util
from pathlib import Path
import unittest
import tempfile
import json
import subprocess
from unittest.mock import MagicMock, patch

spec = importlib.util.spec_from_file_location('external_supervisors', Path(__file__).resolve().parents[1] / 'scripts/launch_core9_supervisors.py')
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


class ExternalSupervisorTests(unittest.TestCase):
    def test_alias_preserves_arguments_and_prompt_stdin(self):
        import shlex
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = root / 'wrapper.sh'
            wrapper.write_text('#!/bin/bash\npython3 -c \'import json,sys; print(json.dumps([sys.argv[1:],sys.stdin.read()]))\' "$@"\n')
            wrapper.chmod(0o755)
            rc = root / 'bashrc'
            rc.write_text('alias wangzihang_codex=' + shlex.quote(str(wrapper)) + '\n')
            args = ['exec', '-C', 'path with spaces; $(false)', '-']
            command = manager.supervisor_command(manager.supervisor_executable(), args)
            command[1:1] = ['--noprofile', '--rcfile', str(rc)]
            result = subprocess.run(command, input='supervision prompt\n', text=True,
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [args, 'supervision prompt\n'])

    def test_real_prompt_generation_without_site_packages(self):
        root = Path(__file__).resolve().parents[1]
        code = '''
import sys
from pathlib import Path
from astra.config import CORE9
from astra.task_run import make_prompt
root = Path.cwd()
for task in CORE9:
    manifest = dict(conditioning_mode='l3_l2', max_actions_per_decision=15,
                    high_level_monitor_interval_actions=15,
                    authorized_demo_waivers=['general_pickup'])
    prompt = make_prompt(root / 'outputs/preflight/workers' / task,
                         root / 'outputs/preflight', task, [0, 1, 2], 'probe', manifest)
    assert '15' in prompt and 'fastwam-dpnew-qwen/bin/python' in prompt
assert 'PIL' not in sys.modules and 'numpy' not in sys.modules
'''
        result = subprocess.run(['/usr/bin/python3', '-S', '-c', code], cwd=root,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_preflight_failure_precedes_detached_process_and_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            worker = batch / 'worker'
            worker.mkdir()
            (batch / 'parallel_manifest.json').write_text(json.dumps(dict(
                workers=[dict(task='hang_mugs', session=str(worker))])))
            with patch.object(manager.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(ValueError, 'Missing worker manifest'):
                    manager.launch_detached(batch)
                popen.assert_not_called()
            self.assertFalse((batch / 'external_supervisor_launcher.json').exists())
            self.assertFalse((worker / 'supervisor_owner.json').exists())

    def test_detached_launch_has_independent_session_and_persistent_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            (batch / 'parallel_manifest.json').write_text('{}')
            child = MagicMock(pid=4321)
            child.poll.return_value = None
            def launch(*args, **kwargs):
                self.assertTrue(kwargs['start_new_session'])
                self.assertEqual(kwargs['stdin'], manager.subprocess.DEVNULL)
                self.assertEqual(Path(kwargs['stdout'].name), batch / 'external_supervisor_manager.log')
                (batch / 'external_supervisor_manager.json').write_text(
                    json.dumps(dict(pid=4321, heartbeat_at=1)))
                return child
            with patch.object(manager.subprocess, 'Popen', side_effect=launch) as popen:
                self.assertEqual(manager.launch_detached(batch)['pid'], 4321)
                with self.assertRaises(FileExistsError):
                    manager.launch_detached(batch)
                self.assertEqual(popen.call_count, 1)

    def workers(self, count=11):
        return [dict(task=f'task{i}', status='running' if i < 8 else 'queued', returncode=None) for i in range(count)]

    def schedules(self, count=11):
        return {f'task{i}': dict(status='waiting_for_supervisor', all_layouts_settled=False) for i in range(count)}

    def test_only_allocated_waiting_tasks_start_up_to_eight(self):
        workers, schedules = self.workers(), self.schedules()
        launches, stops = manager.plan_supervisors(workers, schedules, set(), {})
        self.assertEqual(launches, [f'task{i}' for i in range(8)])
        self.assertFalse(stops)
        schedules['task0']['status'] = 'starting_supervisor'
        launches, _ = manager.plan_supervisors(workers, schedules, set(), {})
        self.assertNotIn('task0', launches)
        # Even an incorrect pool that marks eleven workers running cannot spawn eleven CLI processes.
        for worker in workers:
            worker['status'] = 'running'
        schedules['task0']['status'] = 'waiting_for_supervisor'
        self.assertEqual(len(manager.plan_supervisors(workers, schedules, set(), {})[0]), 8)

    def test_layout_workers_for_one_task_get_independent_supervision_slots(self):
        workers = [dict(task='hang_mugs_random', worker_id=f'hang_mugs_random:layout_{i}',
                        status='running', returncode=None) for i in (1, 2)]
        schedules = {worker['worker_id']: dict(status='waiting_for_supervisor') for worker in workers}
        launches, stops = manager.plan_supervisors(workers, schedules, set(), {}, capacity=2)
        self.assertEqual(launches, ['hang_mugs_random:layout_1', 'hang_mugs_random:layout_2'])
        self.assertFalse(stops)

    def test_supervision_capacity_obeys_requested_batch_limit(self):
        workers = [dict(task=f'task{i}', worker_id=f'task{i}:layout_0',
                        status='running', returncode=None) for i in range(4)]
        schedules = {worker['worker_id']: dict(status='waiting_for_supervisor') for worker in workers}
        launches, _ = manager.plan_supervisors(workers, schedules, set(), {}, capacity=2)
        self.assertEqual(launches, ['task0:layout_0', 'task1:layout_0'])

    def test_slot_not_reused_until_old_child_exits_and_never_respawns_prior_owner(self):
        workers, schedules = self.workers(), self.schedules()
        started = {f'task{i}' for i in range(8)}
        active = dict.fromkeys(started)
        workers[8]['status'] = 'running'
        self.assertEqual(manager.plan_supervisors(workers, schedules, started, active)[0], [])
        active.pop('task0')
        self.assertEqual(manager.plan_supervisors(workers, schedules, started, active)[0], ['task8'])
        self.assertNotIn('task0', manager.plan_supervisors(workers, schedules, started, {})[0])

    def test_cancelled_allocation_stops_active_work_and_never_launches_queue(self):
        launches, stops = manager.plan_supervisors(self.workers(), self.schedules(), {'task0'}, {'task0': {}}, job_terminal=True)
        self.assertEqual(launches, [])
        self.assertIn('task0', stops)
        self.assertFalse(manager.finished_waiting({'status': 'running'}, True, {'task0': {}}))
        self.assertTrue(manager.finished_waiting({'status': 'running'}, True, {}))

    def test_normal_settled_task_can_finalize_naturally_but_manager_error_stops_it(self):
        workers, schedules = self.workers(), self.schedules()
        schedules['task0']['all_layouts_settled'] = True
        workers[0].update(returncode=0, status='settled', finalized=True)
        self.assertFalse(manager.plan_supervisors(workers, schedules, {'task0'}, {'task0': {}}, pool_terminal=True)[1])
        workers[0].update(returncode=2, status='task_error')
        self.assertIn('task0', manager.plan_supervisors(workers, schedules, {'task0'}, {'task0': {}})[1])

    def test_empty_or_long_pending_queue_is_not_completion(self):
        for config in ({'status': 'prepared'}, {'status': 'running'}, {'status': 'submitted'}):
            self.assertFalse(manager.finished_waiting(config, False, {}))
        workers = self.workers()
        for worker in workers:
            worker['status'] = 'queued'
        self.assertEqual(manager.plan_supervisors(workers, self.schedules(), set(), {}), ([], {}))
        self.assertTrue(manager.finished_waiting({'status': 'complete_or_blocked_review_results'}, False, {}))

    def test_lock_cannot_be_released_while_child_is_alive(self):
        child, lock = MagicMock(), MagicMock()
        child.poll.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'lock while owned child is alive'):
            manager.release(dict(child=child, lock=lock))
        lock.close.assert_not_called()

    def test_empty_squeue_requires_accounting_confirmation(self):
        queue = MagicMock(returncode=0, stdout='')
        accounting = MagicMock(returncode=0, stdout='')
        with patch.object(manager.subprocess, 'run', side_effect=[queue, accounting]):
            self.assertIsNone(manager.slurm_state('10200'))
        accounting.stdout = 'CANCELLED by 1000|\n'
        with patch.object(manager.subprocess, 'run', side_effect=[queue, accounting]):
            self.assertEqual(manager.slurm_state('10200'), 'CANCELLED')


if __name__ == '__main__':
    unittest.main()
