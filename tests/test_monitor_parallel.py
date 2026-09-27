import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('monitor_parallel', Path(__file__).resolve().parents[1] / 'scripts/monitor_parallel.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.batch = Path(self.tmp.name)
        self.worker = self.batch / 'worker'
        self.task = 'example'
        write(self.batch / 'parallel_manifest.json', dict(job_id=10176, seed=0, workers=[
            dict(task=self.task, session=str(self.worker), gpu=0, layout_ids=[0, 1, 2])]))
        write(self.worker / 'run_manifest.json', dict(layout_ids=[0, 1, 2], seed=0))
        self.process(None)

    def process(self, rc):
        write(self.batch / 'workers_status.json', [dict(task=self.task, returncode=rc)])

    def episode(self, layout, result=None, suffix='test', seed=0, folder=None):
        path = (folder or self.worker) / self.task / f'seed_{seed}' / f'layout_{layout}_{suffix}'
        path.mkdir(parents=True)
        if result is not None:
            write(path / 'summary.json', result)
        return path

    def row(self):
        return monitor.snapshot(self.batch)['tasks'][0]

    def test_finished_failure_does_not_mean_success_and_running_not_started_differ(self):
        self.episode(0, dict(status='failed', official_terminal=True, official_success=False,
                             retries={'grasp': 3}, takeovers=2))
        self.episode(1)
        row = self.row()
        self.assertEqual((row['planned_episodes'], row['tested_episodes'], row['success_episodes']), (3, 1, 0))
        self.assertEqual((row['official_failures'], row['running_episodes'], row['not_run_episodes']), (1, 1, 1))
        self.assertEqual(row['retries'], {'grasp': 3})
        self.assertEqual(row['takeover_count'], 2)
        self.assertEqual(row['completed_episodes'], 1)
        self.assertEqual(row['success_rate_planned'], 0)
        monitor.save(self.batch, monitor.snapshot(self.batch))
        markdown = (self.batch / 'monitor_status.md').read_text()
        self.assertIn('成功/计划', markdown)
        self.assertIn('0/3', markdown)
        self.assertIn('0/1', markdown)
        self.process(1)
        self.assertEqual(self.row()['running_episodes'], 0)
        self.assertEqual(self.row()['interrupted_episodes'], 1)

    def test_success_breakdown_and_official_terminal_required(self):
        self.episode(0, dict(status='success', official_terminal=True, official_success=True, corrections=0, takeovers=0))
        self.episode(1, dict(status='success', official_terminal=True, official_success=True, corrections=2, takeovers=0))
        self.episode(2, dict(status='success', official_terminal=False, official_success=True, corrections=0, takeovers=0))
        self.process(0)
        row = self.row()
        self.assertEqual((row['zero_intervention_successes'], row['assisted_successes'], row['success_episodes']), (1, 1, 2))
        self.assertEqual(row['unclassified_episodes'], 1)
        self.assertFalse(row['fully_evaluated'])
        self.assertAlmostEqual(row['success_rate_planned'], 2/3)

    def test_distinct_layout_workers_for_one_task_aggregate_without_key_collisions(self):
        workers, processes, supervisors = [], [], []
        for layout in (1, 2):
            worker_id = f'{self.task}:layout_{layout}'
            folder = self.batch / f'worker_{layout}'
            workers.append(dict(worker_id=worker_id, task=self.task, session=str(folder),
                                gpu=layout, layout_ids=[layout]))
            write(folder / 'run_manifest.json', dict(layout_ids=[layout], seed=0))
            write(folder / 'task_scheduler_status.json', dict(
                status='finished', all_layouts_settled=True,
                layouts=[dict(layout_id=layout, state='settled')]))
            self.episode(layout, dict(status='success', official_terminal=True,
                                      official_success=True, corrections=0, takeovers=0), folder=folder)
            processes.append(dict(worker_id=worker_id, task=self.task, status='settled', returncode=0))
            supervisors.append(dict(worker_id=worker_id, task=self.task, returncode=0, ready=True))
        write(self.batch / 'parallel_manifest.json', dict(job_id=10176, seed=0, workers=workers))
        write(self.batch / 'workers_status.json', processes)
        write(self.batch / 'codex_supervision/status.json', dict(workers=supervisors))
        snapshot = monitor.snapshot(self.batch)
        self.assertEqual(len(snapshot['tasks']), 2)
        self.assertEqual((snapshot['planned_episodes'], snapshot['tested_episodes'],
                          snapshot['success_episodes']), (2, 2, 2))
        self.assertEqual({row['worker_id'] for row in snapshot['tasks']},
                         {f'{self.task}:layout_1', f'{self.task}:layout_2'})

    def test_duplicate_and_unexpected_attempts_never_inflate_successes(self):
        success = dict(status='success', official_terminal=True, official_success=True, corrections=0, takeovers=0)
        self.episode(0, success)
        self.episode(0, success, suffix='duplicate')
        self.episode(9, success)
        self.episode(1, success, seed=99)
        self.process(0)
        snapshot = monitor.snapshot(self.batch)
        row = snapshot['tasks'][0]
        self.assertEqual(row['attempts_finished'], 4)
        self.assertEqual(row['tested_episodes'], 1)
        self.assertEqual(row['success_episodes'], 0)
        self.assertEqual(row['ambiguous_episodes'], 1)
        self.assertEqual(row['duplicate_layouts'], [0])
        self.assertEqual(row['not_started_episodes'], 2)
        self.assertFalse(row['fully_evaluated'])
        self.assertIn('unexpected_layout_or_seed', [alert['reason'] for alert in snapshot['alerts']])
        self.assertEqual(snapshot['completed_episodes'], 1)
        self.assertEqual(snapshot['requested_episodes'], 3)

    def test_agent_failure_blocked_and_infrastructure_are_separate(self):
        self.episode(0, dict(status='failed', official_terminal=False, official_success=False, failure='agent_aborted: evidence'))
        self.episode(1, dict(status='blocked', official_terminal=False, official_success=None))
        self.episode(2, dict(status='infrastructure_error', official_terminal=None, official_success=None))
        row = self.row()
        self.assertEqual(row['agent_aborted_failures'], 1)
        self.assertEqual(row['official_failures'], 0)
        self.assertEqual(row['blocked_episodes'], 2)
        self.assertEqual(row['infrastructure_errors'], 1)
        self.assertEqual(row['tested_episodes'], 3)

    def test_continuations_keep_original_plan_and_history_and_use_active_request(self):
        self.episode(0, dict(status='blocked', official_terminal=False, official_success=None, retries={'grasp': 3}))
        self.process(1)
        continuation = self.batch / 'new_session'
        write(continuation / 'run_manifest.json', dict(layout_ids=[1, 2], seed=0))
        write(self.batch / 'continuations.json', dict(workers=[dict(task=self.task, session=str(continuation),
                                                                 status_file=str(continuation / 'status.json'))]))
        write(continuation / 'status.json', dict(session=str(continuation), worker_returncode=None, worker_pid=100, codex_pid=101,
                                               codex_thread_id='new-thread', codex_returncode=None, ready=True))
        self.episode(1, dict(status='success', official_terminal=True, official_success=True, corrections=1, takeovers=0), folder=continuation)
        episode = self.episode(2, folder=continuation)
        request = episode / 'request.json'
        write(request, dict(episode=episode.name, request_id='new:1', step=8, retries={'insert': 5}, takeover_count=4))
        write(continuation / 'controller_status.json', dict(status='waiting_for_agent', request=str(request),
                                                          response=str(episode / 'response.json')))
        row = self.row()
        self.assertEqual((row['planned_episodes'], row['tested_episodes'], row['success_episodes']), (3, 2, 1))
        self.assertEqual((row['blocked_episodes'], row['running_episodes'], row['not_started_episodes']), (1, 1, 0))
        self.assertEqual(row['retries'], {'insert': 5})
        self.assertEqual(row['codex_thread_id'], 'new-thread')
        self.assertEqual(row['execution_history'][0]['worker_returncode'], 1)
        self.assertEqual(row['active_worker_session'], str(continuation))
        self.assertEqual(row['expected_layouts'], [0, 1, 2])
        self.assertFalse(row['duplicate_layouts'])

    def test_watch_reexecs_on_source_change_and_records_pid(self):
        class Reexecuted(Exception):
            pass
        args = ['scripts/monitor_parallel.py', '--session', str(self.batch), '--watch']
        with patch.object(monitor.sys, 'argv', args), patch.object(monitor, 'source_signature', side_effect=[(1, 'a'), (2, 'b')]), \
                patch.object(monitor.os, 'execv', side_effect=Reexecuted) as execute:
            with self.assertRaises(Reexecuted):
                monitor.main()
        execute.assert_called_once_with(monitor.sys.executable, [monitor.sys.executable, *args])
        self.assertEqual(json.loads((self.batch / 'monitor_pid.json').read_text())['pid'], monitor.os.getpid())

    def test_actual_step_budget_does_not_claim_unloaded_multiplier(self):
        self.assertEqual(monitor.episode_budget({'step_limit':400}, {'step_limit_multiplier':3}),
                         dict(original_step_limit=400,step_limit_multiplier=1,step_limit=400,configured_step_limit_multiplier=3))
        budget=monitor.episode_budget({'step_limit':1200,'original_step_limit':400,'step_limit_multiplier':3}, {})
        self.assertEqual((budget['step_limit'],budget['step_limit_multiplier']),(1200,3))
        self.assertIsNone(monitor.episode_budget({}, {'step_limit_multiplier':3})['step_limit'])


    def runtime_authorization(self):
        write(self.worker / 'runtime_control.json', dict(status='active', authorization_id='unlimited-v1',
              user_instruction='Unlimited recovery; sub-agent may not terminate.',
              unlimited_recovery=True, allow_agent_termination=False))

    def runtime_response(self, episode, counters):
        request = episode / 'requests/000001/request.json'
        native = dict(episode=episode.name, request_id=episode.name + ':1', step=12,
                      state='LOW_POLICY', takeover_count=0, retries={}, max_takeovers=10,
                      max_retries_per_subgoal=6)
        write(request, native)
        write(request.with_name('response.json'), dict(request_id=native['request_id'], mode='correct',
              runtime_control=dict(authorization_id='unlimited-v1', after=counters)))
        return request

    def test_runtime_completed_logical_counts_exclude_transport_and_preserve_history(self):
        self.runtime_authorization()
        historical = self.episode(0, dict(status='success', official_terminal=True,
              official_success=True, takeovers=0, corrections=0, retries={}))
        # A historical request exists, but no runtime_control response was published.
        write(historical / 'requests/000001/request.json', dict(episode=historical.name,
              request_id=historical.name + ':1', state='LOW_POLICY', takeover_count=0, retries={}))
        episode = self.episode(1, dict(status='success', official_terminal=True,
              official_success=True, takeovers=0, corrections=4, retries={},
              max_retries_per_subgoal=6, max_takeovers=10))
        self.runtime_response(episode, dict(takeovers=2, retries={'grasp': 8}, active=False,
              active_subgoal=None, transport_corrections=3, transport_handbacks=1))
        monitor.os.utime(historical / 'summary.json', (100, 100))
        monitor.os.utime(episode / 'summary.json', (200, 200))
        row = self.row()
        final = next(r for r in row['episodes'] if r['layout_id'] == 1)
        old = next(r for r in row['episodes'] if r['layout_id'] == 0)
        self.assertEqual(final['corrections'], 4)
        self.assertEqual(final['logical_corrections'], 0)
        self.assertEqual(final['logical_takeovers'], 2)
        self.assertEqual(final['logical_retries'], {'grasp': 8})
        self.assertNotIn('runtime_control', old)
        self.assertNotIn('logical_takeovers', old)
        self.assertEqual((row['zero_intervention_successes'], row['assisted_successes']), (1, 1))
        self.assertTrue(row['unlimited_recovery'])
        self.assertEqual(row['takeover_count'], 2)
        self.assertIsNone(row['max_retries_per_subgoal'])

    def test_runtime_live_counts_and_limits_override_native_six_and_ten(self):
        self.runtime_authorization()
        episode = self.episode(0)
        request = self.runtime_response(episode, dict(takeovers=11, retries={'grasp': 15}, active=True,
              active_subgoal='grasp', transport_corrections=7, transport_handbacks=2))
        write(self.worker / 'controller_status.json', dict(status='waiting_for_agent', request=str(request),
              response=str(request.parent / 'next_response.json')))
        row = self.row()
        self.assertTrue(row['unlimited_recovery'])
        self.assertFalse(row['allow_agent_termination'])
        self.assertEqual(row['takeover_count'], 11)
        self.assertEqual(row['retries'], {'grasp': 15})
        self.assertIsNone(row['max_takeovers'])
        self.assertIsNone(row['max_retries_per_subgoal'])
        self.assertEqual(row['native_takeover_count'], 0)
        monitor.save(self.batch, monitor.snapshot(self.batch), 'monitor_runtime_status')
        self.assertFalse((self.batch / 'monitor_status.md').exists())
        markdown = (self.batch / 'monitor_runtime_status.md').read_text()
        self.assertIn('无限/无限', markdown)
        self.assertIn('子会话不得提出终止', markdown)

    def test_runtime_monitor_default_output_does_not_overwrite_legacy_monitor(self):
        legacy = self.batch / 'monitor_status.json'
        legacy.write_text('old monitor evidence')
        write(self.batch / 'runtime_control.json', {'authorization_id': 'unlimited-v1'})
        monitor.save(self.batch, monitor.snapshot(self.batch))
        self.assertEqual(legacy.read_text(), 'old monitor evidence')
        self.assertTrue((self.batch / 'monitor_runtime_status.json').is_file())
        self.assertTrue((self.batch / 'monitor_runtime_status.md').is_file())


if __name__ == '__main__':
    unittest.main()
