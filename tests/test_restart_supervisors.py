import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import restart_supervisors as restart
from scripts import launch_core9_supervisors as manager


class RestartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.schedule = dict(status='paused_for_main', manager_pid=os.getpid(), all_layouts_settled=False)
        self.old = dict(pid=99999999, hostname=socket.gethostname(), owner_id='old',
                        status='exited', returncode=0, thread_id='old-thread')
        self.write('task_scheduler_status.json', self.schedule)
        self.write('supervisor_status.json', self.old)
        self.write('supervisor_owner.json', dict(owner_id='old', status='exited'))
        self.write('supervisor_ready.json', dict(owner_id='old'))
        self.write('run_manifest.json', dict(layout_ids=[2]))
        self.write('request.json', dict(request_id='ep:21', episode='ep', step=156))
        self.write('controller_status.json', dict(status='waiting_for_agent',
                   request=str(self.folder/'request.json'), response=str(self.folder/'response.json')))

    def write(self, name, data):
        (self.folder/name).write_text(json.dumps(data))

    def test_exit_zero_restarts_but_alive_remote_finished_or_draining_does_not(self):
        self.assertTrue(restart.restart_eligibility(self.folder)[0])
        self.write('supervisor_status.json', dict(self.old, pid=os.getpid()))
        self.assertFalse(restart.restart_eligibility(self.folder)[0])
        self.write('supervisor_status.json', dict(self.old, hostname='remote'))
        self.assertFalse(restart.restart_eligibility(self.folder)[0])
        self.write('supervisor_status.json', self.old)
        self.write('response.json', {})
        self.assertFalse(restart.restart_eligibility(self.folder)[0])
        (self.folder/'response.json').unlink()
        self.write('task_scheduler_status.json', dict(self.schedule, all_layouts_settled=True))
        self.assertFalse(restart.restart_eligibility(self.folder)[0])

    def test_backoff_is_bounded(self):
        self.assertEqual([restart.retry_delay(x) for x in [1,2,3,20]], [5,10,20,300])

    def test_stalled_pending_request_is_detected_only_after_request_and_event_idle_timeout(self):
        events = self.folder/'events.jsonl'
        events.write_text('{"type":"turn.started"}\n')
        request = self.folder/'request.json'
        response = self.folder/'response.json'
        old = time.time() - 100
        os.utime(request, (old, old))
        os.utime(events, (old, old))
        entry = dict(folder=self.folder, state=dict(events=str(events), started_at=old,
                     last_activity_at=old, activity_size=events.stat().st_size,
                     activity_mtime_ns=events.stat().st_mtime_ns))
        self.write('controller_status.json', dict(status='waiting_for_agent', request=str(request),
                   response=str(response)))
        self.assertIsNone(restart.stalled_request(entry, time.time(), 120))
        time.sleep(0.01)
        now = time.time() + 30
        stall = restart.stalled_request(entry, now, 120)
        self.assertEqual(stall['request_id'], 'ep:21')
        self.assertGreaterEqual(stall['idle_seconds'], 120)
        response.write_text('{}')
        self.assertIsNone(restart.stalled_request(entry, now + 1, 120))

    def test_stalled_timeout_pause_is_detected_from_incident_request_id(self):
        events = self.folder/'events.jsonl'
        events.write_text('{"type":"turn.started"}\n')
        request = self.folder/'request.json'
        old = time.time() - 100
        os.utime(events, (old, old))
        os.utime(request, (old, old))
        self.write('conditioning_incident.json', dict(
            status='paused_for_main', pause_id='pause', request_id='ep:21',
            error='TimeoutError: Agent decision timeout; no unsupervised fallback'))
        self.write('controller_status.json', dict(status='paused_for_main', pause_id='pause',
                   request_id='ep:21', episode='ep'))
        entry = dict(folder=self.folder, state=dict(events=str(events), started_at=old,
                     last_activity_at=old, activity_size=events.stat().st_size,
                     activity_mtime_ns=events.stat().st_mtime_ns))
        stall = restart.stalled_request(entry, time.time() + 30, 120)
        self.assertEqual(stall['request_id'], 'ep:21')
        restart.record_stall_loss(self.folder, dict(stall, request_path=str(request)))
        loss = json.loads((self.folder/'supervisor_loss_incident.json').read_text())
        self.assertEqual(loss['controller']['request'], str(request))
        self.assertEqual(loss['pause_id'], 'pause')

    def test_real_child_handoff_retains_request_and_prevents_duplicate_owner(self):
        executable = self.folder/'fake-codex'
        executable.write_text('#!/usr/bin/env python3\nimport time\nprint(\'{"type":"thread.started","thread_id":"replacement-thread"}\',flush=True)\ntime.sleep(2)\n')
        executable.chmod(0o755)
        before = (self.folder/'request.json').read_bytes()
        with patch.object(manager, 'make_prompt', return_value='本次是全新隔离测试，不是接手旧局；'):
            entry = manager.launch(dict(task='task',session=str(self.folder)),self.folder,str(executable),'new',restart=True)
        try:
            self.assertEqual((self.folder/'request.json').read_bytes(),before)
            self.assertFalse((self.folder/'response.json').exists())
            self.assertFalse(entry['state']['ready'])  # Old ready never transfers.
            with self.assertRaises(BlockingIOError):
                manager.launch(dict(task='task',session=str(self.folder)),self.folder,str(executable),'duplicate',restart=True)
            entry['child'].wait(timeout=5)
            manager.refresh(entry)
            self.assertEqual(entry['state']['thread_id'],'replacement-thread')
            archive=Path(entry['state']['handoff_record']).parent
            self.assertEqual(json.loads((archive/'handoff.json').read_text())['step'],156)
            self.assertEqual(json.loads((archive/'prior_supervisor_status.json').read_text())['thread_id'],'old-thread')
        finally:
            manager.stop_child(entry['child'])
            manager.release(entry)

    def test_failed_spawn_remains_retryable(self):
        with patch.object(manager, 'make_prompt', return_value='handoff'):
            with self.assertRaises(FileNotFoundError):
                manager.launch(dict(task='task', session=str(self.folder)), self.folder,
                               '/nonexistent/codex', 'new', restart=True)
        self.assertEqual(json.loads((self.folder/'supervisor_status.json').read_text())['status'], 'launch_error')
        self.assertTrue(restart.restart_eligibility(self.folder)[0])

    def test_watchdog_restarts_repeated_premature_exit_and_stops_when_done(self):
        executable = self.folder/'codex'
        executable.write_text('#!/usr/bin/env python3\nprint(\'{"type":"thread.started","thread_id":"fake-thread"}\',flush=True)\n')
        executable.chmod(0o755)
        config = dict(status='running', workers=[dict(task='task', session=str(self.folder))])
        self.write('parallel_manifest.json', config)
        started = time.monotonic()
        def poll_delay(_):
            time.sleep(0.02)
            records = self.folder/'supervisor_restarts.jsonl'
            if records.exists() and len(records.read_text().splitlines()) >= 2:
                self.write('task_scheduler_status.json', dict(self.schedule, all_layouts_settled=True))
                self.write('parallel_manifest.json', dict(config, status='finished'))
            if time.monotonic() - started > 5:
                raise AssertionError('watchdog did not restart twice')
        # Patch only the watchdog's time facade; child monitoring still uses real time.
        from types import SimpleNamespace
        clock = SimpleNamespace(time=time.time, sleep=poll_delay)
        with patch.object(manager, 'make_prompt', return_value='handoff'), \
             patch.object(restart, 'supervisor_executable', return_value=str(executable)), \
             patch.object(restart, 'time', clock), \
             patch.object(restart, 'retry_delay', return_value=0), \
             patch.object(restart.signal, 'signal'), \
             patch('sys.argv', ['restart_supervisors.py', '--session', str(self.folder)]):
            self.assertEqual(restart.main(), 0)
        records = [json.loads(line) for line in (self.folder/'supervisor_restarts.jsonl').read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0]['owner_id'], records[1]['owner_id'])
        self.assertEqual(json.loads((self.folder/'supervisor_restart_status.json').read_text())['status'], 'stopped')
        self.assertFalse((self.folder/'response.json').exists())

    def test_watchdog_restarts_live_supervisor_with_stalled_pending_request(self):
        executable = self.folder/'codex'
        executable.write_text(
            '#!/usr/bin/env python3\nimport time\n'
            'print(\'{"type":"thread.started","thread_id":"stalled-thread"}\',flush=True)\n'
            'time.sleep(5)\n')
        executable.chmod(0o755)
        config = dict(status='running', workers=[dict(task='task', session=str(self.folder))])
        self.write('parallel_manifest.json', config)
        started = time.monotonic()
        def poll_delay(_):
            records = self.folder/'supervisor_stalls.jsonl'
            if records.exists() and records.read_text().splitlines():
                self.write('task_scheduler_status.json', dict(self.schedule, all_layouts_settled=True))
                self.write('parallel_manifest.json', dict(config, status='finished'))
            if time.monotonic() - started > 5:
                raise AssertionError('watchdog did not record a live supervisor stall')
            time.sleep(0.01)
        from types import SimpleNamespace
        clock = SimpleNamespace(time=time.time, sleep=poll_delay)
        with patch.object(manager, 'make_prompt', return_value='handoff'), \
             patch.object(restart, 'supervisor_executable', return_value=str(executable)), \
             patch.object(restart, 'time', clock), \
             patch.object(restart.signal, 'signal'), \
             patch('sys.argv', ['restart_supervisors.py', '--session', str(self.folder), '--stall-timeout', '0.01']):
            self.assertEqual(restart.main(), 0)
        records = [json.loads(line) for line in (self.folder/'supervisor_stalls.jsonl').read_text().splitlines()]
        self.assertEqual(records[0]['reason'], 'supervisor_no_response')
        self.assertEqual(records[0]['request_id'], 'ep:21')

    def test_only_diagnosed_supervisor_timeout_resumes_after_ready(self):
        incident=dict(pause_id='pause',request_id='ep:21',error='TimeoutError: Agent decision timeout; no unsupervised fallback')
        self.write('conditioning_incident.json',incident)
        self.write('controller_status.json',dict(status='paused_for_main',pause_id='pause'))
        state=dict(ready=True,owner_id='new')
        restart.resume_supervisor_timeout(self.folder,state)
        self.assertFalse((self.folder/'conditioning_resume.json').exists())
        self.write('supervisor_loss_incident.json',dict(event='supervisor_lost',controller=dict(request_id='ep:21', episode='ep')))
        restart.resume_supervisor_timeout(self.folder,dict(state,ready=False))
        self.assertFalse((self.folder/'conditioning_resume.json').exists())
        restart.resume_supervisor_timeout(self.folder,state)
        self.assertEqual(json.loads((self.folder/'conditioning_resume.json').read_text())['pause_id'],'pause')
        (self.folder/'conditioning_resume.json').unlink()
        self.write('conditioning_incident.json',dict(incident,error='receipt mismatch'))
        restart.resume_supervisor_timeout(self.folder,state)
        self.assertFalse((self.folder/'conditioning_resume.json').exists())


if __name__ == '__main__':
    unittest.main()
