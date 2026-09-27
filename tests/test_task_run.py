import unittest
import json
import tempfile
import time
from unittest.mock import patch
from pathlib import Path
from astra.task_run import remaining_layouts, supervisor_unavailable, make_prompt, project_root, external_supervisor_available, cleanup_verified, main


class TaskSchedulerTests(unittest.TestCase):
    def test_external_heartbeat_and_exit_detection(self):
        status = dict(pid=123, owner_id='remote', returncode=None, ready=True, updated_at=100)
        self.assertTrue(external_supervisor_available(status, 150))
        self.assertFalse(external_supervisor_available(status, 221))
        self.assertFalse(external_supervisor_available(dict(status, returncode=0), 150))
        self.assertFalse(external_supervisor_available({}, 150))
        self.assertFalse(external_supervisor_available(dict(status, ready=False), 150))
        self.assertTrue(external_supervisor_available(dict(status, ready=False, thread_id='started'), 150))
        self.assertFalse(external_supervisor_available(dict(status, ready=False, thread_id='started', returncode=0), 150))
        self.assertEqual(status['updated_at'], 100)

    def test_cleanup_requires_finished_audit_and_all_child_exits(self):
        good = dict(finished_at=1, policy_pid=10, simulator_pid=11,
                    cleanup_errors=[], cleanup=[dict(pid=10, returncode=-15), dict(pid=11, returncode=0)])
        self.assertTrue(cleanup_verified(good, 0))
        self.assertFalse(cleanup_verified(good, -9))
        self.assertFalse(cleanup_verified(dict(good, cleanup_errors=['still alive']), 1))
        self.assertFalse(cleanup_verified(dict(good, cleanup=[dict(pid=10, returncode=-15)]), 0))
        self.assertFalse(cleanup_verified({}, 0))
        self.assertTrue(cleanup_verified(dict(status='runtime_error', finished_at=1), 1))
        self.assertTrue(cleanup_verified({}, 127, runner_started=False))

    def test_external_mode_never_owns_or_starts_remote_codex(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'source_snapshot').mkdir()
            (folder / 'run_manifest.json').write_text(json.dumps(dict(tasks=['cover_blocks'], layout_ids=[0, 1, 2])))
            owner = {'owner_id': 'external-owner'}
            (folder / 'supervisor_owner.json').write_text(json.dumps(owner))
            remote = dict(pid=7654321, owner_id='external-owner', returncode=None,
                          updated_at=time.time(), thread_id='real-external-thread', ready=False)
            status_path = folder / 'supervisor_status.json'
            status_path.write_text(json.dumps(remote))
            started = []
            class FakeRunner:
                pid = 999991
                def __init__(self, cmd, **kwargs):
                    self.returncode = 0
                    assert cmd[1:3] == ['-m', 'astra.run']
                    layout = int(cmd[cmd.index('--layout-id') + 1]); started.append(layout)
                    run_dir = folder / 'runs' / f'layout_{layout}'
                    run_dir.mkdir(parents=True)
                    (run_dir / 'run_status.json').write_text(json.dumps(dict(finished_at=1, cleanup=[], cleanup_errors=[])))
                    summary = folder / 'cover_blocks' / 'seed_0' / f'layout_{layout}_mock' / 'summary.json'
                    summary.parent.mkdir(parents=True); summary.write_text('{}')
                def poll(self):
                    return self.returncode
                def wait(self, timeout=None):
                    return self.returncode
            def advance(_):
                scheduler = json.loads((folder / 'task_scheduler_status.json').read_text())
                if scheduler['all_layouts_settled']:
                    remote['returncode'] = 0
                    status_path.write_text(json.dumps(remote))
                    (folder / 'supervisor_summary.json').write_text('{}')
            with patch.dict('os.environ', {'SLURM_JOB_ID': 'mock'}), \
                 patch('astra.task_run.subprocess.Popen', FakeRunner), \
                 patch('astra.task_run.time.sleep', advance), \
                 patch('astra.task_run.signal.signal'), \
                 patch('astra.task_run.fcntl.flock', side_effect=AssertionError('external mode cannot lock')), \
                 patch('astra.task_run.os.killpg', side_effect=AssertionError('remote PID must not be signalled')):
                result = main(['--session', str(folder), '--batch', str(folder), '--gpu', '0', '--external-supervisor'])
            self.assertEqual(result, 0)
            self.assertEqual(started, [0, 1, 2])
            self.assertEqual(json.loads(status_path.read_text()), remote)
            self.assertEqual(json.loads((folder / 'supervisor_owner.json').read_text()), owner)
            self.assertFalse((folder / 'supervisor.lock').exists())
            scheduler = json.loads((folder / 'task_scheduler_status.json').read_text())
            self.assertTrue(scheduler['all_layouts_settled'])
            self.assertEqual(scheduler['codex']['updated_at'], remote['updated_at'])

    def test_lost_supervisor_preserves_live_runner_until_supervision_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'source_snapshot').mkdir()
            (folder / 'run_manifest.json').write_text(json.dumps(dict(tasks=['cover_blocks'], layout_ids=[0, 1])))
            status_path = folder / 'supervisor_status.json'
            remote = dict(pid=123, owner_id='external', returncode=None, ready=True, updated_at=time.time())
            status_path.write_text(json.dumps(remote))
            started, runners = [], []
            controller = dict(status='waiting_for_agent', request='pending_request', response='pending_response')
            class LiveRunner:
                pid = 999991
                def __init__(self, cmd, **kwargs):
                    layout = int(cmd[cmd.index('--layout-id') + 1])
                    started.append(layout)
                    runners.append(self)
                    self.returncode = None if layout == 0 else 0
                    run_dir = folder / 'runs' / f'layout_{layout}'
                    run_dir.mkdir(parents=True)
                    (run_dir / 'run_status.json').write_text(json.dumps(dict(finished_at=1, cleanup=[], cleanup_errors=[])))
                    summary = folder / 'cover_blocks' / 'seed_0' / f'layout_{layout}_mock' / 'summary.json'
                    summary.parent.mkdir(parents=True)
                    summary.write_text('{}')
                    if layout == 0:
                        (folder / 'controller_status.json').write_text(json.dumps(controller))
                        status_path.write_text(json.dumps(dict(remote, returncode=1)))
                def poll(self):
                    return self.returncode
            pauses = []
            def advance(_):
                scheduler = json.loads((folder / 'task_scheduler_status.json').read_text())
                if scheduler['status'] == 'paused_for_main':
                    self.assertEqual(started, [0])
                    self.assertIsNone(runners[0].returncode)
                    self.assertEqual(scheduler['settled_layouts'], [])
                    self.assertEqual(scheduler['remaining_layouts'], [1])
                    self.assertEqual(json.loads((folder / 'controller_status.json').read_text()), controller)
                    pauses.append(scheduler['supervisor_pause']['pause_id'])
                    if len(pauses) == 2:
                        remote['updated_at'] = time.time()
                        status_path.write_text(json.dumps(remote))
                elif scheduler['status'] == 'running_layout':
                    self.assertEqual(len(pauses), 2)
                    runners[0].returncode = 0
                elif scheduler['all_layouts_settled']:
                    status_path.write_text(json.dumps(dict(remote, returncode=0)))
                    (folder / 'supervisor_summary.json').write_text('{}')
                else:
                    self.fail(f'Unexpected wait state: {scheduler["status"]}')
            with patch.dict('os.environ', {'SLURM_JOB_ID': 'mock'}), \
                 patch('astra.task_run.subprocess.Popen', LiveRunner), \
                 patch('astra.task_run.time.sleep', advance), \
                 patch('astra.task_run.signal.signal'), \
                 patch('astra.task_run.os.killpg', side_effect=AssertionError('live scene must not be signalled')):
                result = main(['--session', str(folder), '--batch', str(folder), '--gpu', '0', '--external-supervisor'])
            self.assertEqual(result, 0)
            self.assertEqual(started, [0, 1])
            self.assertEqual(pauses[0], pauses[1])
            incidents = [json.loads(line) for line in (folder / 'task_supervisor_incidents.jsonl').read_text().splitlines()]
            self.assertEqual([item['event'] for item in incidents], ['supervisor_lost', 'supervisor_available'])
            self.assertEqual(incidents[0]['controller'], controller)
            self.assertEqual(json.loads((folder / 'supervisor_loss_incident.json').read_text())['status'], 'supervision_restored')

    def test_missing_cleanup_quarantines_gpu_before_next_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'source_snapshot').mkdir()
            (folder / 'run_manifest.json').write_text(json.dumps(dict(tasks=['cover_blocks'], layout_ids=[0, 1, 2])))
            (folder / 'supervisor_status.json').write_text(json.dumps(dict(pid=123, owner_id='external', returncode=None, ready=True, updated_at=time.time())))
            started = []
            class UncleanRunner:
                pid = 999999
                returncode = 1
                def __init__(self, cmd, **kwargs):
                    started.append(int(cmd[cmd.index('--layout-id') + 1]))
                def poll(self):
                    return self.returncode
            with patch.dict('os.environ', {'SLURM_JOB_ID': 'mock'}), \
                 patch('astra.task_run.subprocess.Popen', UncleanRunner), \
                 patch('astra.task_run.signal.signal'):
                result = main(['--session', str(folder), '--batch', str(folder), '--gpu', '0', '--external-supervisor'])
            state = json.loads((folder / 'task_scheduler_status.json').read_text())
            self.assertEqual(result, 4)
            self.assertEqual(started, [0])
            self.assertTrue(state['gpu_quarantined'])
            self.assertEqual(state['status'], 'cleanup_unverified')
            self.assertEqual(state['not_run_layouts'], [1, 2])
            self.assertEqual(state['layouts'][1]['state'], 'not_run_gpu_quarantined')

    def test_failed_started_layout_is_not_retried(self):
        self.assertEqual(remaining_layouts([0, 1, 2], [0]), [1, 2])
        self.assertEqual(remaining_layouts([0, 1, 2], [0, 1, 2]), [])

    def test_bad_plan_is_rejected(self):
        with self.assertRaises(ValueError):
            remaining_layouts([0, 0], [])
        with self.assertRaises(ValueError):
            remaining_layouts([0, 1], [2])

    def test_exit_zero_still_means_loss_of_supervision(self):
        self.assertTrue(supervisor_unavailable(0, [0, 1, 2], [0], False))
        self.assertTrue(supervisor_unavailable(0, [0, 1], [0, 1], True))
        self.assertFalse(supervisor_unavailable(None, [0, 1], [], True))
        self.assertFalse(supervisor_unavailable(0, [0, 1], [0, 1], False))

    def test_waiver_is_scoped_to_manifest_task(self):
        root = Path(__file__).resolve().parents[1]
        worker = root / 'outputs/test/workers/test'
        manifest = {'authorized_demo_waivers': ['general_pickup']}
        waived = make_prompt(worker, root / 'outputs/test', 'general_pickup', [0, 1, 2], 'owner', manifest)
        normal = make_prompt(worker, root / 'outputs/test', 'cover_blocks', [0, 1, 2], 'owner', manifest)
        self.assertIn('仅任务 general_pickup 无训练示范执行', waived)
        self.assertIn('reviewed_demo_images=[]', waived)
        self.assertIn('不能声称已看训练视频', waived)
        self.assertNotIn('必须实际 view_image 查看示范 contact_sheets', waived)
        self.assertIn('必须实际 view_image 查看示范 contact_sheets', normal)
        self.assertNotIn('demo_waiver=true', normal)

    def test_root_discovery_ignores_snapshot_location(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(project_root(root / 'outputs/test/workers/test'), root)

    def test_l2_supervisor_prompt_prepares_conditions_before_inference(self):
        root = Path(__file__).resolve().parents[1]
        text = make_prompt(root / 'outputs/test/workers/test', root / 'outputs/test',
                           'cover_blocks', [0, 1, 2], 'owner', {'conditioning_mode': 'l3_l2'})
        for required in ('PREPARE_CONDITIONING', 'condition 0', '--conditioning-file',
                         '--conditioning-version', 'l2_instruction', 'cue_rgb_path',
                         'l2_bboxes_xyxy', 'box_source', 'unboxed_reason',
                         'validated_visual_locator', '--scene-review "当前场景审阅证据"',
                         'observation.images.cam_head', 'conditioning_version',
                         '同一子目标沿用固定起始 cue', '不能归为 low policy 独立成功'):
            self.assertIn(required, text)
        self.assertIn('condition/approve/correct/takeover/replan/hand_back', text)
        self.assertNotIn('保持本局实际模式', text)

    def test_supervisor_uses_recorded_monitoring_interval(self):
        root = Path(__file__).resolve().parents[1]
        for manifest, interval in [({}, 4), ({'max_actions_per_decision': 15,
                                            'high_level_monitor_interval_actions': 15}, 15)]:
            text = make_prompt(root / 'outputs/test/workers/test', root / 'outputs/test',
                               'cover_blocks', [0], 'owner', manifest)
            self.assertIn(f'实际高层监控间隔为 {interval} 次环境动作', text)
            self.assertIn('--short-batch-reason', text)
            self.assertIn('replan_steps=16', text)
            self.assertNotIn('每次只批准 1–4', text)

    def test_single_layout_supervisor_releases_slot_without_waiting_for_same_task(self):
        root = Path(__file__).resolve().parents[1]
        text = make_prompt(root / 'outputs/test/workers/test', root / 'outputs/test',
                           'hang_mugs_random', [2], 'owner')
        self.assertIn('只负责 layout=2 这一局', text)
        self.assertIn('从全局队列分配下一个独立 worker', text)
        self.assertIn('分母为 1', text)
        self.assertNotIn('监督全部计划布局', text)

    def test_legacy_manifest_does_not_claim_l2_conditions(self):
        root = Path(__file__).resolve().parents[1]
        for manifest in ({}, {'conditioning_mode': 'pure_l3'}):
            text = make_prompt(root / 'outputs/test/workers/test', root / 'outputs/test',
                               'cover_blocks', [0, 1, 2], 'owner', manifest)
            self.assertIn('conditioning_mode=pure_l3', text)
            self.assertNotIn('PREPARE_CONDITIONING', text)
            self.assertNotIn('condition/approve/', text)

    def test_unknown_conditioning_mode_fails_before_supervisor_launch(self):
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, 'Unsupported conditioning_mode'):
            make_prompt(root / 'outputs/test/workers/test', root / 'outputs/test',
                        'cover_blocks', [0], 'owner', {'conditioning_mode': 'typo'})

    def test_prompt_uses_entire_new_plan_without_historical_handoff(self):
        root = Path(__file__).resolve().parents[1]
        text = make_prompt(root / 'outputs/test/workers/test', root / 'outputs/test', 'cover_blocks', [0, 1, 2], 'owner')
        self.assertIn('all_layouts_settled=true', text)
        self.assertIn('unlimited_recovery=true', text)
        self.assertIn('allow_agent_termination=false', text)
        self.assertIn('supervisor_ready.json', text)
        self.assertNotIn('10176', text)
        self.assertNotIn('约28步', text)

    def test_direct_bash_prompt_does_not_claim_slurm_allocation(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            (batch / 'parallel_manifest.json').write_text(json.dumps({'execution_mode': 'bash_direct'}))
            text = make_prompt(root / 'outputs/test/workers/test', batch,
                               'cover_blocks', [0], 'owner', {})
        self.assertIn('主会话通过 bash 分配', text)
        self.assertNotIn('当前资源由 Slurm 分配', text)


if __name__ == '__main__':
    unittest.main()
