import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('prepare_core9', Path(__file__).resolve().parents[1] / 'scripts/prepare_core9.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PrepareCore9Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root / 'cache'
        self.batch = self.root / 'batch'
        for task in prepare.CORE9[:-1]:
            prepare.write_json(self.cache / 'demos' / task / 'demo_context.json',
                               dict(task=task, status='reviewed', segments={'operate': {'frames': [0, 10]}}))

    def fake_prepare(self, command, cwd, check):
        folder = Path(command[command.index('--session') + 1])
        task = command[command.index('--tasks') + 1]
        folder.mkdir(parents=True)
        context = (dict(task=task, status='waived_by_user', segments={'observe': {}})
                   if '--waive-demo' in command else
                   json.loads((self.cache / 'demos' / task / 'demo_context.json').read_text()))
        prepare.write_json(folder / 'demos' / task / 'demo_context.json', context)
        prepare.write_json(folder / 'run_manifest.json', dict(tasks=[task], layout_ids=[0, 1, 2]))

    def test_core9_plan_preserves_missing_demo_and_validated_budgets(self):
        with patch.object(prepare.subprocess, 'run', side_effect=self.fake_prepare) as run, \
                patch.object(prepare, 'snapshot_source'):
            manifest = prepare.prepare_batch(self.batch, self.cache, max_workers=8)
        self.assertEqual(len(manifest['workers']), 33)
        self.assertEqual(manifest['scheduling_unit'], 'one_layout_per_worker')
        self.assertEqual(run.call_count, 30)
        self.assertEqual(manifest['concurrency'], 8)
        self.assertEqual(manifest['status'], 'prepared')
        self.assertIsNone(manifest['job_id'])
        self.assertEqual(manifest['conditioning_mode'], 'l3_l2')
        self.assertEqual(manifest['conditioning_schema_version'], 1)
        self.assertEqual(manifest['conditioning_provider'], 'astra_visual')
        self.assertTrue(manifest['high_level_conditioning_assistance'])
        self.assertEqual(manifest['max_actions_per_decision'], 15)
        self.assertEqual(manifest['high_level_monitor_interval_actions'], 15)
        for call in run.call_args_list:
            command = call.args[0]
            for flag, expected in [('--eval-num', '3'), ('--step-limit-multiplier', '3')]:
                self.assertEqual(command[command.index(flag) + 1], expected)
            self.assertIn('--unlimited-recovery', command)
            self.assertIn('--no-allow-agent-termination', command)
        blocked = manifest['workers'][-1]
        self.assertEqual(blocked['task'], 'general_pickup')
        self.assertEqual(blocked['layout_ids'], [2])
        self.assertFalse(blocked['runnable'])
        self.assertEqual(blocked['blocked_reason'], 'blocked_missing_demo')
        folder = Path(blocked['session'])
        self.assertFalse((folder / 'runtime').exists())
        self.assertTrue((folder / 'blocking.json').is_file())
        planned = json.loads((folder / 'run_manifest.json').read_text())
        self.assertEqual(planned['layout_ids'], [2])
        self.assertEqual(planned['eval_num'], 1)
        self.assertEqual(planned['planned_episodes'], 1)
        self.assertFalse(planned['execution_started'])
        self.assertEqual(planned['conditioning_mode'], 'l3_l2')
        self.assertTrue(planned['high_level_conditioning_assistance'])
        statuses = json.loads((self.batch / 'workers_status.json').read_text())
        self.assertEqual(statuses[-1]['status'], 'blocked')
        self.assertTrue(all(w['returncode'] is None and w['gpu'] is None for w in statuses))
        self.assertFalse(list(folder.rglob('summary.json')))

    def test_validate_only_never_prepares_or_writes(self):
        with patch.object(prepare.subprocess, 'run') as run, redirect_stdout(io.StringIO()) as output:
            prepare.main(['--session', str(self.batch), '--demo-cache', str(self.cache),
                          '--max-workers', '8', '--validate-only'])
        run.assert_not_called()
        self.assertFalse(self.batch.exists())
        self.assertEqual(len(json.loads(output.getvalue())['workers']), 33)

    def test_parallel_limit_is_user_configurable_and_persisted(self):
        with patch.object(prepare.subprocess, 'run', side_effect=self.fake_prepare), \
                patch.object(prepare, 'snapshot_source'):
            manifest = prepare.prepare_batch(self.batch, self.cache, tasks=['plug_in_charger'],
                                             task_layout_ids={'plug_in_charger': [0, 1]}, max_workers=2)
        self.assertEqual(manifest['max_workers'], 2)
        self.assertEqual(manifest['concurrency'], 2)
        self.assertEqual(len(manifest['workers']), 2)
        self.assertEqual([w['layout_ids'] for w in manifest['workers']], [[0], [1]])

    def test_unreviewed_and_mismatched_contexts_stay_blocked(self):
        for task, context in [
            ('hang_mugs', dict(task='hang_mugs', status='awaiting_visual_review', segments={'operate': {}})),
            ('hang_mugs_random', dict(task='hang_mugs', status='reviewed', segments={'operate': {}})),
            ('sort_nesting_dolls_by_size', dict(task='sort_nesting_dolls_by_size', status='blocked_demo_decode')),
            ('classify_objects', None),
        ]:
            prepare.write_json(self.cache / 'demos' / task / 'demo_context.json', context)
        workers = prepare.plan_workers(self.batch, self.cache)
        self.assertTrue(all(not w['runnable'] for w in workers if w['task'] in {
            'hang_mugs', 'hang_mugs_random', 'sort_nesting_dolls_by_size'}))
        nesting = [w for w in workers if w['task'] == 'sort_nesting_dolls_by_size']
        self.assertTrue(all(w['blocked_reason'] == 'blocked_demo_decode' for w in nesting))
        self.assertFalse(next(w for w in workers if w['task'] == 'classify_objects')['runnable'])

    def test_existing_session_is_not_overwritten(self):
        self.batch.mkdir()
        marker = self.batch / 'keep.txt'
        marker.write_text('old evidence')
        with patch.object(prepare.subprocess, 'run') as run, self.assertRaises(FileExistsError):
            prepare.prepare_batch(self.batch, self.cache, max_workers=8)
        run.assert_not_called()
        self.assertEqual(marker.read_text(), 'old evidence')

    def test_explicit_general_pickup_waiver_prepares_without_cached_demo(self):
        with patch.object(prepare.subprocess, 'run', side_effect=self.fake_prepare) as run, \
                patch.object(prepare, 'snapshot_source'):
            manifest = prepare.prepare_batch(self.batch, self.cache, waive_demo_tasks=['general_pickup'], max_workers=8)
        self.assertEqual(run.call_count, 33)
        self.assertEqual(manifest['authorized_demo_waivers'], ['general_pickup'])
        general = manifest['workers'][-1]
        self.assertEqual(general['task'], 'general_pickup')
        self.assertEqual(general['layout_ids'], [2])
        self.assertTrue(general['runnable'])
        self.assertTrue(general['demo_waived'])
        self.assertIsNone(general['demo_context_path'])
        command = run.call_args_list[-1].args[0]
        self.assertIn('--waive-demo', command)
        self.assertNotIn('--demo-cache', command)
        self.assertFalse((Path(general['session']) / 'blocking.json').exists())
        self.assertTrue(all('--waive-demo' not in c.args[0] for c in run.call_args_list[:-3]))

    def test_other_task_waivers_are_rejected_before_any_write(self):
        with patch.object(prepare.subprocess, 'run') as run, self.assertRaises(ValueError):
            prepare.prepare_batch(self.batch, self.cache, waive_demo_tasks=['hang_mugs'], max_workers=8)
        run.assert_not_called()
        self.assertFalse(self.batch.exists())

    def test_recovery_batch_selects_layouts_and_links_prior_session(self):
        prior = self.root / 'prior'
        prior.mkdir()
        layouts = {'plug_in_charger': [2], 'general_pickup': [0, 1, 2]}
        tasks = list(layouts)
        with patch.object(prepare.subprocess, 'run', side_effect=self.fake_prepare), \
                patch.object(prepare, 'snapshot_source'):
            manifest = prepare.prepare_batch(self.batch, self.cache, eval_num=3,
                waive_demo_tasks=['general_pickup'], tasks=tasks,
                task_layout_ids=layouts, recovery_of=prior, max_workers=8)
        self.assertEqual(manifest['tasks'], tasks)
        self.assertEqual(manifest['planned_layouts'], layouts)
        self.assertEqual(manifest['recovery_of'], str(prior.resolve()))
        for worker in manifest['workers']:
            worker_manifest = json.loads((Path(worker['session']) / 'run_manifest.json').read_text())
            self.assertEqual(worker_manifest['layout_ids'], worker['layout_ids'])
            self.assertEqual(len(worker['layout_ids']), 1)
            self.assertEqual(worker_manifest['eval_num'], 1)
            self.assertEqual(worker_manifest['recovery_of'], str(prior.resolve()))

    def test_invalid_task_layout_selection_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare.plan_workers(self.batch, self.cache, tasks=['plug_in_charger'],
                                 task_layout_ids={'plug_in_charger': [0, 3]})


if __name__ == '__main__':
    unittest.main()
