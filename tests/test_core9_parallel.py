import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from astra import core9_parallel
from astra.core9_parallel import allocate_pending, gated_workers, validate_worker_plan


class Core9PoolTests(unittest.TestCase):
    def run_pool(self, returncodes, env=None, max_workers=8):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            workers, children = [], []
            for i, rc in enumerate(returncodes):
                session = folder / f'w{i}'
                (session / 'source_snapshot').mkdir(parents=True)
                workers.append(dict(task=f'task{i}', session=str(session)))
                # A robot's official failure is a normal, completed evaluation.
                (session / 'task_scheduler_status.json').write_text(json.dumps(dict(
                    all_layouts_settled=rc == 0, gpu_quarantined=False)))
                (session / 'summary.json').write_text(json.dumps(dict(success=False)))
                child = MagicMock(pid=1000+i, returncode=rc)
                child.poll.return_value = rc
                children.append(child)
            (folder / 'parallel_manifest.json').write_text(json.dumps(dict(workers=workers, max_workers=max_workers)))
            runtime_env = env or {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7', 'SLURM_JOB_NUM_NODES': '1',
                                   'SLURM_JOB_NAME': 'wangzihang_astra', 'SLURM_JOB_ID': 'fixture'}
            with patch.dict(os.environ, runtime_env, clear=True), \
                    patch('sys.argv', ['core9_parallel', '--session', tmp]), \
                    patch.object(core9_parallel.subprocess, 'Popen', side_effect=children), \
                    patch.object(core9_parallel.signal, 'signal'), \
                    patch.object(core9_parallel.time, 'sleep'), \
                    patch('scripts.monitor_parallel.snapshot', return_value={}), \
                    patch('scripts.monitor_parallel.save'):
                result = core9_parallel.main()
            return (result, json.loads((folder / 'parallel_manifest.json').read_text()),
                    json.loads((folder / 'workers_status.json').read_text()),
                    json.loads((folder / 'scheduler_status.json').read_text()))

    def test_all_worker_errors_propagate_after_entire_queue_settles(self):
        result, manifest, records, scheduler = self.run_pool([1] * 11)
        self.assertEqual(result, 1)
        self.assertEqual(manifest['status'], 'stopped_or_error')
        self.assertEqual(scheduler['status'], 'stopped_or_error')
        self.assertEqual(scheduler['queued'], [])
        self.assertEqual(len(scheduler['settled']), 11)
        self.assertEqual([record['returncode'] for record in records], [1] * 11)
        self.assertTrue(all(record['status'] == 'task_error' and record['finalized'] for record in records))
        self.assertIn('task10', manifest['error'])

    def test_one_worker_error_is_not_hidden_by_successful_workers(self):
        result, manifest, records, _ = self.run_pool([0, 1, 0])
        self.assertEqual(result, 1)
        self.assertEqual(manifest['status'], 'stopped_or_error')
        self.assertEqual([record['status'] for record in records], ['settled', 'task_error', 'settled'])

    def test_completed_evaluations_return_zero_even_for_official_failure(self):
        result, manifest, records, scheduler = self.run_pool([0] * 11)
        self.assertEqual(result, 0)
        self.assertEqual(manifest['status'], 'complete_or_blocked_review_results')
        self.assertEqual(scheduler['status'], 'complete_or_blocked_review_results')
        self.assertTrue(all(record['status'] == 'settled' for record in records))
        self.assertNotIn('error', manifest)

    def test_direct_bash_uses_only_the_visible_gpu_as_a_task_slot(self):
        direct = dict(ASTRA_GPU_ALLOCATION_CONFIRMED='1', ASTRA_ASSIGNED_GPUS='0',
                      ASTRA_JOB_ID='bash-fixture', CUDA_VISIBLE_DEVICES='0')
        result, manifest, records, scheduler = self.run_pool([0, 0, 0], env=direct)
        self.assertEqual(result, 0)
        self.assertEqual(manifest['execution_mode'], 'bash_direct')
        self.assertEqual(manifest['concurrency'], 1)
        self.assertEqual(scheduler['max_simultaneous_tasks'], 1)
        self.assertEqual(len(scheduler['settled']), 3)
        self.assertTrue(all(record['returncode'] == 0 for record in records))

    def test_validation_gate_requires_main_review_before_other_slots(self):
        workers = [dict(task='plug_in_charger'), dict(task='hang_mugs')]
        config = dict(workers=workers, startup_validation_gate=dict(task='plug_in_charger'))
        with tempfile.TemporaryDirectory() as tmp:
            approval = Path(tmp) / 'startup_validation_release.json'
            self.assertEqual(gated_workers(tmp, config), workers[:1])
            for invalid in ({}, dict(status='verified', authorized_by='worker', task='plug_in_charger', evidence=['receipt']),
                            dict(status='verified', authorized_by='main_session', task='hang_mugs', evidence=['receipt']),
                            dict(status='verified', authorized_by='main_session', task='plug_in_charger', evidence=[])):
                approval.write_text(json.dumps(invalid))
                self.assertEqual(gated_workers(tmp, config), workers[:1])
            approval.write_text(json.dumps(dict(status='verified', authorized_by='main_session',
                                               task='plug_in_charger', evidence=['receipt'])))
            self.assertEqual(gated_workers(tmp, config), workers)
            self.assertEqual(gated_workers(tmp, dict(workers=workers)), workers)

    def test_eleven_units_use_eight_slots_and_skip_explicit_block(self):
        workers = [dict(task=str(i), runnable=i != 10) for i in range(11)]
        first = allocate_pending(workers, {}, {'10'})
        self.assertEqual([w['task'] for w, _ in first], list(map(str, range(8))))
        active = {w['task']:dict(gpu=gpu) for w,gpu in first}
        self.assertEqual(allocate_pending(workers, active, {'10'}), [])
        del active['3']
        second=allocate_pending(workers, active, {'3','10'})
        self.assertEqual([(w['task'],gpu) for w,gpu in second],[('8',3)])

    def test_nonzero_direct_gpu_id_is_preserved(self):
        workers = [dict(task='task_a'), dict(task='task_b')]
        first = allocate_pending(workers, {}, set(), gpu_ids=[5])
        self.assertEqual([(w['task'], gpu) for w, gpu in first], [('task_a', 5)])
        active = {'task_a': dict(gpu=5)}
        self.assertEqual(allocate_pending(workers, active, set(), gpu_ids=[5]), [])

    def test_user_parallel_limit_caps_allocated_gpu_slots(self):
        result, manifest, records, scheduler = self.run_pool([0] * 3, max_workers=2)
        self.assertEqual(result, 0)
        self.assertEqual(manifest['max_workers'], 2)
        self.assertEqual(manifest['concurrency'], 2)
        self.assertEqual(scheduler['max_simultaneous_tasks'], 2)
        self.assertEqual([record['gpu'] for record in records], [0, 1, 0])

    def test_layout_units_of_the_same_task_are_independent_and_slot_refills(self):
        workers = [dict(worker_id=f'task:layout_{i}', task='task', layout_ids=[i]) for i in range(3)]
        first = allocate_pending(workers, {}, set(), gpu_ids=[0, 1])
        self.assertEqual([(w['worker_id'], gpu) for w, gpu in first],
                         [('task:layout_0', 0), ('task:layout_1', 1)])
        active = {w['worker_id']: dict(gpu=gpu) for w, gpu in first}
        active.pop('task:layout_0')
        refill = allocate_pending(workers, active, {'task:layout_0'}, gpu_ids=[0, 1])
        self.assertEqual([(w['worker_id'], gpu) for w, gpu in refill], [('task:layout_2', 0)])

    def test_one_layout_plan_rejects_multi_episode_workers_and_duplicate_pairs(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            validate_worker_plan(dict(scheduling_unit='one_layout_per_worker', workers=[
                dict(worker_id='multi', task='task', layout_ids=[1, 2])]))
        with self.assertRaisesRegex(ValueError, 'pair may only'):
            validate_worker_plan(dict(scheduling_unit='one_layout_per_worker', workers=[
                dict(worker_id='a', task='task', layout_ids=[1]),
                dict(worker_id='b', task='task', layout_ids=[1])]))
        validate_worker_plan(dict(scheduling_unit='one_layout_per_worker', workers=[
            dict(worker_id='a', task='task', layout_ids=[1]),
            dict(worker_id='b', task='task', layout_ids=[2])]))


if __name__ == '__main__':
    unittest.main()
