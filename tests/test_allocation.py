import os
import unittest
from unittest.mock import patch

from astra.allocation import allocated_gpu_ids, runtime_job_id


class RuntimeAllocationTests(unittest.TestCase):
    def test_direct_bash_requires_matching_explicit_single_gpu(self):
        env = dict(ASTRA_GPU_ALLOCATION_CONFIRMED='1', ASTRA_ASSIGNED_GPUS='0',
                   ASTRA_JOB_ID='bash-gpu04-test', CUDA_VISIBLE_DEVICES='0')
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(allocated_gpu_ids(require_core9=True), [0])
            self.assertEqual(runtime_job_id(), 'bash-gpu04-test')

    def test_direct_bash_rejects_unconfirmed_or_mismatched_gpu_list(self):
        for env in (
            dict(ASTRA_ASSIGNED_GPUS='0', ASTRA_JOB_ID='test', CUDA_VISIBLE_DEVICES='0'),
            dict(ASTRA_GPU_ALLOCATION_CONFIRMED='1', ASTRA_ASSIGNED_GPUS='1',
                 ASTRA_JOB_ID='test', CUDA_VISIBLE_DEVICES='0'),
        ):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaises(RuntimeError):
                    allocated_gpu_ids(require_core9=True)

    def test_slurm_core9_requires_full_named_single_node_allocation(self):
        env = dict(SLURM_JOB_ID='123', SLURM_JOB_NAME='wangzihang_astra',
                   SLURM_JOB_NUM_NODES='1', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(allocated_gpu_ids(require_core9=True), list(range(8)))
            self.assertEqual(runtime_job_id(), '123')

    def test_slurm_single_gpu_slot_is_valid_for_user_selected_concurrency(self):
        env = dict(SLURM_JOB_ID='123', SLURM_JOB_NAME='wangzihang_astra',
                   SLURM_JOB_NUM_NODES='1', CUDA_VISIBLE_DEVICES='5')
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(allocated_gpu_ids(), [5])

    def test_slurm_single_gpu_slot_is_valid_for_user_selected_concurrency(self):
        env = dict(SLURM_JOB_ID='123', SLURM_JOB_NAME='wangzihang_astra',
                   SLURM_JOB_NUM_NODES='1', CUDA_VISIBLE_DEVICES='5')
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(allocated_gpu_ids(), [5])


if __name__ == '__main__':
    unittest.main()
