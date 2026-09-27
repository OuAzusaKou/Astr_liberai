import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

from astra import bridge
from astra import parallel
from test_bridge import Env, action


def fake_worker(folder):
    os.environ['ASTRA_SESSION'] = folder
    os.environ['ASTRA_CONTROLLER_TIMEOUT'] = '15'
    env = Env()
    calls = []
    def call(func_name, **kwargs):
        calls.append(func_name)
        if func_name == 'get_action':
            return [action(0.02)] * 4
    bridge.eval_one_episode(env, SimpleNamespace(call=call))
    bridge.atomic_json(Path(folder) / 'model_calls.json', calls)


class ParallelTests(unittest.TestCase):
    def test_failed_worker_is_propagated_to_batch_and_final_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            workers = []
            for i in range(8):
                session = folder / f'w{i}'
                (session / 'source_snapshot').mkdir(parents=True)
                workers.append(dict(task=f'task{i}', session=str(session), gpu=i))
            bridge.atomic_json(folder / 'parallel_manifest.json', {'workers': workers})
            children = []
            for i in range(8):
                child = MagicMock(pid=1000+i, returncode=1 if i == 4 else 0)
                child.poll.return_value = child.returncode
                children.append(child)
            sock = MagicMock()
            sock.getsockname.return_value = ('127.0.0.1', 40000)
            env = {'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7', 'SLURM_JOB_NUM_NODES': '1',
                   'SLURM_JOB_NAME': 'wangzihang_astra', 'SLURM_JOB_ID': 'fixture'}
            with patch.dict(os.environ, env), patch('sys.argv', ['parallel', '--session', tmp]), \
                    patch.object(parallel.subprocess, 'Popen', side_effect=children), \
                    patch.object(parallel.socket, 'socket', return_value=sock), \
                    patch.object(parallel.signal, 'signal'):
                with self.assertRaisesRegex(RuntimeError, 'task4'):
                    parallel.main()
            status = json.loads((folder / 'workers_status.json').read_text())
            self.assertEqual([w['returncode'] for w in status], [0, 0, 0, 0, 1, 0, 0, 0])
            self.assertEqual(json.loads((folder / 'parallel_manifest.json').read_text())['status'], 'stopped_or_error')

    def test_eight_workers_wait_independently_and_one_takeover_does_not_reset_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            folders = [Path(tmp) / f'w{i}' for i in range(8)]
            processes = []
            def wait_request(folder, after=None):
                until = time.monotonic() + 10
                while time.monotonic() < until:
                    file = folder / 'controller_status.json'
                    if file.exists():
                        status = json.loads(file.read_text())
                        if status['status'] == 'waiting_for_agent':
                            request = json.loads(Path(status['request']).read_text())
                            if request['request_id'] != after:
                                return request, Path(status['response'])
                    time.sleep(0.01)
                self.fail('worker request timeout')
            def respond(request, path, **kwargs):
                response = dict(request_id=request['request_id'], subgoal='grasp', demo_segment='grasp',
                                reason='independent fixture review', scene_review='fixture checked', **kwargs)
                bridge.decision_for(request, response)
                bridge.atomic_json(path, response)
            try:
                for folder in folders:
                    bridge.atomic_json(folder / 'demos/plug_in_charger/demo_context.json',
                                       {'status': 'reviewed', 'segments': {'grasp': {}}})
                    process = multiprocessing.get_context('fork').Process(target=fake_worker, args=(str(folder),))
                    process.start()
                    processes.append(process)
                pending = [wait_request(folder) for folder in folders]
                self.assertEqual(len({request['request_id'] for request, _ in pending}), 8)
                for request, _ in pending:
                    self.assertEqual(request['step'], 0)
                request, path = pending[0]
                respond(request, path, mode='takeover', actions=[action(0.01)])
                next_request, path = wait_request(folders[0], request['request_id'])
                self.assertEqual(next_request['step'], 1)
                for folder, (other, _) in zip(folders[1:], pending[1:]):
                    self.assertEqual(wait_request(folder)[0]['request_id'], other['request_id'])
                    self.assertFalse(list(folder.glob('plug_in_charger/seed_*/*/interventions.jsonl')))
                respond(next_request, path, mode='hand_back', recovery_evidence='fixture stable')
                request, path = wait_request(folders[0], next_request['request_id'])
                respond(request, path, mode='approve', execute=4)
                for request, path in pending[1:]:
                    respond(request, path, mode='approve', execute=4)
                for process in processes:
                    process.join(10)
                    self.assertEqual(process.exitcode, 0)
                counts = [json.loads((folder / 'model_calls.json').read_text()).count('reset') for folder in folders]
                self.assertEqual(counts, [2, 1, 1, 1, 1, 1, 1, 1])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(2)
