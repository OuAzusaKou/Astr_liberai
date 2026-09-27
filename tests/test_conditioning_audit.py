import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from astra.conditioning_audit import ConditioningAuditMixin, pixel_sha256
from astra.config import PROJECT, SOURCE, RUNTIME_DEPS


def observation():
    cue = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    return dict(instruction='Put the object away.', task_instruction='Put the object away.',
                l2_instruction='Grasp the object.', l2_visual_cue_rgb=cue,
                l2_bboxes_xyxy=np.array([[1, 2, 5, 7]], dtype=np.float32),
                _astra_conditioning=dict(inference_id='inference-1', request_id='request-1',
                    conditioning_version=1, l3_instruction='Put the object away.',
                    l2_instruction='Grasp the object.', cue_pixel_sha256=pixel_sha256(cue),
                    cue_rgb_path='/recorded/current_scene.png',
                    l2_bboxes_xyxy=[[1, 2, 5, 7]], subgoal='grasp', start_step=0))


class Parent:
    conditioning_mode = 'l3_l2'

    def update_obs(self, obs):
        self.last_obs = dict(visual_cue_rgb=obs['l2_visual_cue_rgb'].copy(),
                            visual_cue_bboxes_xyxy=obs['l2_bboxes_xyxy'].copy())
        self.last_instruction = obs['instruction'] + '\nCurrent subtask instruction: ' + obs['l2_instruction']

    def get_action(self):
        self.inference_count = getattr(self, 'inference_count', 0) + 1
        return [{'action': [0.1]}]

    def reset(self):
        self.last_obs = None


class AuditedModel(ConditioningAuditMixin, Parent):
    pass


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / 'model_audit.jsonl'
        env = patch.dict(os.environ, ASTRA_CONDITIONING_AUDIT_LOG=str(self.log))
        env.start()
        self.addCleanup(env.stop)

    def test_receipt_tracks_observation_inference_and_reset(self):
        model = AuditedModel()
        model.update_obs(observation())
        receipt = model.get_conditioning_receipt()
        self.assertEqual(receipt['status'], 'observation_verified')
        self.assertEqual(receipt['actual_bboxes_xyxy'], [[1, 2, 5, 7]])
        receipt['actual_bboxes_xyxy'][0][0] = 99
        self.assertEqual(model.get_conditioning_receipt()['actual_bboxes_xyxy'][0][0], 1)
        self.assertEqual(model.get_action(), [{'action': [0.1]}])
        self.assertEqual(model.get_conditioning_receipt()['status'], 'inference_completed')
        saved = json.loads(Path(str(self.log) + '.receipt.json').read_text())
        self.assertEqual(saved, model.get_conditioning_receipt())
        rows = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual([r['status'] for r in rows], ['observation_verified', 'inference_started', 'inference_completed'])
        self.assertTrue(all(r['inference_id'] == 'inference-1' for r in rows))
        model.reset()
        self.assertIsNone(model.get_conditioning_receipt())
        with self.assertRaises(ValueError):
            model.get_action()

    def test_invalid_update_clears_old_receipt_and_refuses_stale_inference(self):
        for field in ('_astra_conditioning', 'l2_visual_cue_rgb', 'l2_bboxes_xyxy', 'l2_instruction'):
            with self.subTest(field=field):
                model = AuditedModel()
                model.update_obs(observation())
                invalid = observation()
                invalid.pop(field)
                with self.assertRaises(ValueError):
                    model.update_obs(invalid)
                self.assertIsNone(model.get_conditioning_receipt())
                with self.assertRaises(ValueError):
                    model.get_action()
                self.assertFalse(hasattr(model, 'inference_count'))

    def test_cue_boxes_and_instruction_are_checked_before_inference(self):
        for field, value in [('cue_pixel_sha256', 'wrong'), ('l2_bboxes_xyxy', [[2, 2, 5, 7]]),
                             ('l3_instruction', 'Wrong task')]:
            with self.subTest(field=field):
                obs = observation()
                obs['_astra_conditioning'][field] = value
                model = AuditedModel()
                with self.assertRaises(ValueError):
                    model.update_obs(obs)
                with self.assertRaises(ValueError):
                    model.get_action()
        model = AuditedModel()
        model.update_obs(observation())
        model.last_obs['visual_cue_rgb'][0, 0, 0] = 42
        with self.assertRaises(ValueError):
            model.get_action()
        self.assertFalse(hasattr(model, 'inference_count'))

    def test_stale_task_instruction_override_is_rejected(self):
        obs = observation()
        obs['task_instruction'] = 'Wrong prior instruction'
        with self.assertRaisesRegex(ValueError, 'override'):
            AuditedModel().update_obs(obs)

    def test_no_audit_sink_refuses_inference(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'AUDIT_LOG'):
                AuditedModel().update_obs(observation())

    def test_pure_l3_does_not_silently_accept_l2_metadata(self):
        model = AuditedModel()
        model.conditioning_mode = 'pure_l3'
        with self.assertRaisesRegex(ValueError, 'not in l3_l2'):
            model.update_obs(observation())
        # Legacy parent behavior is preserved for observations without metadata.
        obs = observation()
        obs.pop('_astra_conditioning')
        model.update_obs(obs)
        self.assertEqual(model.get_action(), [{'action': [0.1]}])
        self.assertIsNone(model.get_conditioning_receipt())

    def test_failure_receipt_does_not_claim_inference_completed(self):
        model = AuditedModel()
        model.update_obs(observation())
        with patch.object(Parent, 'get_action', side_effect=RuntimeError('inference unavailable')):
            with self.assertRaises(RuntimeError):
                model.get_action()
        self.assertEqual(model.get_conditioning_receipt()['status'], 'inference_failed')
        saved = json.loads(Path(str(self.log) + '.receipt.json').read_text())
        self.assertEqual(saved['status'], 'inference_failed')

    def test_receipt_publication_failure_withholds_actions_and_preserves_previous_file(self):
        model = AuditedModel()
        model.update_obs(observation())
        receipt_path = Path(str(self.log) + '.receipt.json')
        old_receipt = receipt_path.read_text()
        with patch('astra.conditioning_audit.os.replace', side_effect=OSError('publication failed')):
            with self.assertRaisesRegex(OSError, 'publication failed'):
                model.get_action()
        self.assertEqual(receipt_path.read_text(), old_receipt)
        self.assertFalse(hasattr(model, 'inference_count'))
        self.assertEqual(list(Path(self.tmp.name).glob('*.tmp')), [])

    def test_real_adapter_preserves_conditions_and_forwards_to_policy(self):
        if not (PROJECT / 'policy_server/imagewam/xpolicylab_policy/model.py').is_file():
            self.skipTest('Shared ImageWAM adapter is unavailable')
        with patch.object(sys, 'path', [str(PROJECT), str(SOURCE)] + sys.path):
            adapter = importlib.import_module('policy_server.imagewam.xpolicylab_policy.model')
        class Policy:
            state_dim = action_dim = 14
            action_horizon = 32
            robotwin_requires_visual_cue = True

            def infer_action_chunk(self, obs, instruction, request_seed):
                self.received = copy.deepcopy(obs), instruction
                return np.zeros((32, 14), dtype=np.float32)

        class RealAuditedAdapter(ConditioningAuditMixin, adapter.ImageWAMAdapter):
            pass

        policy = Policy()
        model = RealAuditedAdapter(policy, action_type='joint',
            robot_action_dim_info={'arm_dim': [6, 6], 'ee_dim': [1, 1]}, conditioning_mode='l3_l2')
        obs = observation()
        obs['vision'] = {camera: {'color': np.zeros((8, 10, 3), dtype=np.uint8)}
                         for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')}
        obs['state'] = {f'{side}_arm_joint_state': np.zeros(6) for side in ('left', 'right')}
        obs['state'].update({f'{side}_ee_joint_state': np.zeros(1) for side in ('left', 'right')})
        model.update_obs(obs)
        actions = model.get_action()
        self.assertEqual(len(actions), 16)
        self.assertIn('left_arm_joint_state', actions[0])
        actual_obs, instruction = policy.received
        np.testing.assert_array_equal(actual_obs['visual_cue_rgb'], obs['l2_visual_cue_rgb'])
        np.testing.assert_array_equal(actual_obs['visual_cue_bboxes_xyxy'], obs['l2_bboxes_xyxy'])
        self.assertEqual(instruction, 'Put the object away.\nCurrent subtask instruction: Grasp the object.')
        self.assertEqual(model.get_conditioning_receipt()['status'], 'inference_completed')

        # Exercise the real WS client -> msgpack -> server handler -> adapter
        # path without opening sockets. update_obs is local-only; the single
        # get_action/infer request must carry the complete observation.
        with patch.object(sys, 'path', [str(SOURCE / 'XPolicyLab'), str(RUNTIME_DEPS)] + sys.path):
            client_module = importlib.import_module('client_server.ws.model_client')
            server_module = importlib.import_module('client_server.ws.model_server')
            codec = importlib.import_module('client_server.ws.protocol.codec')
            schemas = importlib.import_module('client_server.ws.protocol.schemas')
        server = server_module.PolicyServer(model)

        class InProcessTransport:
            async def connect(self, handshake=True):
                pass

            async def close(self):
                pass

            async def infer(self, observation, **kwargs):
                frame = schemas.Frame(message_type='infer', request_id='wire-request-1',
                    evaluation_id='test', payload={'observation': observation}, **kwargs)
                transmitted = codec.decode_envelope(codec.encode_frame(frame))
                response = await server.process_frame(transmitted)
                return codec.decode_envelope(codec.encode_frame(response))

        client = client_module.WsModelClient(url='inprocess://test', evaluation_id='test',
            trial_id='layout-0', client=InProcessTransport())
        try:
            obs['_astra_conditioning']['inference_id'] = 'inference-over-ws'
            client.call(func_name='update_obs', obs=obs)
            # Real update_obs does not yet contact the server.
            self.assertNotEqual(model.get_conditioning_receipt()['inference_id'], 'inference-over-ws')
            received_actions = client.call(func_name='get_action')
            self.assertEqual(len(received_actions), 16)
            saved = json.loads(Path(str(self.log) + '.receipt.json').read_text())
            self.assertEqual(saved['inference_id'], 'inference-over-ws')
            self.assertEqual(saved['status'], 'inference_completed')
            self.assertEqual(saved['actual_cue_pixel_sha256'], obs['_astra_conditioning']['cue_pixel_sha256'])
            self.assertEqual(saved['actual_bboxes_xyxy'], [[1, 2, 5, 7]])
        finally:
            client.close()


if __name__ == '__main__':
    unittest.main()
