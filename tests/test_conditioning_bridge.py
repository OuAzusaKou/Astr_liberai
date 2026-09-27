"""Closed-loop conditioning transitions with changing live camera observations."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from astra import bridge
from astra.conditioning import pixel_hash
from test_bridge import Env, action
from scripts.monitor_parallel import classify


class ChangingEnv(Env):
    step_lim = 4

    def get_obs(self):
        obs = super().get_obs()
        obs['vision']['cam_head']['color'][:] = self.take_action_cnt[0] * 20
        return obs


class ConditioningBridgeTests(unittest.TestCase):
    def run_loop(self, commands, env=None, bad_receipt=False):
        env = env or ChangingEnv()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        session = Path(temp.name)
        write = bridge.atomic_json
        write(session / 'run_manifest.json', dict(conditioning_mode='l3_l2',
              unlimited_recovery=True, allow_agent_termination=False))
        write(session / 'demos/plug_in_charger/demo_context.json',
              dict(status='reviewed', segments={'grasp': {}, 'insert': {}}))
        requests, received, calls, receipts = [], [], [], []

        def call(func_name, **kwargs):
            calls.append(func_name)
            if func_name == 'update_obs':
                received.append(deepcopy(kwargs['obs']))
            if func_name == 'get_action':
                obs = received[-1]
                m = obs['_astra_conditioning']
                receipt = dict(verified=True, status='inference_completed',
                    inference_id=m['inference_id'], request_id=m['request_id'],
                    conditioning_version=m['conditioning_version'], actual_conditioning_mode='l3_l2',
                    actual_instruction=f"{m['l3_instruction']}\nCurrent subtask instruction: {m['l2_instruction']}",
                    actual_cue_pixel_sha256=pixel_hash(obs['l2_visual_cue_rgb']),
                    actual_bboxes_xyxy=obs['l2_bboxes_xyxy'].tolist())
                if bad_receipt and not receipts:
                    receipt['conditioning_version'] = 'wrong-version'
                receipts.append(receipt)
                write(session / 'model_conditioning.jsonl.receipt.json', receipt)
                return [action(0.02)] * 2

        def publish(path, value):
            write(path, value)
            if Path(path).name != 'request.json':
                return
            requests.append(deepcopy(value))
            if len(requests) == 1:
                self.assertEqual(value['state'], 'PREPARE_CONDITIONING')
                self.assertEqual(calls, ['reset'])
                self.assertEqual(value['proposed_actions'], [])
            command = dict(commands.pop(0))
            segment = command.pop('segment', 'grasp')
            if command.pop('fresh', False):
                command['conditioning'] = dict(subgoal=segment, l2_instruction='Operate ' + segment,
                    cue_rgb_path=value['observation']['images']['cam_head'],
                    l2_bboxes_xyxy=[[1, 1, 6, 7]], box_source='astra_visual',
                    evidence='Reviewed current head image', change_reason='Observed next stage')
            if command.pop('bad_box', False):
                command['conditioning']['l2_bboxes_xyxy'] = [[0, 0, 90, 80]]
            if command.pop('reuse', False):
                command['conditioning_version'] = value['conditioning']['conditioning_version']
            response = dict(request_id=value['request_id'], subgoal=segment, demo_segment=segment,
                            reason='Observed scene evidence', scene_review='Reviewed current scene', **command)
            write(Path(path).with_name('response.json'), response)

        def pause(*args):
            self.assertFalse(env.executed)
            self.assertFalse(list(session.glob('plug_in_charger/seed_*/*/summary.json')))
            self.assertIn('receipt', str(args[-1]))
        with patch.dict(os.environ, {'ASTRA_SESSION': str(session), 'ASTRA_CONDITIONING_AUDIT_LOG': str(session / 'model_conditioning.jsonl')}), patch.object(bridge, 'atomic_json', publish), patch.object(bridge, 'pause_for_main', side_effect=pause) as paused:
            bridge.eval_one_episode(env, SimpleNamespace(call=call))
            self.assertEqual(paused.call_count, int(bad_receipt))
        summary_path = next(session.glob('plug_in_charger/seed_*/*/summary.json'))
        return env, requests, received, json.loads(summary_path.read_text()), summary_path.parent

    def test_reject_invalid_condition_then_keep_cue_across_chunks_and_refresh_next_subgoal(self):
        commands = [dict(mode='condition', fresh=True, bad_box=True),
                    dict(mode='condition', fresh=True),
                    dict(mode='approve', execute=2),
                    dict(mode='replan', segment='insert', fresh=True),
                    dict(mode='approve', segment='insert', execute=2)]
        env, requests, received, summary, folder = self.run_loop(commands)
        self.assertEqual(requests[0]['step'], requests[1]['step'])
        self.assertIn('bounds', requests[1]['rejected_action'])
        self.assertEqual(len(received), 3)
        self.assertTrue((received[0]['l2_visual_cue_rgb'] == 0).all())
        self.assertTrue((received[1]['l2_visual_cue_rgb'] == 0).all())
        self.assertTrue((received[1]['vision']['cam_head']['color'] == 40).all())
        self.assertTrue((received[2]['l2_visual_cue_rgb'] == 40).all())
        self.assertEqual(received[2]['l2_instruction'], 'Operate insert')
        self.assertEqual(summary['conditioning_versions'], 2)
        self.assertEqual(classify(summary), 'assisted_success')
        self.assertEqual(len(list((folder / 'conditioning').glob('*.png'))), 2)
        self.assertEqual(len((folder / 'inference_conditions.jsonl').read_text().splitlines()), 3)
        self.assertEqual(len(env.executed), 4)

    def test_takeover_does_not_infer_and_handback_requires_explicit_version(self):
        env = ChangingEnv(); env.step_lim = 3
        commands = [dict(mode='condition', fresh=True),
                    dict(mode='takeover', actions=[action(0.01)]),
                    dict(mode='takeover', actions=[action(0.01)], new_attempt=False),
                    dict(mode='hand_back', recovery_evidence='Stable held object'),
                    dict(mode='hand_back', recovery_evidence='Stable held object', reuse=True),
                    dict(mode='approve', execute=1)]
        env, requests, received, summary, _ = self.run_loop(commands, env)
        self.assertEqual(len(received), 2)
        self.assertIn('exactly one', requests[-2]['rejected_action'])
        self.assertEqual(received[0]['_astra_conditioning']['conditioning_version'],
                         received[1]['_astra_conditioning']['conditioning_version'])
        self.assertEqual(summary['takeovers'], 1)
        self.assertEqual(summary['retries'], {'grasp': 1})
        self.assertEqual(len(env.executed), 3)

    def test_model_receipt_mismatch_pauses_scene_then_retries_after_main_repair(self):
        env, requests, received, summary, _ = self.run_loop(
            [dict(mode='condition', fresh=True), dict(mode='approve', execute=2),
             dict(mode='approve', execute=2)], bad_receipt=True)
        self.assertEqual(len(env.executed), 4)
        self.assertEqual(summary['status'], 'success')
        self.assertEqual(len(requests), 3)
        self.assertNotEqual(received[0]['_astra_conditioning']['inference_id'],
                            received[1]['_astra_conditioning']['inference_id'])

    def test_l2_assistance_counts_even_without_action_intervention(self):
        env = ChangingEnv(); env.step_lim = 1
        _, _, _, summary, _ = self.run_loop([
            dict(mode='condition', fresh=True), dict(mode='approve', execute=1)], env)
        self.assertEqual(summary['takeovers'], 0)
        self.assertEqual(summary['corrections'], 0)
        self.assertEqual(classify(summary), 'assisted_success')


if __name__ == '__main__':
    unittest.main()
