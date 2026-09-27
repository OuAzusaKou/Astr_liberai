import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra.supervision import completion_transition, publish_decision, with_runtime_limits
from astra.bridge import decision_for


class SupervisionTests(unittest.TestCase):
    def test_old_live_budget_keeps_its_actual_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'run_manifest.json').write_text(json.dumps({'max_retries_per_subgoal':3,'max_takeovers':10}))
            request=dict(request_id='r',demo_segments=['grasp'],retries={'grasp':3},takeover_count=3)
            command=dict(request_id='r',mode='abort',subgoal='grasp',demo_segment='grasp',reason='documented inability to continue',
                         termination_condition='recovery_budget_exhausted',termination_evidence='three distinct attempts; cannot safely hand back')
            with self.assertRaises(ValueError):decision_for(request,command)
            decision_for(with_runtime_limits(directory,request),command)
            request['max_retries_per_subgoal']=6
            with self.assertRaises(ValueError):decision_for(with_runtime_limits(directory,request),command)

    def request(self, step):
        return dict(episode='layout_0_a', step=step, request_id=f'a:{step}',
                    observation={'additional_info': {'frequency': 25}})

    def test_completion_deadline_requires_changed_plan_not_repeated_claim(self):
        command = dict(mode='approve', execute=4, completion_candidate=True,
                       completion_evidence='released upright; final operation visually complete')
        watch = completion_transition(self.request(10), command, {})
        with self.assertRaises(ValueError):
            completion_transition(self.request(14), command, watch)
        with self.assertRaises(ValueError):
            completion_transition(self.request(34), dict(mode='approve', execute=2), watch)
        completion_transition(self.request(34), dict(mode='approve', execute=1), watch)
        with self.assertRaises(ValueError):
            completion_transition(self.request(35), dict(mode='approve', execute=1), watch)
        adjusted = completion_transition(self.request(35), dict(mode='replan', reason='check alignment and reseat'), watch)
        with self.assertRaises(ValueError):
            completion_transition(self.request(35), command, adjusted)
        renewed = completion_transition(self.request(39), command, adjusted)
        self.assertEqual(renewed['candidate_step'], 39)

    def test_wall_clock_wait_and_new_episode_do_not_cause_false_deadline(self):
        command = dict(mode='approve', execute=4, completion_candidate=True, completion_evidence='visual evidence')
        with patch('astra.supervision.time.time', return_value=1):
            watch = completion_transition(self.request(10), command, {})
        with patch('astra.supervision.time.time', return_value=5000):
            result = completion_transition(self.request(10), dict(mode='approve', execute=4), watch)
        self.assertEqual(result['elapsed_sim_s'], 0)
        request=self.request(0);request['episode']='layout_1_b'
        self.assertEqual(completion_transition(request, dict(mode='approve', execute=4), watch), {})

    def test_owner_directive_ack_and_response_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);target=folder/'response.json'
            (folder/'supervisor_owner.json').write_text(json.dumps({'owner_id':'a'}))
            (folder/'supervisor_directive.json').write_text(json.dumps({'version':'v2'}))
            with patch.dict(os.environ, {'ASTRA_SUPERVISOR_ID':'a'}):
                with self.assertRaises(PermissionError):publish_decision(folder,target,{'request_id':'r'})
                (folder/'supervisor_directive_ack.json').write_text(json.dumps({'owner_id':'a','version':'v2'}))
                publish_decision(folder,target,{'request_id':'r'})
                with self.assertRaises(FileExistsError):publish_decision(folder,target,{'request_id':'other'})
            self.assertEqual(json.loads(target.read_text())['request_id'],'r')


if __name__ == '__main__':
    unittest.main()
