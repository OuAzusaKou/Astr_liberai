import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from astra import bridge


def action(value=0.01):
    return {f'{side}_{name}': [value] * size for side in ('left', 'right')
            for name, size in (('arm_joint_state', 6), ('ee_joint_state', 1))}


class Tensor:
    def __getitem__(self, key):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array([[-2, 2]] * 6)


class Env:
    num_envs = 1
    task_name = 'plug_in_charger'
    env_seeds = [0]
    eval_seed = 0
    step_lim = 2

    def __init__(self):
        self.take_action_cnt = [0]
        self.success = [False]
        self.executed = []
        robots = [SimpleNamespace(type='target', arm_name=f'{s}_arm',
                                 gripper_name=f'{s}_ee', arm_joint_indices=list(range(6)))
                  for s in ('left', 'right')]
        self.robot_manager = SimpleNamespace(
            robot_list=robots, robot_key=[SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=Tensor()))] * 2,
            process_name=lambda n: n + '_joint_state', get_joint=lambda *a, **k: {0: np.zeros(6)})

    def get_obs(self):
        return {'vision': {'cam_head': {'color': np.zeros((8, 8, 3), dtype=np.uint8)}}, 'state': {}, 'instruction': 'Plug in charger'}

    def validate_action_dict(self, value):
        for k, v in value.items():
            if np.asarray(v).shape != ((1,) if '_ee_joint' in k else (6,)):
                raise ValueError('shape')

    def take_action(self, value):
        assert not self.is_episode_end()
        self.executed.append(value)
        self.take_action_cnt[0] += 1

    def is_episode_end(self):
        self.success[0] = self.take_action_cnt[0] >= self.step_lim
        return self.success[0]


class BridgeTests(unittest.TestCase):
    def test_monitoring_batch_limits_and_short_batch_evidence(self):
        request = dict(request_id='r', demo_segments=['grasp'], proposed_actions=[action()] * 16,
                       max_actions_per_decision=15, high_level_monitor_interval_actions=15)
        base = dict(request_id='r', subgoal='grasp', demo_segment='grasp', reason='observed contact risk')
        for mode in ('approve', 'correct', 'takeover'):
            command = dict(base, mode=mode, execute=15, actions=[action()] * 15)
            self.assertEqual(bridge.decision_for(request, command), command)
            self.assertEqual(bridge.batch_audit(request, command)['selected_actions'], 15)
            command.update(execute=16, actions=[action()] * 16)
            with self.assertRaises(ValueError):
                bridge.decision_for(request, command)
        command = dict(base, mode='approve', execute=2)
        self.assertEqual(bridge.batch_audit(request, command)['short_batch_reason'], 'observed contact risk')
        request['proposed_actions'] = [action()] * 2
        self.assertIn('remaining reviewed chunk', bridge.batch_audit(request, command)['short_batch_reason'])
        legacy = dict(request_id='r', demo_segments=['grasp'], proposed_actions=[action()] * 15)
        with self.assertRaises(ValueError):
            bridge.decision_for(legacy, dict(base, mode='approve', execute=15))
        for config in ({'max_actions_per_decision': 0}, {'max_actions_per_decision': True},
                       {'max_actions_per_decision': 15, 'high_level_monitor_interval_actions': 4}):
            with self.assertRaises(ValueError):
                bridge.monitoring_config(config)

    def test_fifteen_action_batch_stops_at_official_terminal(self):
        for mode in ('approve', 'correct', 'takeover'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                session, env = Path(tmp), Env()
                original_write = bridge.atomic_json
                original_write(session / 'run_manifest.json', dict(
                    max_actions_per_decision=15, high_level_monitor_interval_actions=15))
                original_write(session / 'demos/plug_in_charger/demo_context.json',
                               {'status': 'reviewed', 'segments': {'grasp': {}}})
                def write(path, value):
                    original_write(path, value)
                    if Path(path).name == 'request.json':
                        self.assertEqual(value['high_level_monitor_interval_actions'], 15)
                        original_write(Path(path).with_name('response.json'), dict(
                            request_id=value['request_id'], mode=mode, execute=15,
                            actions=[action()] * 15, subgoal='grasp', demo_segment='grasp',
                            scene_review='reviewed fixture', reason='reviewed safe motion'))
                def call(func_name, **kwargs):
                    if func_name == 'get_action':
                        return [action()] * 16
                with patch.dict(os.environ, {'ASTRA_SESSION': str(session)}), patch.object(bridge, 'atomic_json', write):
                    bridge.eval_one_episode(env, SimpleNamespace(call=call))
                self.assertEqual(len(env.executed), 2)
                summary = json.loads(next(session.glob('plug_in_charger/seed_*/*/summary.json')).read_text())
                self.assertTrue(summary['official_success'])
                self.assertEqual(summary['high_level_monitor_interval_actions'], 15)

    def test_unlimited_recovery_and_rejected_termination_keep_official_step_budget(self):
        class BudgetEnv(Env):
            step_lim = 4

            def is_episode_end(self):
                self.success[0] = False
                return self.take_action_cnt[0] >= self.step_lim

        env = BudgetEnv()
        commands = [dict(mode='abort')]
        for index in range(12):
            commands.append(dict(mode='takeover', new_attempt=True, actions=[action()]))
            if index == 0:
                commands.append(dict(mode='blocked'))
            if index != 11:
                commands.append(dict(mode='hand_back', recovery_evidence='verified safe pose'))
        requests = []
        original_write = bridge.atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            original_write(session / 'run_manifest.json', dict(unlimited_recovery=True,
                           allow_agent_termination=False, max_retries_per_subgoal=None,
                           max_takeovers=None, step_limit_multiplier=3))
            original_write(session / 'demos/plug_in_charger/demo_context.json',
                           {'status': 'reviewed', 'segments': {'grasp': {}}})
            def write(path, value):
                original_write(path, value)
                if Path(path).name == 'request.json':
                    requests.append(json.loads(json.dumps(value)))
                    command = commands.pop(0)
                    if command['mode'] in {'abort', 'blocked'}:
                        # Even malformed agent stop proposals must not terminate the scene.
                        response = dict(request_id=value['request_id'], **command)
                    else:
                        response = dict(request_id=value['request_id'], subgoal='grasp',
                                        demo_segment='grasp', reason='distinct observed retry',
                                        scene_review='reviewed fixture', **command)
                    original_write(Path(path).with_name('response.json'), response)
            def call(func_name, **kwargs):
                if func_name == 'get_action':
                    return [action()] * 4
            with patch.dict(os.environ, {'ASTRA_SESSION': str(session)}), patch.object(bridge, 'atomic_json', write):
                bridge.eval_one_episode(env, SimpleNamespace(call=call))
            self.assertFalse(commands)
            self.assertEqual(len(env.executed), 12)
            self.assertEqual(requests[1]['step'], 0)
            self.assertEqual(requests[1]['state'], 'LOW_POLICY')
            self.assertEqual(requests[3]['step'], 1)
            self.assertEqual(requests[3]['state'], 'TAKEOVER')
            for request in requests:
                self.assertIsNone(request['max_retries_per_subgoal'])
                self.assertIsNone(request['max_takeovers'])
                self.assertTrue(request['unlimited_recovery'])
                self.assertFalse(request['allow_agent_termination'])
                self.assertEqual(request['step_limit'], 12)
            path = next(session.glob('plug_in_charger/seed_*/*/summary.json'))
            summary = json.loads(path.read_text())
            self.assertEqual(summary['takeovers'], 12)
            self.assertEqual(summary['retries']['grasp'], 12)
            self.assertEqual(summary['status'], 'failed')
            self.assertTrue(summary['official_terminal'])
            self.assertFalse(summary['official_success'])
            self.assertEqual(summary['steps'], 12)
            self.assertIsNone(summary['failure'])
            self.assertIsNone(summary['max_takeovers'])
            events = [json.loads(line) for line in path.with_name('interventions.jsonl').read_text().splitlines()]
            rejected = [event for event in events if event['event'] == 'decision_rejected']
            self.assertEqual([event['submitted_command']['mode'] for event in rejected], ['abort', 'blocked'])
            self.assertFalse(path.with_name('error.json').exists())

    def test_real_service_error_remains_infrastructure_error_under_new_policy(self):
        env = Env()
        def call(func_name, **kwargs):
            if func_name == 'get_action':
                raise RuntimeError('Model connection lost')
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            bridge.atomic_json(session / 'run_manifest.json', dict(unlimited_recovery=True,
                               allow_agent_termination=False, step_limit_multiplier=3))
            bridge.atomic_json(session / 'demos/plug_in_charger/demo_context.json',
                               {'status': 'reviewed', 'segments': {'grasp': {}}})
            with patch.dict(os.environ, {'ASTRA_SESSION': str(session)}):
                with self.assertRaisesRegex(RuntimeError, 'Model connection lost'):
                    bridge.eval_one_episode(env, SimpleNamespace(call=call))
            summary = json.loads(next(session.glob('plug_in_charger/seed_*/*/summary.json')).read_text())
            self.assertEqual(summary['status'], 'infrastructure_error')
            self.assertFalse(summary['allow_agent_termination'])
            self.assertTrue(summary['unlimited_recovery'])
            self.assertEqual(summary['steps'], 0)
            self.assertFalse(env.executed)

    def test_unlimited_budget_cannot_be_exhausted_and_legacy_limits_stay_bounded(self):
        limits = bridge.recovery_limits({'unlimited_recovery': True})
        self.assertIsNone(limits['max_retries_per_subgoal'])
        self.assertIsNone(limits['max_takeovers'])
        self.assertEqual(bridge.recovery_limits({})['max_retries_per_subgoal'], 6)
        self.assertEqual(bridge.recovery_limits({})['max_takeovers'], 10)
        request = dict(request_id='r', demo_segments=['grasp'], proposed_actions=[],
                       unlimited_recovery=True, takeover_count=1000, retries={'grasp': 1000})
        response = dict(request_id='r', mode='abort', reason='retries', subgoal='grasp', demo_segment='grasp',
                        termination_condition='recovery_budget_exhausted', termination_evidence='many retries')
        with self.assertRaisesRegex(ValueError, 'has not been exhausted'):
            bridge.decision_for(request, response)
        for flag in ('unlimited_recovery', 'allow_agent_termination'):
            with self.assertRaisesRegex(ValueError, 'must be booleans'):
                bridge.recovery_limits({flag: 'true'})

    def test_step_limit_applies_before_terminal_checks_without_compounding(self):
        class BoundaryEnv(Env):
            step_lim = 400

            def __init__(self):
                super().__init__()
                self.checked_limits = []

            def is_episode_end(self):
                self.checked_limits.append(self.step_lim)
                # Task success is independent of its action-budget termination.
                self.success[0] = self.take_action_cnt[0] >= 401
                return self.success[0] or self.take_action_cnt[0] >= self.step_lim

        for config, expected_limit, expected_success in [({'step_limit_multiplier': 3}, 1200, True), ({}, 400, False)]:
            with self.subTest(config=config), tempfile.TemporaryDirectory() as tmp:
                session = Path(tmp)
                env = BoundaryEnv()
                original_write = bridge.atomic_json
                original_write(session / 'run_manifest.json', config)
                original_write(session / 'demos/plug_in_charger/demo_context.json',
                               {'status': 'reviewed', 'segments': {'grasp': {}}})
                requests = []
                def write(path, value):
                    original_write(path, value)
                    if Path(path).name == 'request.json':
                        requests.append(value)
                        original_write(Path(path).with_name('response.json'), dict(
                            request_id=value['request_id'], mode='approve', execute=4,
                            reason='Continue toward independent task success',
                            subgoal='grasp', demo_segment='grasp'))
                def call(func_name, **kwargs):
                    if func_name == 'get_action':
                        return [action()] * 4
                with patch.dict(os.environ, {'ASTRA_SESSION': str(session)}), patch.object(bridge, 'atomic_json', write):
                    for layout in (0, 1):
                        env.env_seeds = [layout]
                        env.take_action_cnt = [399]
                        env.success = [False]
                        bridge.eval_one_episode(env, SimpleNamespace(call=call))
                        self.assertEqual(env.step_lim, expected_limit)
                self.assertEqual(set(env.checked_limits), {expected_limit})
                self.assertEqual(len(requests), 2)
                for request in requests:
                    self.assertEqual(request['original_step_limit'], 400)
                    self.assertEqual(request['step_limit_multiplier'], config.get('step_limit_multiplier', 1))
                    self.assertEqual(request['step_limit'], expected_limit)
                summaries = list(session.glob('plug_in_charger/seed_*/*/summary.json'))
                self.assertEqual(len(summaries), 2)
                for path in summaries:
                    summary = json.loads(path.read_text())
                    self.assertEqual(summary['original_step_limit'], 400)
                    self.assertEqual(summary['step_limit'], expected_limit)
                    self.assertEqual(summary['step_limit_multiplier'], config.get('step_limit_multiplier', 1))
                    self.assertEqual(summary['official_success'], expected_success)
                    self.assertEqual(summary['steps'], 401 if expected_success else 400)

    def test_invalid_step_limit_multiplier_rejected_before_mutation(self):
        for value in (0, -1, True, False, 1.5, '3', None):
            with self.subTest(value=value):
                env = Env()
                env.step_lim = 400
                with self.assertRaises(ValueError):
                    bridge.configure_step_limit(env, {'step_limit_multiplier': value})
                self.assertEqual(env.step_lim, 400)
                self.assertFalse(hasattr(env, '_astra_original_step_limit'))

    def test_reject_stale_and_unreviewed_segment(self):
        request = {'request_id': 'new', 'demo_segments': ['grasp'], 'proposed_actions': [action()]}
        response = dict(request_id='old', mode='approve', reason='aligned', subgoal='grasp', demo_segment='grasp')
        with self.assertRaises(ValueError):
            bridge.decision_for(request, response)
        response.update(request_id='new', demo_segment='invented')
        with self.assertRaises(ValueError):
            bridge.decision_for(request, response)

    def test_action_bounds(self):
        env = Env()
        bridge.validated_action(env, action(), direct=True)
        for invalid in (action(3), action(float('nan')), action(0.2)):
            with self.assertRaises(ValueError):
                bridge.validated_action(env, invalid, direct=True)
        invalid = action()
        del invalid['left_ee_joint_state']
        with self.assertRaises(ValueError):
            bridge.validated_action(env, invalid)

    def test_takeover_handback_discards_old_chunk_and_stops_at_terminal(self):
        env = Env()
        calls = []
        def call(func_name, **kwargs):
            calls.append(func_name)
            if func_name == 'get_action':
                return [action(0.02 if calls.count('get_action') == 1 else 0.03)] * 4
        commands = [dict(mode='takeover', actions=[action(0.01)]),
                    dict(mode='hand_back', recovery_evidence='object now follows gripper'),
                    dict(mode='approve', execute=4)]
        original_write = bridge.atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            original_write(session / 'demos/plug_in_charger/demo_context.json', {'status': 'reviewed', 'segments': {'grasp': {}}})
            def write(path, value):
                original_write(path, value)
                if Path(path).name == 'request.json':
                    response = dict(request_id=value['request_id'], subgoal='grasp', demo_segment='grasp', reason='test evidence', scene_review='fixture scene', **commands.pop(0))
                    original_write(Path(path).with_name('response.json'), response)
            with patch.dict(os.environ, {'ASTRA_SESSION': str(session)}), patch.object(bridge, 'atomic_json', write):
                bridge.eval_one_episode(env, SimpleNamespace(call=call))
            self.assertEqual(len(env.executed), 2)
            self.assertAlmostEqual(env.executed[0]['left_arm_joint_state'][0], 0.01)
            self.assertAlmostEqual(env.executed[1]['left_arm_joint_state'][0], 0.03)
            self.assertEqual(calls.count('get_action'), 2)
            summary = json.loads(next(session.glob('plug_in_charger/seed_*/*/summary.json')).read_text())
            self.assertTrue(summary['official_success'])
            self.assertEqual(summary['takeovers'], 1)

    def test_continuous_takeover_counts_distinct_retries_not_action_batches(self):
        env=Env();env.step_lim=4
        commands=[dict(mode='takeover',actions=[action(0.01)]),
                  dict(mode='takeover',new_attempt=False,actions=[action(0.01)]),
                  dict(mode='takeover',new_attempt=True,actions=[action(0.01)]),
                  dict(mode='hand_back',recovery_evidence='safe posture'),
                  dict(mode='approve',execute=1)]
        original_write=bridge.atomic_json
        with tempfile.TemporaryDirectory() as tmp:
            session=Path(tmp)
            original_write(session/'demos/plug_in_charger/demo_context.json',{'status':'reviewed','segments':{'grasp':{}}})
            def write(path,value):
                original_write(path,value)
                if Path(path).name=='request.json':
                    original_write(Path(path).with_name('response.json'),dict(request_id=value['request_id'],subgoal='grasp',demo_segment='grasp',reason='changed approach after failed lift',scene_review='fixture',**commands.pop(0)))
            def call(func_name,**kwargs):
                if func_name=='get_action':return [action(0.02)]*4
            with patch.dict(os.environ,{'ASTRA_SESSION':str(session)}),patch.object(bridge,'atomic_json',write):
                bridge.eval_one_episode(env,SimpleNamespace(call=call))
            summary=json.loads(next(session.glob('plug_in_charger/seed_*/*/summary.json')).read_text())
            self.assertEqual(summary['takeovers'],1)
            self.assertEqual(summary['retries'],{'grasp':2})
            self.assertEqual(len(env.executed),4)

    def test_timeout_never_executes_policy(self):
        env = Env()
        def call(func_name, **kwargs):
            if func_name == 'get_action':
                return [action()]
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            bridge.atomic_json(session / 'demos/plug_in_charger/demo_context.json', {'status': 'reviewed', 'segments': {'grasp': {}}})
            with patch.dict(os.environ, {'ASTRA_SESSION': str(session), 'ASTRA_CONTROLLER_TIMEOUT': '0'}):
                with self.assertRaises(TimeoutError):
                    bridge.eval_one_episode(env, SimpleNamespace(call=call))
            self.assertFalse(env.executed)
            summary = json.loads(next(session.glob('plug_in_charger/seed_*/*/summary.json')).read_text())
            self.assertEqual(summary['status'], 'infrastructure_error')
            self.assertIsNone(summary['official_success'])

    def test_finite_gripper_overshoot_preserved_for_native_executor(self):
        value = action()
        value['left_ee_joint_state'] = [1.0095]
        value['right_ee_joint_state'] = [-0.001]
        result = bridge.validated_action(Env(), value)
        self.assertEqual(result['left_ee_joint_state'][0], 1.0095)
        self.assertEqual(result['right_ee_joint_state'][0], -0.001)

    def test_stagnation_is_not_termination_and_unspent_budget_cannot_abort(self):
        request = dict(request_id='r', demo_segments=['grasp'], proposed_actions=[],
                       takeover_count=2, retries={'grasp': 2})
        response = dict(request_id='r', mode='abort', reason='no progress',
                        subgoal='grasp', demo_segment='grasp')
        with self.assertRaises(ValueError):
            bridge.decision_for(request, response)
        response.update(termination_condition='recovery_budget_exhausted', termination_evidence='two attempts')
        with self.assertRaises(ValueError):
            bridge.decision_for(request, response)

    def test_abort_uses_the_recorded_recovery_limits(self):
        request = dict(request_id='r', demo_segments=['grasp'], proposed_actions=[],
                       takeover_count=1, retries={'grasp': 3},
                       max_retries_per_subgoal=6, max_takeovers=10)
        response = dict(request_id='r', mode='abort', reason='unable to resume safely',
                        subgoal='grasp', demo_segment='grasp',
                        termination_condition='recovery_budget_exhausted',
                        termination_evidence='Recorded attempts and no safe effective continuation')
        with self.assertRaises(ValueError):
            bridge.decision_for(request, response)
        request['retries']['grasp'] = 6
        self.assertEqual(bridge.decision_for(request, response), response)
        request.update(max_retries_per_subgoal=3, retries={'grasp': 3})
        self.assertEqual(bridge.decision_for(request, response), response)
        request.update(max_retries_per_subgoal=6, max_takeovers=2, retries={'grasp': 1}, takeover_count=2)
        self.assertEqual(bridge.decision_for(request, response), response)

    def test_excess_recovery_is_rejected_without_ending_episode(self):
        for exhausted_limit in ('retries', 'takeovers'):
            with self.subTest(exhausted_limit=exhausted_limit), tempfile.TemporaryDirectory() as tmp:
                session = Path(tmp)
                env = Env()
                maximum = 6 if exhausted_limit == 'retries' else 2
                env.step_lim = maximum + 1
                commands = []
                for _ in range(maximum):
                    commands.append(dict(mode='takeover', new_attempt=True, actions=[action(0.01)]))
                    if exhausted_limit == 'takeovers':
                        commands.append(dict(mode='hand_back', recovery_evidence='safe verified posture'))
                rejected_index = len(commands)
                commands.append(dict(mode='takeover', new_attempt=True, actions=[action(0.02)]))
                if exhausted_limit == 'retries':
                    commands.append(dict(mode='hand_back', recovery_evidence='safe verified posture'))
                commands.append(dict(mode='approve', execute=1))
                requests = []
                original_write = bridge.atomic_json
                original_write(session / 'demos/plug_in_charger/demo_context.json',
                               {'status': 'reviewed', 'segments': {'grasp': {}}})
                original_write(session / 'run_manifest.json', dict(max_retries_per_subgoal=6,
                               max_takeovers=10 if exhausted_limit == 'retries' else 2))
                def write(path, value):
                    original_write(path, value)
                    if Path(path).name == 'request.json':
                        requests.append(json.loads(json.dumps(value)))
                        response = dict(request_id=value['request_id'], subgoal='grasp',
                                        demo_segment='grasp', reason='changed recovery approach',
                                        scene_review='fixture scene', **commands.pop(0))
                        original_write(Path(path).with_name('response.json'), response)
                def call(func_name, **kwargs):
                    if func_name == 'get_action':
                        return [action(0.03)] * 4
                with patch.dict(os.environ, {'ASTRA_SESSION': str(session)}), patch.object(bridge, 'atomic_json', write):
                    bridge.eval_one_episode(env, SimpleNamespace(call=call))
                self.assertFalse(commands)
                after_rejection = requests[rejected_index + 1]
                self.assertIn('additional intervention rejected', after_rejection['rejected_action'])
                self.assertEqual(after_rejection['step'], maximum)
                self.assertEqual(after_rejection['retries']['grasp'], maximum)
                self.assertEqual(after_rejection['state'], 'TAKEOVER' if exhausted_limit == 'retries' else 'LOW_POLICY')
                self.assertEqual(after_rejection['max_retries_per_subgoal'], 6)
                self.assertEqual(len(env.executed), maximum + 1)
                self.assertAlmostEqual(env.executed[-1]['left_arm_joint_state'][0], 0.03)
                episode = next(session.glob('plug_in_charger/seed_*/*/summary.json')).parent
                summary = json.loads((episode / 'summary.json').read_text())
                self.assertEqual(summary['status'], 'success')
                self.assertTrue(summary['official_terminal'])
                self.assertEqual(summary['retries']['grasp'], maximum)
                self.assertEqual(summary['takeovers'], 1 if exhausted_limit == 'retries' else 2)
                events = [json.loads(line) for line in (episode / 'interventions.jsonl').read_text().splitlines()]
                self.assertEqual(sum(e['event'] == 'intervention_rejected' for e in events), 1)
                self.assertFalse((episode / 'error.json').exists())


if __name__ == '__main__':
    unittest.main()
