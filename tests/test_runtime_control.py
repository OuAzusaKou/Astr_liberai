import json
from pathlib import Path
import tempfile
import unittest
from astra.runtime_control import overlay_request, wire_command, completed_episode_metrics


class RuntimeControlTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.session = Path(self.tmp.name)
        self.folder = self.session / 'task/seed_0/layout_0_test'
        self.auth = dict(status='active', authorization_id='user_v1', user_instruction='unlimited recovery; no termination', unlimited_recovery=True, allow_agent_termination=False)
        (self.session / 'runtime_control.json').write_text(json.dumps(self.auth))
        self.request = dict(request_id='layout_0_test:0', episode='layout_0_test', state='LOW_POLICY', takeover_count=10, retries={'insert':6}, active_recovery_subgoal=None)
        self.pending()

    def pending(self):
        index = int(self.request['request_id'].split(':')[1])
        f = self.folder / 'requests' / f'{index:06d}' / 'request.json'
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(self.request))
        (self.session / 'controller_status.json').write_text(json.dumps(dict(request=str(f), response=str(f.with_name('response.json')))))
        return f

    def command(self, mode, **extra):
        return dict(request_id=self.request['request_id'], mode=mode, subgoal='insert', **extra)

    def publish(self, cmd):
        wire = wire_command(self.session, self.request, cmd)
        self.pending().with_name('response.json').write_text(json.dumps(wire))
        index = int(self.request['request_id'].split(':')[1])+1
        self.request['request_id'] = f'layout_0_test:{index}'
        self.pending()
        return wire

    def test_unlimited_legacy_low_policy_transport_and_ledger(self):
        for n in range(12):
            wire = self.publish(self.command('takeover', new_attempt=True))
            self.assertEqual(wire['mode'], 'correct')
        logical = overlay_request(self.session, self.request)
        self.assertEqual(logical['takeover_count'], 11)
        self.assertEqual(logical['retries']['insert'], 18)
        self.assertIsNone(logical['max_retries_per_subgoal'])
        with self.assertRaises(ValueError):
            wire_command(self.session, self.request, self.command('approve'))
        wire = self.publish(self.command('hand_back', recovery_evidence='verified grip'))
        self.assertEqual(wire['mode'], 'replan')
        self.assertEqual(overlay_request(self.session, self.request)['state'], 'LOW_POLICY')
        metrics = completed_episode_metrics(self.session, self.folder)
        self.assertEqual(metrics['transport_corrections'], 12)
        self.assertEqual(metrics['transport_handbacks'], 1)

    def test_legacy_active_takeover_preserves_native_subgoal(self):
        self.request.update(state='TAKEOVER', active_recovery_subgoal='insert')
        self.pending()
        cmd = self.command('takeover', new_attempt=True)
        cmd['subgoal']='regrasp'
        wire = self.publish(cmd)
        self.assertEqual(wire['mode'], 'takeover')
        self.assertFalse(wire['new_attempt'])
        self.assertEqual(wire['subgoal'], 'insert')
        logical = overlay_request(self.session, self.request)
        self.assertEqual(logical['retries']['regrasp'],1)
        self.assertEqual(logical['takeover_count'],10)
        self.assertEqual(self.publish(self.command('hand_back', recovery_evidence='recovered'))['mode'], 'hand_back')

    def test_no_termination_or_historical_relabel(self):
        for mode in ['abort','blocked']:
            with self.assertRaises(ValueError):
                wire_command(self.session,self.request,self.command(mode))
        self.assertIsNone(completed_episode_metrics(self.session,self.folder))

    def test_native_unlimited_preserves_mode_and_new_attempt(self):
        self.request['unlimited_recovery']=True
        self.pending()
        wire = self.publish(self.command('takeover',new_attempt=True))
        self.assertEqual(wire['mode'],'takeover')
        self.assertTrue(wire['new_attempt'])
        self.assertEqual(wire['runtime_control']['after']['transport_corrections'],0)
