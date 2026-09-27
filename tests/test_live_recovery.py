import json
from pathlib import Path
import tempfile
import unittest

from astra.live_recovery import overlay_request, completed_episode_metrics
from astra.supervision import publish_decision


class LiveRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.episode = 'layout_0_test'
        self.auth = dict(status='active', authorization_id='user-extra3', episode=self.episode,
            episode_relative_path='task/seed_0/'+self.episode, subgoal='mug',
            user_instruction='add three attempts beyond the seven audited attempts',
            transport='legacy_correct_explicit_actions', native_takeovers=3,
            native_retries={'mug':3}, audited_attempts_before=7, additional_attempts=3, max_takeovers=10)
        (self.folder/'live_recovery_extension.json').write_text(json.dumps(self.auth))

    def request(self, index, **changes):
        r=dict(episode=self.episode, request_id=f'{self.episode}:{index}', step=714+index,
            state='LOW_POLICY', takeover_count=3, retries={'mug':3}, demo_segments=['mug'],
            proposed_actions=[{}], observation={'additional_info': {'frequency':25}})
        r.update(changes)
        return r

    def publish(self, index, mode, **kwargs):
        r=self.request(index)
        path=self.folder/self.auth['episode_relative_path']/'requests'/f'{index:06d}'
        path.mkdir(parents=True,exist_ok=True)
        (path/'request.json').write_text(json.dumps(r))
        (self.folder/'controller_status.json').write_text(json.dumps(dict(status='waiting_for_agent',
            request=str(path/'request.json'), response=str(path/'response.json'))))
        cmd=dict(request_id=r['request_id'], mode=mode, reason='reviewed evidence', subgoal='mug', demo_segment='mug', **kwargs)
        publish_decision(self.folder,path/'response.json',cmd)
        return json.loads((path/'response.json').read_text())

    def test_direct_transport_and_exclusive_handback_preserve_history(self):
        wire=self.publish(0,'takeover',new_attempt=True,actions=[{'explicit':'unchanged'}])
        self.assertEqual(wire['mode'],'correct')
        self.assertEqual(wire['live_recovery']['logical_mode'],'takeover')
        self.assertEqual(wire['actions'],[{'explicit':'unchanged'}])
        logical=overlay_request(self.folder,self.request(1))
        self.assertEqual((logical['state'],logical['retries']['mug'],logical['takeover_count']),('TAKEOVER',8,4))
        self.assertEqual(logical['native_retries'],{'mug':3})
        for mode,extra in [('approve',{'execute':1}),('correct',{'actions':[{}]}),('replan',{})]:
            with self.assertRaises(ValueError): self.publish(1,mode,**extra)
        with self.assertRaises(ValueError): self.publish(1,'hand_back')
        wire=self.publish(1,'hand_back',recovery_evidence='open safe pose, fresh object observation')
        self.assertEqual(wire['mode'],'replan')
        logical=overlay_request(self.folder,self.request(2))
        self.assertEqual(logical['state'],'LOW_POLICY')
        self.assertEqual(logical['retries']['mug'],8)

    def test_three_extra_attempts_count_separately_from_batches_and_events(self):
        self.publish(0,'takeover',new_attempt=True,actions=[{}])
        self.publish(1,'takeover',new_attempt=False,actions=[{}])
        self.publish(2,'takeover',new_attempt=True,actions=[{}])
        self.publish(3,'takeover',new_attempt=True,actions=[{}])
        with self.assertRaises(ValueError): self.publish(4,'takeover',new_attempt=True,actions=[{}])
        self.assertFalse((self.folder/self.auth['episode_relative_path']/'requests/000004/response.json').exists())
        self.publish(4,'takeover',new_attempt=False,actions=[{}])
        logical=overlay_request(self.folder,self.request(5))
        self.assertEqual((logical['retries']['mug'],logical['takeover_count']),(10,4))
        self.assertEqual(logical['live_recovery_extension']['additional_remaining'],0)

    def test_scope_native_divergence_and_new_attempt_required(self):
        other=self.request(0,episode='layout_1_other')
        self.assertIs(overlay_request(self.folder,other),other)
        for changes in [dict(state='TAKEOVER'),dict(retries={'mug':4}),dict(takeover_count=4)]:
            with self.assertRaises(ValueError): overlay_request(self.folder,self.request(0,**changes))
        with self.assertRaises(ValueError): self.publish(0,'takeover',new_attempt=False,actions=[{}])
        with self.assertRaises(ValueError): self.publish(0,'takeover',actions=[{}])

    def test_failed_publication_cannot_double_charge(self):
        self.publish(0,'takeover',new_attempt=True,actions=[{}])
        with self.assertRaises(FileExistsError): self.publish(0,'takeover',new_attempt=True,actions=[{}])
        self.assertEqual(overlay_request(self.folder,self.request(1))['retries']['mug'],8)

    def test_wrong_internal_ledger_path_fails_closed(self):
        self.publish(0,'takeover',new_attempt=True,actions=[{}])
        self.auth['episode_relative_path']='other/seed_0/'+self.episode
        (self.folder/'live_recovery_extension.json').write_text(json.dumps(self.auth))
        with self.assertRaises(ValueError): overlay_request(self.folder,self.request(1))

    def test_terminal_metrics_include_last_published_batch_after_layout_changed(self):
        self.publish(0,'takeover',new_attempt=True,actions=[{}])
        self.publish(1,'takeover',new_attempt=True,actions=[{}])
        (self.folder/'controller_status.json').write_text(json.dumps(dict(request=str(self.folder/'next/requests/0/request.json'))))
        metrics=completed_episode_metrics(self.folder,self.folder/self.auth['episode_relative_path'])
        self.assertEqual((metrics['additional_used'],metrics['logical_retries']['mug']),(2,9))
        self.assertEqual(metrics['native_correct_transports'],2)


if __name__ == '__main__':
    unittest.main()
