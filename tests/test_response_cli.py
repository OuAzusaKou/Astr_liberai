import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ResponseCliTests(unittest.TestCase):
    def test_continuous_takeover_explicit_retry_marker_and_budget_guard(self):
        script=Path(__file__).resolve().parents[1]/'scripts/respond_live.py'
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);request=folder/'request.json';response=folder/'response.json'
            value=dict(request_id='r',episode='layout_0_test',step=3,state='TAKEOVER',
                       demo_segments=['grasp'],retries={'grasp':1},takeover_count=1,
                       observation={'images':{},'additional_info':{'frequency':25}})
            request.write_text(json.dumps(value));(folder/'controller_status.json').write_text(json.dumps(dict(status='waiting_for_agent',request=str(request),response=str(response))))
            (folder/'run_manifest.json').write_text(json.dumps(dict(max_retries_per_subgoal=3,max_takeovers=10)))
            actions=folder/'actions.json';actions.write_text(json.dumps([{'left_arm_joint_state':[0]*6}]))
            args=[sys.executable,str(script),str(folder),'r','takeover','1','changed approach after observed failure','--segment','grasp','--actions-file',str(actions)]
            missing=subprocess.run(args,capture_output=True,text=True)
            self.assertNotEqual(missing.returncode,0);self.assertFalse(response.exists())
            new=subprocess.run(args+['--new-attempt'],capture_output=True,text=True)
            self.assertEqual(new.returncode,0,new.stderr);self.assertIs(json.loads(response.read_text())['new_attempt'],True)
            response.unlink()
            same=subprocess.run(args+['--continue-attempt'],capture_output=True,text=True)
            self.assertEqual(same.returncode,0,same.stderr);self.assertIs(json.loads(response.read_text())['new_attempt'],False)
            response.unlink();value['retries']['grasp']=3;request.write_text(json.dumps(value))
            over=subprocess.run(args+['--new-attempt'],capture_output=True,text=True)
            self.assertNotEqual(over.returncode,0);self.assertFalse(response.exists());self.assertIn('budget would be exceeded',over.stderr)


if __name__=='__main__':unittest.main()
