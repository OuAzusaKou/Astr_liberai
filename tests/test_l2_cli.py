"""Exercise the real response CLI at the file publication boundary."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


class L2CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.scripts = Path(__file__).resolve().parents[1] / 'scripts'
        self.request_path = self.folder / 'request.json'
        self.response_path = self.folder / 'response.json'
        self.cue_path = self.folder / 'current_cam_head.png'
        Image.new('RGB', (32, 24), (12, 34, 56)).save(self.cue_path)
        self.request = dict(
            request_id='r', episode='layout_0_test', step=0, state='PREPARE_CONDITIONING',
            worker_id='test', conditioning_mode='l3_l2', conditioning=None,
            instruction='Put the cup on the shelf.', l3_instruction='Put the cup on the shelf.',
            demo_segments=['grasp', 'home'], proposed_actions=[], rejected_action=None,
            retries={}, takeover_count=0,
            observation=dict(images={'cam_head': str(self.cue_path)}, state={},
                             instruction='Put the cup on the shelf.', additional_info={'frequency': 25}))
        self.save_request()
        (self.folder / 'controller_status.json').write_text(json.dumps(dict(
            status='waiting_for_agent', request=str(self.request_path), response=str(self.response_path))))
        self.condition = dict(l2_instruction='Grasp the cup.', cue_rgb_path=str(self.cue_path),
                              l2_bboxes_xyxy=[[3, 4, 16, 20]], box_source='astra_visual',
                              evidence='The cup occupies this region in the current global image.',
                              subgoal='grasp')
        self.condition_path = self.folder / 'condition.json'
        self.save_condition()

    def save_request(self):
        self.request_path.write_text(json.dumps(self.request))

    def save_condition(self):
        self.condition_path.write_text(json.dumps(self.condition))

    def run_response(self, mode='condition', count=0, flags=()):
        return subprocess.run([
            sys.executable, str(self.scripts / 'respond_live.py'), str(self.folder), 'r', mode,
            str(count), 'Reviewed current target and global cue.', '--segment', 'grasp',
            '--scene-review', 'Current global image shows the cup before grasping.', *flags],
            capture_output=True, text=True)

    def test_condition_publishes_complete_reviewed_payload_without_actions(self):
        result = self.run_response(flags=['--conditioning-file', str(self.condition_path)])
        self.assertEqual(result.returncode, 0, result.stderr)
        command = json.loads(self.response_path.read_text())
        self.assertEqual(command['conditioning'], self.condition)
        self.assertEqual(command['mode'], 'condition')
        self.assertNotIn('actions', command)
        self.assertNotIn('execute', command)

    def test_invalid_boxes_and_foreign_cue_never_publish(self):
        for change in ({'l2_bboxes_xyxy': [[0, 0, 33, 24]]},
                       {'cue_rgb_path': str(self.folder / 'demo_frame.png')},
                       {'l2_bboxes_xyxy': []}):
            with self.subTest(change=change):
                original = dict(self.condition)
                self.condition.update(change)
                self.save_condition()
                result = self.run_response(flags=['--conditioning-file', str(self.condition_path)])
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.response_path.exists())
                self.condition = original

    def test_condition_file_and_version_are_mutually_exclusive(self):
        result = self.run_response(flags=['--conditioning-file', str(self.condition_path),
                                          '--conditioning-version', 'v1'])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.response_path.exists())

    def test_action_decisions_cannot_attach_changed_condition(self):
        for mode in ('approve', 'correct', 'takeover'):
            with self.subTest(mode=mode):
                result = self.run_response(mode=mode, count=1,
                                           flags=['--conditioning-file', str(self.condition_path)])
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.response_path.exists())

    def test_replan_explicitly_confirms_current_version_and_rejects_stale(self):
        self.request.update(step=4, state='LOW_POLICY',
                            conditioning=dict(self.condition, conditioning_version='v1'))
        self.save_request()
        for flags in ([], ['--conditioning-version', 'stale']):
            result = self.run_response(mode='replan', flags=flags)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.response_path.exists())
        result = self.run_response(mode='replan', flags=['--conditioning-version', 'v1'])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(self.response_path.read_text())['conditioning_version'], 'v1')

    def test_condition_does_not_bypass_exclusive_supervisor(self):
        (self.folder / 'supervisor_owner.json').write_text(json.dumps({'owner_id': 'assigned-owner'}))
        result = self.run_response(flags=['--conditioning-file', str(self.condition_path)])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('independent supervisor', result.stderr)
        self.assertFalse(self.response_path.exists())

    def test_inspect_reports_native_cue_dimensions_without_modifying_original(self):
        before = self.cue_path.read_bytes()
        result = subprocess.run([sys.executable, str(self.scripts / 'inspect_live.py'), str(self.folder)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('conditioning_review_required', result.stdout)
        cue_line = next(line for line in result.stdout.splitlines() if line.startswith('current_cue_rgb '))
        cue = json.loads(cue_line.split(' ', 1)[1])
        self.assertEqual((cue['width'], cue['height']), (32, 24))
        self.assertEqual(cue['path'], str(self.cue_path))
        self.assertEqual(cue['camera'], 'cam_head')
        self.assertEqual(self.cue_path.read_bytes(), before)
        self.assertTrue((self.folder / 'live_contact.png').exists())


if __name__ == '__main__':
    unittest.main()
