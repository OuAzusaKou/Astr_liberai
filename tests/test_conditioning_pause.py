import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra.conditioning_pause import pause_for_main


class ConditioningPauseTests(unittest.TestCase):
    def test_stale_resume_cannot_unpause_live_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); episode = folder / 'episode'; episode.mkdir()
            resume = folder / 'conditioning_resume.json'
            resume.write_text(json.dumps(dict(pause_id='old', action='retry',
                              authorized_by='main_session', reason='old repair')))
            def main_repairs(_):
                status = json.loads((folder / 'controller_status.json').read_text())
                self.assertEqual(status['status'], 'paused_for_main')
                self.assertEqual(status['step'], 17)
                self.assertFalse((episode / 'summary.json').exists())
                resume.write_text(json.dumps(dict(pause_id=status['pause_id'], action='retry',
                                  authorized_by='main_session', reason='Interface repaired')))
            with patch('astra.conditioning_pause.time.sleep', side_effect=main_repairs) as wait:
                pause_for_main(folder, episode, 17, 'episode:20', ValueError('bad receipt'))
                self.assertEqual(wait.call_count, 1)
            incident = json.loads((folder / 'conditioning_incident.json').read_text())
            self.assertEqual(incident['status'], 'retry_authorized')
            self.assertFalse((episode / 'summary.json').exists())
