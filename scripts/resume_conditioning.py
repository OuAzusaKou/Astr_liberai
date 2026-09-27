"""Main-session-only acknowledgement of a diagnosed live conditioning fault."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra.bridge import atomic_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--session', required=True, type=Path)
p.add_argument('--pause-id', required=True)
p.add_argument('--reason', required=True)
a = p.parse_args()
if os.environ.get('ASTRA_SUPERVISOR_ID'):
    p.error('Only the main session may authorize an infrastructure retry')
s = a.session.resolve()
status = json.loads((s / 'controller_status.json').read_text())
if status.get('status') != 'paused_for_main' or status.get('pause_id') != a.pause_id:
    p.error('Stale or non-paused controller; inspect the current incident')
if not a.reason.strip():
    p.error('A diagnosed recovery reason is required')
atomic_json(s / 'conditioning_resume.json', dict(pause_id=a.pause_id, action='retry',
            authorized_by='main_session', reason=a.reason, authorized_at=time.time()))
print('Authorized retry for', a.pause_id)
