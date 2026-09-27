"""Terminate only a verified, already-blocked Astra worker so its GPU can be reused."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);p.add_argument('--pid',type=int,required=True);a=p.parse_args()
session=a.session.resolve();status=json.loads((session/'controller_status.json').read_text())
if status.get('status')!='blocked_or_error':raise RuntimeError('Worker is not in a recorded blocked/error state')
proc=Path('/proc')/str(a.pid)
args=(proc/'cmdline').read_bytes().decode().split('\0')
if '-m' not in args or 'astra.run' not in args or '--session' not in args or args[args.index('--session')+1]!=str(session):
    raise RuntimeError('PID does not belong to the exact requested Astra worker')
before={'runner_pid':a.pid,'session':str(session),'signal':'SIGTERM','time':time.time(),'reason':'Already blocked worker stuck in simulator teardown; preserve result and release resources for remaining layouts','controller_status':status}
(session/'blocked_worker_cleanup.json').write_text(json.dumps(before,indent=2)+'\n')
os.kill(a.pid,signal.SIGTERM)
deadline=time.monotonic()+45
while proc.exists() and time.monotonic()<deadline:
    time.sleep(1)
before['runner_exited']=not proc.exists();before['checked_at']=time.time()
(session/'blocked_worker_cleanup.json').write_text(json.dumps(before,indent=2)+'\n')
print(json.dumps(before))
if not before['runner_exited']:raise RuntimeError('Runner still present; do not start a replacement yet')
