"""Keep the live scene while the main session resolves an L2 interface fault."""
import json
from pathlib import Path
import time
import uuid


def pause_for_main(session, folder, step, request_id, error):
    from astra.bridge import atomic_json, append_event
    session, folder = Path(session), Path(folder)
    pause_id = uuid.uuid4().hex
    incident = dict(status='paused_for_main', pause_id=pause_id, request_id=request_id,
                    step=step, episode=folder.name, error=f'{type(error).__name__}: {error}',
                    created_at=time.time(), resume_instruction=
                    'Main session: diagnose first, then use scripts/resume_conditioning.py with this exact pause_id. '
                    'Execution supervisors may not authorize resume or terminate the episode.')
    atomic_json(session / 'conditioning_incident.json', incident)
    append_event(folder / 'conditioning_incidents.jsonl', incident)
    atomic_json(session / 'controller_status.json', incident)
    resume_path = session / 'conditioning_resume.json'
    while True:
        try:
            resume = json.loads(resume_path.read_text())
        except (OSError, ValueError):
            resume = {}
        if (resume.get('pause_id') == pause_id and resume.get('action') == 'retry'
                and resume.get('authorized_by') == 'main_session'
                and isinstance(resume.get('reason'), str) and resume['reason'].strip()):
            append_event(folder / 'conditioning_incidents.jsonl', dict(
                event='main_authorized_retry', resumed_at=time.time(), **resume))
            atomic_json(session / 'conditioning_incident.json', dict(
                incident, status='retry_authorized', resumed_at=time.time(), resume=resume))
            return
        time.sleep(0.5)
