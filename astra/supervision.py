"""Exclusive publication for independently owned live worker decisions."""
import json
import os
from pathlib import Path
import uuid
import math
import time


def with_runtime_limits(session, request):
    """Old live bridges omit limits in requests; preserve their recorded manifest."""
    result = dict(request)
    path = Path(session) / 'run_manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {}
    for key in ('max_retries_per_subgoal', 'max_takeovers'):
        if key not in result and key in manifest:
            result[key] = manifest[key]
    from astra.live_recovery import overlay_request
    result = overlay_request(session, result)
    from astra.runtime_control import overlay_request as runtime_overlay
    return runtime_overlay(session, result)


def check_supervisor_directive(session, owner):
    directive_path = session / 'supervisor_directive.json'
    if not directive_path.exists():
        return
    directive = json.loads(directive_path.read_text())
    acknowledgement = session / 'supervisor_directive_ack.json'
    ack = json.loads(acknowledgement.read_text()) if acknowledgement.exists() else {}
    if ack.get('version') != directive['version'] or ack.get('owner_id') != owner.get('owner_id'):
        raise PermissionError(
            f'New user supervision instructions: READ {directive_path} and the updated AGENTS.md. '
            f'Then write {acknowledgement} with version={directive["version"]!r}, '
            f'owner_id={owner.get("owner_id")!r}, acknowledged_at, and your understanding. '
            'Reinspect the current request before sending a decision; do not reset the scene.')


def completion_transition(request, command, previous):
    """Track declared final completion using verified native action time, not wall time."""
    episode, step = request['episode'], request['step']
    watch = dict(previous) if previous and previous.get('episode') == episode else {}
    if command.get('completion_candidate'):
        if not command.get('completion_evidence'):
            raise ValueError('Final completion candidate requires visual evidence')
        if watch.get('status') == 'pending_official':
            raise ValueError('An existing completion window cannot be restarted by declaring completion again')
        if watch and step <= max(watch['candidate_step'], watch.get('replan_step', watch['candidate_step'])):
            raise ValueError('A new completion candidate requires executed adjustment and new evidence')
        frequency = request['observation'].get('additional_info', {}).get('frequency')
        if not isinstance(frequency, (int, float)) or not math.isfinite(frequency) or frequency <= 0:
            raise ValueError('Verified native action frequency unavailable; cannot time completion window')
        watch = dict(episode=episode, candidate_step=step, candidate_request_id=request['request_id'],
                     evidence=command['completion_evidence'], frequency_hz=frequency,
                     clock_basis='native collect_interval physics steps per environment action; frequency from observation',
                     window_s=1.0, status='pending_official', created_at=time.time())
    if watch.get('status') == 'pending_official':
        elapsed = (step - watch['candidate_step']) / watch['frequency_hz']
        watch['elapsed_sim_s'] = elapsed
        mode = command['mode']
        if mode == 'approve':
            remaining = math.ceil((watch['window_s'] - elapsed) * watch['frequency_hz'] - 1e-8)
            if remaining <= 0:
                raise ValueError('Final completion has lacked official success for 1.0 simulation second. '
                                 'Diagnose and change the plan using replan/correct/takeover; do not keep approving idle actions.')
            if command.get('execute', 1) > remaining:
                raise ValueError(f'Completion confirmation window has only {remaining} action(s) remaining; shorten this approval.')
        elif mode in {'replan', 'correct', 'takeover', 'hand_back'}:
            watch.update(status='adjusting_plan', replan_step=step,
                         replan_request_id=request['request_id'], replan_reason=command['reason'])
    return watch


def publish_decision(session, path, command):
    session, path = Path(session).resolve(), Path(path).resolve()
    if not path.is_relative_to(session):
        raise ValueError('Response outside assigned worker')
    owner_file = session / 'supervisor_owner.json'
    owner = {}
    if owner_file.exists():
        owner = json.loads(owner_file.read_text())
        if owner['owner_id'] != os.environ.get('ASTRA_SUPERVISOR_ID'):
            raise PermissionError('Worker has an independent supervisor; do not submit from another session')
        command['supervisor_id'] = owner['owner_id']
    check_supervisor_directive(session, owner)
    status_path = session / 'controller_status.json'
    watch = None
    if status_path.exists():
        status = json.loads(status_path.read_text())
        request = json.loads(Path(status['request']).read_text())
        if request['request_id'] != command['request_id'] or Path(status['response']).resolve() != path:
            raise ValueError('Decision no longer matches the pending worker request')
        from astra.runtime_control import authorization, overlay_request as runtime_overlay, wire_command as runtime_wire
        from astra.live_recovery import context, overlay_request, wire_command
        if authorization(session) is not None:
            from astra.bridge import decision_for
            decision_for(runtime_overlay(session, request), command)
            native_command = runtime_wire(session, request, command)
        elif context(session, request) is not None:
            from astra.bridge import decision_for
            decision_for(overlay_request(session, request), command)
            native_command = wire_command(session, request, command)
        else:
            native_command = command
        watch_path = session / 'completion_watch.json'
        previous = json.loads(watch_path.read_text()) if watch_path.exists() else {}
        watch = completion_transition(request, command, previous)
        command = native_command
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(command, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        # Atomic publication without replacing any other supervisor's response.
        os.link(temporary, path)
        if watch:
            watch_path.write_text(json.dumps(watch, ensure_ascii=False, indent=2) + '\n')
            with (session / 'completion_checks.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(request_id=command['request_id'], mode=command['mode'], **watch), ensure_ascii=False) + '\n')
    finally:
        temporary.unlink(missing_ok=True)
