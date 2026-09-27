"""Explicit, audited arbitration for an authorized extension on a legacy bridge.

The legacy bridge retains its native counters. Its validated direct-action
transport is `correct`; logical takeover state and additional attempts live here.
Only an episode-scoped authorization activates this compatibility controller.
Published responses are the durable ledger, including their native/logical modes.
"""
import json
from pathlib import Path


def context(session, request, *, verify_pending=True):
    session = Path(session).resolve()
    path = session / 'live_recovery_extension.json'
    if not path.exists():
        return None
    auth = json.loads(path.read_text())
    if auth.get('status') != 'active' or auth['episode'] != request['episode']:
        return None
    if not auth.get('user_instruction') or auth.get('transport') != 'legacy_correct_explicit_actions':
        raise ValueError('Live extension requires explicit user authorization and transport')
    if request['state'] != 'LOW_POLICY':
        raise ValueError('Compatibility controller requires native LOW_POLICY state')
    if (request['takeover_count'] != auth['native_takeovers'] or
            request['retries'] != auth['native_retries']):
        raise ValueError('Native recovery counters changed outside the compatibility controller')
    folder = (session / auth['episode_relative_path']).resolve()
    if not folder.is_relative_to(session):
        raise ValueError('Extension episode is outside worker')
    if folder.name != auth['episode']:
        raise ValueError('Authorization folder does not match episode')
    status_path = session / 'controller_status.json'
    if verify_pending and status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get('request') and Path(status['request']).resolve().parents[2] != folder:
            raise ValueError('Authorization folder differs from the actual pending request episode')
    used, events, transport_corrections, transport_replans, active = 0, 0, 0, 0, False
    for response_path in sorted((folder / 'requests').glob('*/response.json')):
        response = json.loads(response_path.read_text())
        meta = response.get('live_recovery')
        if not meta or meta.get('authorization_id') != auth['authorization_id']:
            continue
        if response['request_id'].split(':')[0] != auth['episode']:
            raise ValueError('Extension ledger episode mismatch')
        mode = meta['logical_mode']
        if mode == 'takeover':
            if response['mode'] != 'correct':
                raise ValueError('Invalid takeover transport in ledger')
            new = meta['new_attempt']
            if not active and not new:
                raise ValueError('New takeover must start an attempt')
            used += int(new)
            events += int(not active)
            transport_corrections += 1
            active = True
        elif mode == 'hand_back':
            if not active or response['mode'] != 'replan':
                raise ValueError('Invalid hand-back transport in ledger')
            transport_replans += 1
            active = False
        elif active and mode not in {'abort', 'blocked'}:
            raise ValueError('Non-takeover action while compatibility controller owns robot')
    if used > auth['additional_attempts']:
        raise ValueError('Extension attempt ledger exceeds authorization')
    return dict(authorization=auth, active=active, attempts_used=used, takeover_events=events,
                native_correct_transports=transport_corrections, native_replan_handbacks=transport_replans)


def completed_episode_metrics(session, episode_folder):
    """Read the entire durable ledger after terminal, even if another layout started."""
    session, episode_folder = Path(session).resolve(), Path(episode_folder).resolve()
    path = session / 'live_recovery_extension.json'
    if not path.exists():
        return None
    auth = json.loads(path.read_text())
    if (session / auth['episode_relative_path']).resolve() != episode_folder:
        return None
    requests = sorted((episode_folder / 'requests').glob('*/request.json'))
    if not requests:
        return None
    ctx = context(session, json.loads(requests[-1].read_text()), verify_pending=False)
    if ctx is None:
        return None
    return dict(authorization_id=auth['authorization_id'], episode=auth['episode'],
        episode_path=str(episode_folder), audited_attempts_before=auth['audited_attempts_before'],
        additional_attempts=auth['additional_attempts'], additional_used=ctx['attempts_used'],
        additional_remaining=auth['additional_attempts']-ctx['attempts_used'],
        logical_takeovers=auth['native_takeovers']+ctx['takeover_events'],
        logical_retries={**auth['native_retries'], auth['subgoal']:auth['audited_attempts_before']+ctx['attempts_used']},
        native_takeovers=auth['native_takeovers'], native_retries=auth['native_retries'],
        native_correct_transports=ctx['native_correct_transports'], native_replan_handbacks=ctx['native_replan_handbacks'],
        count_basis='all published responses including last terminal batch; execution is recorded separately in native interventions')


def overlay_request(session, request):
    ctx = context(session, request)
    if ctx is None:
        return request
    auth = ctx['authorization']
    result = dict(request)
    result['native_control_state'] = request['state']
    result['native_takeover_count'] = request['takeover_count']
    result['native_retries'] = request['retries']
    result['state'] = 'TAKEOVER' if ctx['active'] else 'LOW_POLICY'
    result['takeover_count'] = auth['native_takeovers'] + ctx['takeover_events']
    result['retries'] = {**request['retries'], auth['subgoal']: auth['audited_attempts_before'] + ctx['attempts_used']}
    result['max_retries_per_subgoal'] = auth['audited_attempts_before'] + auth['additional_attempts']
    result['max_takeovers'] = auth['max_takeovers']
    result['live_recovery_extension'] = dict(authorization_id=auth['authorization_id'],
        additional_attempts=auth['additional_attempts'], additional_used=ctx['attempts_used'],
        additional_remaining=auth['additional_attempts']-ctx['attempts_used'],
        audited_attempts_before=auth['audited_attempts_before'],
        transport='native correct for explicit takeover actions; native replan for hand-back',
        native_correct_transports=ctx['native_correct_transports'], native_replan_handbacks=ctx['native_replan_handbacks'])
    return result


def wire_command(session, request, command):
    ctx = context(session, request)
    if ctx is None:
        return command
    auth, mode = ctx['authorization'], command['mode']
    if ctx['active'] and mode not in {'takeover', 'hand_back', 'abort', 'blocked'}:
        raise ValueError('Explicit hand-back required before any low-policy or correction action')
    if mode in {'takeover', 'hand_back'} and command['subgoal'] != auth['subgoal']:
        raise ValueError('Live recovery extension is scoped to the authorized subgoal')
    wire = dict(command)
    meta = dict(authorization_id=auth['authorization_id'], logical_mode=mode,
                native_state=request['state'], audited_attempts_before=auth['audited_attempts_before'],
                additional_attempts=auth['additional_attempts'], used_before=ctx['attempts_used'],
                transport_note='Explicit compatibility arbitration; native counters unchanged, logical ledger preserved')
    if mode == 'takeover':
        if type(command.get('new_attempt')) is not bool:
            raise ValueError('Takeover requires explicit boolean new_attempt')
        new = command['new_attempt']
        if not ctx['active'] and not new:
            raise ValueError('Starting takeover must consume an additional attempt')
        if ctx['attempts_used'] + int(new) > auth['additional_attempts']:
            raise ValueError('User-authorized additional recovery budget exhausted')
        if auth['native_takeovers'] + ctx['takeover_events'] + int(not ctx['active']) > auth['max_takeovers']:
            raise ValueError('Episode takeover event budget exhausted')
        meta.update(new_attempt=new, used_after=ctx['attempts_used']+int(new))
        wire['mode'] = 'correct'
        wire.pop('new_attempt', None)
    elif mode == 'hand_back':
        if not ctx['active'] or not command.get('recovery_evidence'):
            raise ValueError('Hand-back requires logical takeover and recovery evidence')
        wire['mode'] = 'replan'
    wire['live_recovery'] = meta
    return wire
