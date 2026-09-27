"""Versioned user-authorized control changes for already loaded bridges.

Legacy transport preserves native action validation. Every response carries the
logical decision and counters; native correct transports are never reported as
independent low-policy behavior. No scene reset or process injection is used.
"""
import json
from pathlib import Path


def authorization(session):
    path = Path(session) / 'runtime_control.json'
    if not path.exists():
        return None
    auth = json.loads(path.read_text())
    if auth.get('status') != 'active':
        return None
    if not auth.get('user_instruction') or not auth.get('authorization_id'):
        raise ValueError('Runtime control requires explicit user authorization')
    if auth.get('unlimited_recovery') is not True or auth.get('allow_agent_termination') is not False:
        raise ValueError('Unsupported runtime control policy')
    return auth


def ledger(session, request, folder=None):
    auth = authorization(session)
    if auth is None:
        return None
    if folder is None:
        status = json.loads((Path(session) / 'controller_status.json').read_text())
        pending = Path(status['request'])
        if json.loads(pending.read_text())['episode'] != request['episode']:
            raise ValueError('Runtime control episode differs from pending episode')
        folder = pending.parents[2]
    folder = Path(folder).resolve()
    if not folder.is_relative_to(Path(session).resolve()) or folder.name != request['episode']:
        raise ValueError('Invalid runtime control episode path')
    last = None
    for f in sorted((folder / 'requests').glob('*/response.json')):
        response = json.loads(f.read_text())
        meta = response.get('runtime_control')
        if meta and meta.get('authorization_id') == auth['authorization_id']:
            if response['request_id'].split(':')[0] != request['episode']:
                raise ValueError('Runtime ledger episode mismatch')
            last = meta['after']
    if last is None:
        last = dict(takeovers=request.get('takeover_count', 0), retries=dict(request.get('retries', {})),
                    active=request.get('state') == 'TAKEOVER',
                    active_subgoal=request.get('active_recovery_subgoal'),
                    transport_corrections=0, transport_handbacks=0)
    return auth, last


def overlay_request(session, request):
    found = ledger(session, request)
    if found is None:
        return request
    auth, counters = found
    result = dict(request)
    result.update(native_control_state=request['state'], native_takeover_count=request.get('takeover_count', 0),
                  native_retries=request.get('retries', {}),
                  state='TAKEOVER' if counters['active'] else 'LOW_POLICY',
                  active_recovery_subgoal=counters['active_subgoal'],
                  takeover_count=counters['takeovers'], retries=counters['retries'],
                  max_retries_per_subgoal=None, max_takeovers=None,
                  unlimited_recovery=True, allow_agent_termination=False,
                  runtime_control_authorization=auth['authorization_id'])
    return result


def wire_command(session, request, command):
    found = ledger(session, request)
    if found is None:
        return command
    auth, before = found
    mode = command['mode']
    if mode in {'abort', 'blocked'}:
        raise ValueError('User forbids sub-agent termination. Preserve scene and report to main session; recovery is unlimited.')
    active = before['active']
    if active and mode not in {'takeover', 'hand_back'}:
        raise ValueError('Explicit hand-back required while agent owns control')
    after = {**before, 'retries': dict(before['retries'])}
    wire = dict(command)
    native_unlimited = request.get('unlimited_recovery') is True
    if mode == 'takeover':
        new = command.get('new_attempt', not active)
        if type(new) is not bool or (not active and not new):
            raise ValueError('New takeover requires an explicit recovery attempt')
        key = command['subgoal']
        if active and key != before['active_subgoal'] and not new:
            raise ValueError('Changed subgoal requires a new recovery attempt')
        after['takeovers'] += int(not active)
        after['retries'][key] = after['retries'].get(key, 0) + int(new)
        after.update(active=True, active_subgoal=key)
        if not native_unlimited:
            if request['state'] == 'TAKEOVER':
                # Keep the legacy bridge in its existing takeover event without
                # consuming its obsolete retry budget. Logical counters above
                # retain the real retry and subgoal, including changes.
                wire['new_attempt'] = False
                wire['subgoal'] = request['active_recovery_subgoal']
            else:
                wire['mode'] = 'correct'
                wire.pop('new_attempt', None)
                after['transport_corrections'] += 1
    elif mode == 'hand_back':
        if not active or not command.get('recovery_evidence'):
            raise ValueError('Hand-back requires active recovery and observed evidence')
        after.update(active=False, active_subgoal=None)
        if request['state'] != 'TAKEOVER':
            wire['mode'] = 'replan'
            after['transport_handbacks'] += 1
    wire['runtime_control'] = dict(authorization_id=auth['authorization_id'],
        logical_mode=mode, logical_subgoal=command['subgoal'],
        logical_new_attempt=command.get('new_attempt'), before=before, after=after,
        native_mode=wire['mode'], native_subgoal=wire['subgoal'],
        transport_note='Legacy validated direct actions; logical counters are authoritative after this authorization')
    return wire


def completed_episode_metrics(session, folder):
    files = sorted((Path(folder) / 'requests').glob('*/request.json'))
    if not files or authorization(session) is None:
        return None
    request = json.loads(files[-1].read_text())
    found = ledger(session, request, folder)
    auth, counters = found
    # Do not relabel historical episodes that ended before this authorization.
    used = any('runtime_control' in json.loads(f.read_text()) for f in (Path(folder) / 'requests').glob('*/response.json'))
    if not used:
        return None
    return dict(authorization_id=auth['authorization_id'], unlimited_recovery=True,
                allow_agent_termination=False, **counters)
