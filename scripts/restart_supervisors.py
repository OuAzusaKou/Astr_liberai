#!/usr/bin/env python3
"""Restart exited or unresponsive supervisors without restarting model/scenes/runners."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.launch_core9_supervisors import read, launch, refresh, release, stop_child, terminate, POOL_TERMINAL, supervisor_executable
from astra.task_run import write_json


def unit_id(worker):
    return worker.get('worker_id', worker['task'])


def pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        return stat.rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False
    except OSError:
        return True  # Cannot prove exit: never risk a duplicate controller.


def restart_eligibility(folder):
    folder = Path(folder)
    schedule = read(folder / 'task_scheduler_status.json', {})
    if schedule.get('all_layouts_settled'):
        return False, 'all_layouts_settled'
    if schedule.get('status') not in {'waiting_for_supervisor', 'running_layout', 'starting_layout', 'paused_for_main'}:
        return False, 'task_not_active'
    if not pid_alive(schedule.get('manager_pid')):
        return False, 'task_manager_not_alive'
    state = read(folder / 'supervisor_status.json', {})
    if not state.get('owner_id') or (not state.get('pid') and state.get('status') != 'launch_error'):
        return False, 'no_prior_supervisor'
    if state.get('hostname') != socket.gethostname():
        return False, 'remote_exit_requires_remote_verification'
    if pid_alive(state['pid']):
        return False, 'supervisor_still_alive'
    controller = read(folder / 'controller_status.json', {})
    if controller.get('status') == 'waiting_for_agent':
        if Path(controller.get('response', '/nonexistent')).exists():
            return False, 'wait_for_pending_response_to_drain'
    elif controller.get('status') not in {None, 'starting_layout', 'paused_for_main', 'episode_finished'}:
        return False, 'wait_for_review_boundary'
    return True, 'old_process_exited'


def retry_delay(attempt):
    return min(300, 5 * 2 ** min(max(attempt - 1, 0), 6))


def ensure_watchdog(batch):
    batch = Path(batch).resolve()
    # The launched process owns a separate lock, so duplicate starters cannot supervise twice.
    lock = (batch / 'supervisor_restart.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return
    lock.close()
    with (batch / 'supervisor_restart.log').open('a') as log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--session', str(batch)],
                         cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)


def locate_request(folder, controller, request_id):
    request_path = controller.get('request')
    if request_path:
        request = read(Path(request_path), {})
        if request.get('request_id') == request_id:
            return Path(request_path), request
    task = read(Path(folder) / 'run_manifest.json', {}).get('tasks', [])
    task = task[0] if task else ''
    episode = controller.get('episode')
    roots = [Path(folder) / task / 'seed_0' / episode] if task and episode else []
    for root in roots:
        for path in root.glob('requests/*/request.json'):
            request = read(path, {})
            if request.get('request_id') == request_id:
                return path, request
    # Older handoff records may omit task/layout fields; the worker directory is
    # still isolated, so matching the exact request ID is safe and auditable.
    for path in Path(folder).rglob('request.json'):
        request = read(path, {})
        if request.get('request_id') == request_id:
            return path, request
    return None, {}


def request_for_incident(folder, controller, request_id):
    return locate_request(folder, controller, request_id)[1]


def resume_supervisor_timeout(folder, state):
    """Only the already diagnosed missing-supervisor timeout is auto-acknowledged."""
    if not state.get('ready'):
        return
    controller = read(folder / 'controller_status.json', {})
    incident = read(folder / 'conditioning_incident.json', {})
    loss = read(folder / 'supervisor_loss_incident.json', {})
    if (controller.get('status') == 'paused_for_main'
            and incident.get('pause_id') == controller.get('pause_id')
            and loss.get('event') in {'supervisor_lost', 'supervisor_available'}
            and request_for_incident(folder, loss.get('controller', {}), incident.get('request_id', '')).get('request_id') == incident.get('request_id')
            and incident.get('error') == 'TimeoutError: Agent decision timeout; no unsupervised fallback'):
        write_json(folder / 'conditioning_resume.json', dict(
            pause_id=incident['pause_id'], action='retry', authorized_by='main_session',
            authorized_at=time.time(), reason='User-authorized automatic restart: previous supervisor exited; replacement owns lock and completed visual review.',
            supervisor_id=state['owner_id']))


def stalled_request(entry, now, timeout):
    """Return a stall diagnosis only when a response is overdue and Codex made no progress."""
    folder, state = entry['folder'], entry['state']
    controller = read(folder / 'controller_status.json', {})
    controller_status = controller.get('status')
    if controller_status not in {'waiting_for_agent', 'paused_for_main'}:
        return None
    if controller_status == 'waiting_for_agent':
        request_path = Path(controller.get('request', '/nonexistent'))
        response = Path(controller.get('response', '/nonexistent'))
        if response.exists() or not request_path.is_file():
            return None
        request = read(request_path, {})
    else:
        incident = read(folder / 'conditioning_incident.json', {})
        if incident.get('error') != 'TimeoutError: Agent decision timeout; no unsupervised fallback':
            return None
        request_path, request = locate_request(folder, controller, incident.get('request_id', ''))
        if not request_path:
            return None
        response = request_path.with_name('response.json')
        if response.exists():
            return None
    try:
        request_age = now - request_path.stat().st_mtime
        activity = Path(state['events']).stat()
    except OSError:
        return None
    signature = (activity.st_size, activity.st_mtime_ns)
    previous = (state.get('activity_size'), state.get('activity_mtime_ns'))
    if signature != previous:
        state.update(activity_size=signature[0], activity_mtime_ns=signature[1], last_activity_at=now)
        return None
    idle_for = now - state.get('last_activity_at', state.get('started_at', now))
    if request_age < timeout or idle_for < timeout:
        return None
    return dict(reason='supervisor_no_response', request_id=request.get('request_id'),
                episode=request.get('episode'), step=request.get('step'),
                request_path=str(request_path), response_path=str(response),
                request_age_seconds=round(request_age, 1), idle_seconds=round(idle_for, 1),
                event_log=str(state['events']), event_log_size=activity.st_size,
                detected_at=now, old_owner_id=state.get('owner_id'), old_pid=state.get('pid'))


def record_stall_loss(folder, stall):
    """Persist a manager-owned loss gate when a live supervisor is replaced."""
    controller = read(Path(folder) / 'controller_status.json', {})
    if controller.get('status') != 'paused_for_main':
        return
    controller = dict(controller, request=stall['request_path'])
    write_json(Path(folder) / 'supervisor_loss_incident.json', dict(
        event='supervisor_lost', status='paused_for_main', pause_id=controller.get('pause_id'),
        created_at=time.time(), error='Supervisor unresponsive while a decision was pending',
        controller=controller, request_id=stall.get('request_id'), episode=stall.get('episode'),
        step=stall.get('step'), old_owner_id=stall.get('old_owner_id'), old_pid=stall.get('old_pid'),
        detected_at=stall.get('detected_at'), reason=stall['reason']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--stall-timeout', type=float,
                        default=float(os.environ.get('ASTRA_SUPERVISOR_STALL_TIMEOUT', '900')),
                        help='Seconds without a response and supervisor event progress before replacement')
    args = parser.parse_args()
    if args.stall_timeout <= 0:
        parser.error('--stall-timeout must be positive')
    batch = args.session.resolve()
    lock = (batch / 'supervisor_restart.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0
    executable = supervisor_executable()
    active, attempts, due = {}, {}, {}
    status = dict(pid=os.getpid(), hostname=socket.gethostname(), started_at=time.time(),
                  status='watching', policy='restart_exited_or_unresponsive_supervisors',
                  stall_timeout_seconds=args.stall_timeout, workers={})
    def stopped(signum, frame):
        raise InterruptedError(signum)
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        while True:
            config = read(batch / 'parallel_manifest.json', {})
            for task, entry in list(active.items()):
                refresh(entry)
                resume_supervisor_timeout(entry['folder'], entry['state'])
                status['workers'][task] = dict(entry['state'])
                stall = stalled_request(entry, time.time(), args.stall_timeout) if entry['child'].poll() is None else None
                if stall:
                    stall['restart_attempt'] = attempts.get(task, 0) + 1
                    with (batch / 'supervisor_stalls.jsonl').open('a') as log:
                        log.write(json.dumps(stall) + '\n')
                    entry['state']['stall_diagnosis'] = stall
                    record_stall_loss(entry['folder'], stall)
                    terminate(entry, 'Supervisor made no event progress while a response was overdue')
                if entry['child'].poll() is not None:
                    release(entry)
                    del active[task]
                    due[task] = time.time() + retry_delay(attempts[task])
            terminal = config.get('status') in POOL_TERMINAL
            if terminal:
                for task, entry in list(active.items()):
                    schedule = read(entry['folder'] / 'task_scheduler_status.json', {})
                    if not schedule.get('all_layouts_settled'):
                        stop_child(entry['child'])
                        release(entry)
                        del active[task]
            for worker in config.get('workers', []):
                task, unit = worker['task'], unit_id(worker)
                folder = Path(worker['session'])
                if unit in active or terminal:
                    continue
                eligible, reason = restart_eligibility(folder)
                status['workers'][unit] = dict(task=task, worker_id=unit, status=reason,
                                               restart_attempts=attempts.get(unit, 0), next_retry_at=due.get(unit))
                if not eligible or time.time() < due.get(unit, 0):
                    continue
                try:
                    entry = launch(worker, batch, executable, 'auto_' + uuid.uuid4().hex[:12], restart=True)
                except BlockingIOError:
                    continue  # Existing manager has not released ownership yet.
                except Exception as exc:
                    status['workers'][unit]['restart_error'] = repr(exc)
                    attempts[unit] = attempts.get(unit, 0) + 1
                    due[unit] = time.time() + retry_delay(attempts[unit])
                else:
                    active[unit] = entry
                    attempts[unit] = attempts.get(unit, 0) + 1
                    with (batch / 'supervisor_restarts.jsonl').open('a') as log:
                        log.write(json.dumps(dict(at=time.time(), attempt=attempts[unit], **entry['state'])) + '\n')
            status.update(heartbeat_at=time.time(), active=list(active), restart_attempts=attempts)
            write_json(batch / 'supervisor_restart_status.json', status)
            if terminal and not active:
                break
            time.sleep(3)
    finally:
        for entry in active.values():
            stop_child(entry['child'])
            release(entry)
        status.update(status='stopped', finished_at=time.time())
        write_json(batch / 'supervisor_restart_status.json', status)
        lock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
