#!/usr/bin/env python3
"""Detect known Isaac Sim infrastructure failures and launch explicit attempts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
POLICY_PY = Path('/share-2/home/vla/miniforge3/envs/fastwam-dpnew-qwen/bin/python')
RECOVER_WORKER = ROOT / 'scripts/recover_worker.py'

ISAAC_FAILURE_MARKERS = (
    'parallel simulator requires either one or eight visible gpus',
    'simulator exited 0 without a finalized layout result',
    'unsupported operand type(s) for -: \'nonetype\' and \'nonetype\'',
    'annotator',
    'isaacsim',
    'omni.replicator',
    'carb.recursive unloadallplugins',
    'segmentation fault',
    'cuda error',
    'cuda out of memory',
    'xid',
)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def layout_failure(worker):
    schedule = read(Path(worker['session']) / 'task_scheduler_status.json', {}) or {}
    if schedule.get('status') not in {
            'infrastructure_error', 'supervisor_summary_incomplete',
            'cleanup_unverified', 'scheduler_error'}:
        return None
    results = schedule.get('layout_results', [])
    for result in results:
        if result.get('summaries'):
            continue
        layout = result.get('layout_id')
        evidence = []
        folder = Path(worker['session'])
        for path in [folder / f'layout_{layout}.launcher.log',
                     folder / f'layout_{layout}.launcher.log',
                     folder / 'runs' / f'layout_{layout}' / 'simulator.log',
                     folder / 'runs' / f'layout_{layout}' / 'policy.log']:
            if path.is_file():
                evidence.append(path)
        text = '\n'.join(path.read_text(errors='replace')[-200_000:] for path in evidence)
        lower = text.lower()
        markers = [marker for marker in ISAAC_FAILURE_MARKERS if marker in lower]
        if markers:
            return dict(task=worker['task'], worker_id=worker.get('worker_id'),
                        layout_id=layout, markers=markers,
                        evidence=[str(path) for path in evidence],
                        runner_returncode=result.get('runner_returncode'),
                        detected_at=time.time())
    return None


def active_gpus(config):
    used = set()
    for worker in config.get('workers', []):
        schedule = read(Path(worker['session']) / 'task_scheduler_status.json', {}) or {}
        if schedule.get('status') in {'running_layout', 'starting_layout', 'paused_for_main',
                                      'waiting_for_supervisor', 'awaiting_supervisor_summary'}:
            gpu = schedule.get('gpu', worker.get('gpu'))
            if isinstance(gpu, int):
                used.add(gpu)
    return used


def assigned_gpus():
    raw = os.environ.get('ASTRA_ASSIGNED_GPUS') or os.environ.get('CUDA_VISIBLE_DEVICES', '')
    return [int(x) for x in raw.split(',') if x.strip().isdigit()]


def recovery_session_path(recovery_root, attempt):
    """Return the immutable batch directory used by all worker retries."""
    return Path(recovery_root).resolve()


def next_recovery_attempt(recovery_root, attempt):
    """Skip attempt numbers left by an interrupted watchdog or old state."""
    return attempt


def launch_recovery(batch, failures, gpus, attempt, recovery_root=None):
    # Recovery is worker-scoped.  The batch and worker paths are immutable and
    # remain the source of truth for all attempts.
    root = Path(batch).resolve()
    recovery_root = Path(recovery_root or root).resolve()
    manifest = read(root / 'parallel_manifest.json', {}) or {}
    workers = manifest.get('workers', [])
    session = recovery_session_path(recovery_root, attempt)
    prepare_log = root / 'infrastructure_watchdog.log'
    with prepare_log.open('a') as log:
        launched = []
        for index, failure in enumerate(failures):
            worker = next((item for item in workers
                           if item.get('task') == failure.get('task')
                           or item.get('worker_id') == failure.get('worker_id')), None)
            if worker is None:
                log.write(json.dumps(dict(event='recovery_worker_missing', failure=failure,
                                          at=time.time()), ensure_ascii=False) + '\n')
                continue
            worker_session = Path(worker['session']).resolve()
            gpu = gpus[index]
            attempt_id = f'infra_{attempt:02d}_{failure.get("layout_id")}'
            env = os.environ.copy()
            env.update(ASTRA_GPU_ALLOCATION_CONFIRMED='1', ASTRA_ASSIGNED_GPUS=str(gpu),
                       CUDA_VISIBLE_DEVICES=str(gpu),
                       ASTRA_JOB_ID=f'astra_infra_recovery_{root.name}_{attempt:02d}',
                       ASTRA_INFRA_WATCHDOG_ENABLED='0',
                       CODEX_HOME=env.get('CODEX_HOME', '/share-2/home/vla/.codex-wangzihang'))
            command = [str(POLICY_PY), str(RECOVER_WORKER), '--batch', str(root),
                       '--worker', str(worker_session), '--gpu', str(gpu),
                       '--layout-id', str(failure['layout_id']), '--attempt-id', attempt_id]
            output = worker_session / f'infrastructure_recovery_{attempt_id}.launcher.log'
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=output.open('w'), stderr=subprocess.STDOUT,
                                     start_new_session=True)
            launched.append(dict(task=failure.get('task'), worker_id=failure.get('worker_id'),
                                 layout_id=failure.get('layout_id'), gpu=gpu,
                                 worker_session=str(worker_session), attempt_id=attempt_id,
                                 command=command, launcher_pid=child.pid))
        if not launched:
            return None
        record = dict(event='recovery_started', source_batch=str(root),
                      recovery_batch=str(root), failures=failures, gpus=gpus,
                      recovery_root=str(recovery_root), reused_worker_sessions=True,
                      workers=launched, launcher_pid=launched[0]['launcher_pid'], at=time.time())
        log.write(json.dumps(record, ensure_ascii=False) + '\n')
        return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--poll-seconds', type=float, default=5)
    args = parser.parse_args()
    batch = args.session.resolve()
    state_path = batch / 'infrastructure_watchdog_status.json'
    lock_path = batch / 'infrastructure_watchdog.lock'
    lock = lock_path.open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # A single parent watchdog owns a batch.  A duplicate launcher must
        # leave the existing manager and its recovery chain untouched.
        lock.close()
        return
    state = read(state_path, {}) or dict(status='watching', attempts={}, recoveries=[],
                                      watched_batches=[str(batch)], handled_failures=[])
    state.setdefault('watched_batches', [str(batch)])
    state.setdefault('handled_failures', [])
    # All retries belong to the initial batch.  Older state files predate this
    # field; their top-level session is the correct immutable root.
    state.setdefault('recovery_root', str(batch))
    for record in state.get('recoveries', []):
        recovery_batch = record.get('recovery_batch')
        if recovery_batch and recovery_batch not in state['watched_batches']:
            state['watched_batches'].append(recovery_batch)
        source_batch = record.get('source_batch')
        for failure in record.get('failures', []):
            if source_batch and failure.get('worker_id') is not None:
                key = f'{source_batch}:{failure["worker_id"]}:{failure.get("layout_id")}'
                if key not in state['handled_failures']:
                    state['handled_failures'].append(key)
    state.update(status='watching', pid=os.getpid(), heartbeat_at=time.time())
    write(state_path, state)
    try:
        while True:
            failures = []
            active_configs = []
            for watched in list(dict.fromkeys(state['watched_batches'])):
                watched_path = Path(watched)
                config = read(watched_path / 'parallel_manifest.json', {}) or {}
                active_configs.append((watched_path, config))
                for worker in config.get('workers', []):
                    key = worker.get('worker_id', worker.get('task'))
                    failure = layout_failure(worker)
                    attempt_no = state['attempts'].get(key, 0)
                    failure_key = (f'{watched}:{key}:{failure.get("layout_id")}:{attempt_no}'
                                   if failure else None)
                    if failure and failure_key not in state['handled_failures']:
                        failure['_watch_batch'] = str(watched_path)
                        failure['_failure_key'] = failure_key
                        failures.append((key, failure))
            if failures:
                used = set()
                for watched_path, config in active_configs:
                    used.update(active_gpus(config))
                available = [gpu for gpu in assigned_gpus() if gpu not in used]
                source_batch = failures[0][1]['_watch_batch'] if failures else None
                chosen = [item for item in failures if item[1]['_watch_batch'] == source_batch][:len(available)]
                if chosen:
                    attempt = next_recovery_attempt(
                        state['recovery_root'], max(state['attempts'].values() or [0]) + 1)
                    record = launch_recovery(Path(source_batch), [item[1] for item in chosen],
                                             available[:len(chosen)], attempt,
                                             recovery_root=state['recovery_root'])
                    if record:
                        for key, failure in chosen:
                            state['attempts'][key] = state['attempts'].get(key, 0) + 1
                            state['handled_failures'].append(failure['_failure_key'])
                        state['recoveries'].append(record)
                        if record['recovery_batch'] not in state['watched_batches']:
                            state['watched_batches'].append(record['recovery_batch'])
                        write(state_path, state)
            terminal = all(config.get('status') in {
                'complete_or_blocked_review_results', 'stopped_or_error',
                'cancelled', 'failed', 'interrupted'}
                           for _, config in active_configs)
            if terminal and not failures:
                break
            state.update(heartbeat_at=time.time())
            write(state_path, state)
            time.sleep(args.poll_seconds)
    finally:
        state.update(status='stopped', finished_at=time.time())
        write(state_path, state)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
