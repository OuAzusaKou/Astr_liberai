"""Run a FIFO queue of one-layout workers and refill released GPU slots."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from astra.bridge import atomic_json
from astra.config import POLICY_ENV
from astra.allocation import allocated_gpu_ids, runtime_job_id


def unit_id(worker):
    return worker.get('worker_id', worker['task'])


def allocate_pending(workers, active, settled, gpu_ids=None):
    """Assign only free GPU slots, retaining blocked units in the overall plan."""
    used = {entry['gpu'] for entry in active.values()}
    free = [gpu for gpu in (gpu_ids if gpu_ids is not None else range(8)) if gpu not in used]
    pending = [w for w in workers if w.get('runnable', True)
               and unit_id(w) not in active and unit_id(w) not in settled]
    return [(worker, gpu) for worker, gpu in zip(pending, free)]


def validate_worker_plan(config):
    workers = config.get('workers', [])
    keys = [unit_id(worker) for worker in workers]
    if len(set(keys)) != len(keys):
        raise ValueError('Worker IDs must be unique in a batch')
    if config.get('scheduling_unit') == 'one_layout_per_worker':
        pairs = []
        for worker in workers:
            layouts = worker.get('layout_ids', [])
            if len(layouts) != 1:
                raise ValueError(f"{unit_id(worker)} must contain exactly one planned layout")
            pairs.append((worker['task'], layouts[0]))
        if len(set(pairs)) != len(pairs):
            raise ValueError('A task/layout pair may only be assigned to one worker')


def gated_workers(batch, config):
    """Hold concurrency at the designated validation task until main reviews evidence."""
    gate = config.get('startup_validation_gate')
    if not gate:
        return config['workers']
    try:
        approval = json.loads((Path(batch) / 'startup_validation_release.json').read_text())
    except (OSError, ValueError):
        approval = {}
    if (approval.get('status') == 'verified' and approval.get('authorized_by') == 'main_session'
            and approval.get('task') == gate['task'] and approval.get('evidence')):
        return config['workers']
    return [worker for worker in config['workers'] if worker['task'] == gate['task']][:1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', type=Path, required=True)
    args = p.parse_args()
    batch = args.session.resolve()
    config = json.loads((batch / 'parallel_manifest.json').read_text())
    allocated = allocated_gpu_ids()
    job_id = runtime_job_id()
    validate_worker_plan(config)
    if config.get('startup_validation_gate') and not any(
            w['task'] == config['startup_validation_gate']['task'] and w.get('runnable', True)
            for w in config['workers']):
        raise ValueError('Startup validation task must be runnable')
    max_workers = config.get('max_workers', 8)
    if not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError('max_workers in parallel_manifest.json must be positive')
    capacity = min(len(allocated), max_workers)
    worker_gpu_ids = allocated[:capacity]
    config.update(job_id=job_id, status='running', started_at=time.time(), concurrency=capacity,
                  max_workers=max_workers, requested_gpus=max_workers, allocated_gpus=allocated,
                  hostname=socket.gethostname(),
                  execution_mode='slurm' if os.environ.get('SLURM_JOB_ID') else 'bash_direct')
    active, settled, records, supervisor_rows = {}, set(), {}, {}
    for worker in config['workers']:
        key = unit_id(worker)
        runnable = worker.get('runnable', True)
        records[key] = dict(task=worker['task'], worker_id=key, layout_ids=worker.get('layout_ids'),
            session=worker['session'], pid=None,
            gpu=None, returncode=None, status='queued' if runnable else worker.get('blocked_reason', 'blocked_missing_demo'),
            finalized=not runnable)
        if not runnable:
            settled.add(key)
    def stop(signum, frame):
        raise InterruptedError(f'Core9 pool received signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_monitor = 0
    def save():
        atomic_json(batch / 'parallel_manifest.json', config)
        atomic_json(batch / 'workers_status.json', list(records.values()))
        atomic_json(batch / 'codex_supervision/status.json', dict(
            supervision_session_id=batch.name, model='gpt-6-astra', reasoning_effort='high',
            status='managed_by_task_pool', execution_mode=config['execution_mode'],
            updated_at=time.time(), workers=list(supervisor_rows.values())))
        atomic_json(batch / 'scheduler_status.json', dict(status=config['status'], updated_at=time.time(),
            active=list(active), settled=sorted(settled),
            queued=[unit_id(w) for w in config['workers'] if unit_id(w) not in active and unit_id(w) not in settled],
            max_simultaneous_tasks=capacity))
    try:
        while len(settled) < len(config['workers']):
            for worker, gpu in allocate_pending(gated_workers(batch, config), active, settled, worker_gpu_ids):
                key = unit_id(worker)
                folder = Path(worker['session'])
                worker['gpu'] = gpu
                log = (folder / 'task_launcher.log').open('x')
                env = os.environ.copy()
                env['PYTHONPATH'] = str(folder / 'source_snapshot')
                child = subprocess.Popen([str(POLICY_ENV / 'bin/python'), '-m', 'astra.task_run',
                    '--session', str(folder), '--batch', str(batch), '--gpu', str(gpu), '--external-supervisor'],
                    cwd=folder / 'source_snapshot', env=env, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                active[key] = dict(child=child, log=log, gpu=gpu, session=folder, task=worker['task'])
                records[key].update(pid=child.pid, gpu=gpu, status='running', started_at=time.time())
            for key, running in list(active.items()):
                path = running['session'] / 'supervisor_status.json'
                if path.exists():
                    try: supervisor_rows[key] = json.loads(path.read_text())
                    except json.JSONDecodeError: pass
                rc = running['child'].poll()
                if rc is not None:
                    task_status_file = running['session'] / 'task_scheduler_status.json'
                    task_status = json.loads(task_status_file.read_text()) if task_status_file.exists() else {}
                    if task_status.get('gpu_quarantined') or rc < 0:
                        raise RuntimeError(f'GPU {running["gpu"]} cleanup unverified for {running["task"]}; stopping pool before slot reuse')
                    running['log'].close()
                    task_failed = rc != 0 or task_status.get('status') in {
                        'infrastructure_error', 'supervisor_summary_incomplete',
                        'cleanup_unverified', 'scheduler_error', 'stopped_or_error'}
                    records[key].update(returncode=rc, status='task_error' if task_failed else 'settled',
                                         finalized=True, finished_at=time.time())
                    settled.add(key)
                    del active[key]
            save()
            if time.monotonic() - last_monitor >= 10:
                from scripts.monitor_parallel import snapshot, save as save_monitor
                save_monitor(batch, snapshot(batch))
                last_monitor = time.monotonic()
            if len(settled) < len(config['workers']):
                time.sleep(2)
        failed_tasks = [key for key, record in records.items()
                        if record['returncode'] is not None and record['returncode'] != 0]
        if failed_tasks:
            config.update(status='stopped_or_error',
                          error=f'Worker process errors: {", ".join(failed_tasks)}')
        else:
            config['status'] = 'complete_or_blocked_review_results'
    except BaseException as exc:
        config.update(status='stopped_or_error', error=repr(exc))
        raise
    finally:
        for running in active.values():
            if running['child'].poll() is None:
                os.killpg(running['child'].pid, signal.SIGTERM)
        for key, running in active.items():
            try: running['child'].wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(running['child'].pid, signal.SIGKILL)
                running['child'].wait()
            running['log'].close()
            records[key].update(returncode=running['child'].returncode, status='pool_stopped', finalized=True)
        active.clear()
        config['finished_at'] = time.time()
        save()
        from scripts.monitor_parallel import snapshot, save as save_monitor
        save_monitor(batch, snapshot(batch))
    return 1 if failed_tasks else 0


if __name__ == '__main__':
    raise SystemExit(main())
