#!/usr/bin/env python3
"""Retry one layout inside its existing batch/worker directory.

Infrastructure recovery is deliberately worker-scoped.  The batch and worker
directories remain stable; only a new attempt record under ``runs/`` is added.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
POLICY_PY = Path('/share-2/home/vla/miniforge3/envs/fastwam-dpnew-qwen/bin/python')
sys.path.insert(0, str(ROOT))
from scripts.launch_core9_supervisors import launch, read, release, stop_child, supervisor_executable


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', required=True, type=Path)
    p.add_argument('--worker', required=True, type=Path)
    p.add_argument('--gpu', required=True, type=int)
    p.add_argument('--layout-id', required=True, type=int)
    p.add_argument('--attempt-id', required=True)
    args = p.parse_args()
    batch, folder = args.batch.resolve(), args.worker.resolve()
    manifest = read(folder / 'run_manifest.json', {}) or {}
    task = manifest.get('initial_smoke_task')
    if not task or args.layout_id not in manifest.get('layout_ids', []):
        raise ValueError('Worker manifest does not contain the requested task/layout')
    config = read(batch / 'parallel_manifest.json', {}) or {}
    worker = next((w for w in config.get('workers', [])
                   if Path(w.get('session', '')).resolve() == folder), None)
    if worker is None:
        raise ValueError('Worker is not part of the batch manifest')
    lock = (folder / 'recovery_worker.lock').open('a+')
    import fcntl
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 2
    runner = entry = None
    record = dict(event='worker_recovery_started', batch=str(batch), worker=str(folder),
                  task=task, layout_id=args.layout_id, attempt_id=args.attempt_id,
                  gpu=args.gpu, started_at=time.time(), session=str(batch),
                  worker_session=str(folder))
    with (batch / 'infrastructure_recovery.jsonl').open('a') as log:
        log.write(json.dumps(record, ensure_ascii=False) + '\n')
    try:
        env = os.environ.copy()
        env.update(ASTRA_GPU_ALLOCATION_CONFIRMED='1',
                   ASTRA_ASSIGNED_GPUS=str(args.gpu), CUDA_VISIBLE_DEVICES=str(args.gpu),
                   ASTRA_JOB_ID=os.environ.get('ASTRA_JOB_ID',
                                               f'recovery_{batch.name}_{args.attempt_id}'),
                   ASTRA_RECOVERY_ATTEMPT='1',
                   PYTHONPATH=os.pathsep.join((str(ROOT), str(folder / 'source_snapshot'))),
                   PYTHONDONTWRITEBYTECODE='1')
        cmd = [str(POLICY_PY), '-m', 'astra.task_run', '--session', str(folder),
               '--batch', str(batch), '--gpu', str(args.gpu), '--external-supervisor',
               '--retry-layout', str(args.layout_id), '--attempt-id', str(args.attempt_id)]
        launcher_log = (folder / f'recovery_{args.attempt_id}.launcher.log').open('w')
        # Establish the replacement owner first.  The runner then sees a live
        # supervisor during its startup wait instead of racing the stale owner.
        worker['recovery_attempt_id'] = args.attempt_id
        entry = launch(worker, batch, supervisor_executable(),
                       'recovery_' + args.attempt_id, restart=True,
                       fresh_attempt=True, layout_override=[args.layout_id])
        record.update(supervisor_pid=entry['child'].pid,
                      supervisor_owner=entry['state'].get('owner_id'),
                      handoff_record=entry['state'].get('handoff_record'))
        runner = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                  stdout=launcher_log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        record.update(runner_pid=runner.pid, runner_command=cmd)
        with (batch / 'infrastructure_recovery.jsonl').open('a') as log:
            log.write(json.dumps(record, ensure_ascii=False) + '\n')
        while runner.poll() is None or entry['child'].poll() is None:
            time.sleep(3)
        record.update(status='finished', runner_returncode=runner.returncode,
                      supervisor_returncode=entry['child'].returncode, finished_at=time.time())
        release(entry)
        return 0
    except BaseException as exc:
        record.update(status='error', error=repr(exc), finished_at=time.time())
        if entry is not None:
            try:
                stop_child(entry['child'])
                release(entry)
            except Exception:
                pass
        if runner is not None and runner.poll() is None:
            try:
                os.killpg(runner.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise
    finally:
        with (batch / 'infrastructure_recovery.jsonl').open('a') as log:
            log.write(json.dumps(record, ensure_ascii=False) + '\n')
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
