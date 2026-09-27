"""Run previously unstarted layouts in the existing allocation with one Codex owner."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from launch_codex_supervisors import make_prompt, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--job-id', type=int, required=True)
    p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
    batch, folder = args.batch.resolve(), args.session.resolve()
    config = json.loads((batch / 'parallel_manifest.json').read_text())
    manifest = json.loads((folder / 'run_manifest.json').read_text())
    task = manifest['initial_smoke_task']
    old = next(w for w in config['workers'] if w['task'] == task)
    original = Path(old['session'])
    task_lock = (original / 'continuation.lock').open('a+')
    fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cleanup = json.loads((original / 'blocked_worker_cleanup.json').read_text())
    if config['job_id'] != str(args.job_id) and config['job_id'] != args.job_id:
        raise ValueError('Continuation must use the original allocation')
    if old['gpu'] != args.gpu or not cleanup.get('runner_exited'):
        raise ValueError('Original worker must have released its assigned GPU')
    if (folder / 'supervisor_owner.json').exists():
        raise ValueError('Continuation already has an owner; review before relaunch')
    layouts = manifest['layout_ids']
    old_manifest = json.loads((original / 'run_manifest.json').read_text())
    if len(set(layouts)) != len(layouts) or not set(layouts) <= set(old_manifest['layout_ids']):
        raise ValueError('Only distinct originally requested layouts are permitted')
    if manifest['eval_num'] != len(layouts):
        raise ValueError('eval_num must match remaining layouts')
    for layout in layouts:
        if list((original / task / 'seed_0').glob(f'layout_{layout}_*')):
            raise ValueError(f'Layout {layout} was already started; cannot silently retry')
    links_file = batch / 'continuations.json'
    previous_links = json.loads(links_file.read_text()) if links_file.exists() else {'workers': []}
    for item in previous_links['workers']:
        if item['task'] != task:
            continue
        previous = Path(item['session'])
        previous_status = json.loads(Path(item['status_file']).read_text())
        if previous_status.get('worker_returncode') is None or previous_status.get('codex_returncode') is None:
            raise ValueError('Another continuation for this task is still active')
        for layout in layouts:
            if list((previous / task / 'seed_0').glob(f'layout_{layout}_*')):
                raise ValueError(f'Layout {layout} was already started in a continuation')
    lock = (folder / 'supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner = 'continuation_' + uuid.uuid4().hex[:12] + ':' + task
    write_json(folder / 'supervisor_owner.json', dict(owner_id=owner, assigned_at=time.time(),
               manager_pid=os.getpid(), status='assigned'))
    prompt = folder / 'supervisor.prompt.txt'
    prompt.write_text(make_prompt(dict(task=task, session=str(folder), continuation=True,
                                     layout_ids=layouts), batch, owner))
    events = folder / 'supervisor.events.jsonl'
    status_file = folder / 'continuation_status.json'
    state = dict(task=task, session=str(folder), original_session=str(original), layout_ids=layouts,
                 job_id=args.job_id, gpu=args.gpu, started_at=time.time(), worker_returncode=None,
                 codex_returncode=None, codex_thread_id=None, ready=False, events=str(events))
    if any(item['session'] == str(folder) for item in previous_links['workers']):
        raise ValueError('Continuation is already linked')
    children = []
    def stop(signum, frame):
        raise InterruptedError(f'Continuation manager signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with (folder / 'launcher.log').open('w') as worker_log, events.open('w') as codex_log:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(folder / 'source_snapshot')
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            # Isaac requires physical GPU numbering. The existing job owns all eight;
            # this overlapping step runs the single worker only on its original GPU.
            cmd = ['srun', f'--jobid={args.job_id}', '--overlap', '--ntasks=1',
                   '--cpus-per-task=8', '--gres=gpu:8',
                   '/share-2/home/vla/miniforge3/envs/fastwam-dpnew-qwen/bin/python',
                   '-m', 'astra.run', '--session', str(folder), '--gpu', str(args.gpu)]
            worker = subprocess.Popen(cmd, cwd=folder / 'source_snapshot', env=env,
                                      stdout=worker_log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(worker)
            env = os.environ.copy()
            env.update(ASTRA_SUPERVISOR_ID=owner, PYTHONDONTWRITEBYTECODE='1')
            cmd = ['codex', 'exec', '--json', '--skip-git-repo-check', '--sandbox', 'workspace-write',
                   '-C', str(folder), '-m', 'gpt-6-astra', '-c', 'model_reasoning_effort="high"',
                   '-o', str(folder / 'supervisor.final.md'), '-']
            with prompt.open() as inp:
                codex = subprocess.Popen(cmd, cwd=folder, env=env, stdin=inp,
                                         stdout=codex_log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(codex)
            state.update(worker_pid=worker.pid, codex_pid=codex.pid, status='starting')
            write_json(status_file, state)
            with (batch / 'continuations.lock').open('a+') as links_lock:
                fcntl.flock(links_lock, fcntl.LOCK_EX)
                links = json.loads(links_file.read_text()) if links_file.exists() else {'workers': []}
                links['workers'].append(dict(task=task, session=str(folder), status_file=str(status_file)))
                write_json(links_file, links)
            while True:
                state.update(worker_returncode=worker.poll(), codex_returncode=codex.poll(),
                             ready=(folder / 'supervisor_ready.json').exists(), updated_at=time.time())
                if not state['codex_thread_id']:
                    with events.open() as stream:
                        for _ in range(30):
                            line = stream.readline()
                            if not line: break
                            try: event = json.loads(line)
                            except ValueError: continue
                            if event.get('type') == 'thread.started':
                                state['codex_thread_id'] = event['thread_id']
                                break
                state['status'] = 'supervised' if state['ready'] else 'starting'
                if worker.returncode is not None and codex.returncode is not None:
                    state['status'] = 'exited_review_summaries'
                write_json(status_file, state)
                if state['status'] == 'exited_review_summaries': break
                time.sleep(3)
    except BaseException as exc:
        state.update(status='manager_error', error=repr(exc))
        raise
    finally:
        for child in children:
            if child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try: child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if children: state['worker_returncode'] = children[0].returncode
        if len(children) > 1: state['codex_returncode'] = children[1].returncode
        state['finished_at'] = time.time()
        write_json(status_file, state)
        lock.close()
        task_lock.close()


if __name__ == '__main__':
    main()
