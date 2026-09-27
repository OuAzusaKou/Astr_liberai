"""Eight isolated simulator/policy processes; all actions await the active agent."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from astra.bridge import atomic_json
from astra.config import POLICY_ENV, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=Path, required=True)
    args = parser.parse_args()
    session = args.session.resolve()
    config = json.loads((session / 'parallel_manifest.json').read_text())
    allocated = [int(v) for v in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if v]
    if (len(allocated) != 8 or set(allocated) != set(range(8))
            or int(os.environ.get('SLURM_JOB_NUM_NODES', '0')) != 1
            or os.environ.get('SLURM_JOB_NAME') != 'wangzihang_astra'):
        raise RuntimeError('Requires one full eight-GPU node allocated as wangzihang_astra')
    if len(config['workers']) != 8:
        raise ValueError('Exactly eight independently configured workers required')
    sockets, children, logs = [], [], []
    def stop(signum, frame):
        raise InterruptedError(f'Batch stopped by signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    config.update(job_id=os.environ['SLURM_JOB_ID'], status='starting', started_at=time.time(),
                  supervisor='Active Codex only; waiting workers never advance without explicit decisions')
    try:
        for worker in config['workers']:
            sock = socket.socket()
            sock.bind(('127.0.0.1', 0))
            sockets.append(sock)
            worker['port'] = sock.getsockname()[1]
        atomic_json(session / 'parallel_manifest.json', config)
        for worker, sock in zip(config['workers'], sockets):
            folder = Path(worker['session'])
            log = (folder / 'launcher.log').open('w')
            logs.append(log)
            sock.close()
            worker_env = os.environ.copy()
            worker_env['PYTHONPATH'] = str(folder / 'source_snapshot')
            child = subprocess.Popen([str(POLICY_ENV / 'bin/python'), '-m', 'astra.run',
                                      '--session', str(folder), '--gpu', str(worker['gpu']),
                                      '--port', str(worker['port'])], cwd=folder / 'source_snapshot', env=worker_env,
                                     stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
            worker['pid'] = child.pid
        config['status'] = 'running_active_supervision_required'
        atomic_json(session / 'parallel_manifest.json', config)
        while any(child.poll() is None for child in children):
            atomic_json(session / 'workers_status.json',
                        [dict(task=w['task'], session=w['session'], pid=p.pid, returncode=p.poll())
                         for w, p in zip(config['workers'], children)])
            time.sleep(2)
        failures = [w['task'] for w, child in zip(config['workers'], children) if child.returncode != 0]
        if failures:
            raise RuntimeError('Workers failed: ' + ', '.join(failures))
        config['status'] = 'workers_exited_review_results'
    except BaseException as exc:
        config.update(status='stopped_or_error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                child.kill()
        for log in logs:
            log.close()
        for sock in sockets:
            sock.close()
        config['finished_at'] = time.time()
        config['returncodes'] = [p.poll() for p in children]
        atomic_json(session / 'workers_status.json',
                    [dict(task=w['task'], session=w['session'], pid=p.pid, returncode=p.poll())
                     for w, p in zip(config['workers'], children)])
        atomic_json(session / 'parallel_manifest.json', config)


if __name__ == '__main__':
    main()
