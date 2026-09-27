"""Run one isolated, actively supervised worker inside a Slurm GPU allocation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import uuid

from astra.bridge import atomic_json
from astra.config import POLICY_ENV, PROJECT, ROOT, SIM_ENV, RUNTIME_DEPS
from astra.allocation import allocated_gpu_ids, runtime_job_id


def reserve_layout(session, manifest, layout_id, attempt_id=None):
    task = manifest['initial_smoke_task']
    expected = manifest.get('layout_ids', list(range(int(manifest.get('eval_num', 1)))))
    if layout_id not in expected:
        raise ValueError(f'Layout {layout_id} is not in the planned layouts {expected}')
    seed = manifest.get('seed', 0)
    if attempt_id is None and list((session / task / f'seed_{seed}').glob(f'layout_{layout_id}_*')):
        raise ValueError(f'Layout {layout_id} already started; explicit new attempt required')
    run_name = f'layout_{layout_id}' if attempt_id is None else f'layout_{layout_id}_attempt_{attempt_id}'
    folder = session / 'runs' / run_name
    folder.mkdir(parents=True, exist_ok=True)  # The pool may already have opened launcher.log.
    record = dict(status='initializing', task=task, seed=seed, layout_id=layout_id,
                  layout_ids=[layout_id], eval_num=1, started_at=time.time(),
                  run_id=f'{session.name}_layout{layout_id}'
                          f'{("_attempt_" + str(attempt_id)) if attempt_id is not None else ""}_'
                          f'{uuid.uuid4().hex[:8]}',
                  attempt_id=attempt_id,
                  configuration_manifest=str(session / 'run_manifest.json'))
    status_path = folder / 'run_status.json'
    # Exclusive reservation prevents simultaneous launches, including failed startups.
    with status_path.open('x') as stream:
        json.dump(record, stream)
    atomic_json(session / 'active_run.json', dict(layout_id=layout_id, run_id=record['run_id'],
                run_dir=str(folder), status_file=str(status_path)))
    return folder, status_path, record


def completed_layout_result(session, task, seed, layout_id):
    paths = list((session / task / f'seed_{seed}').glob(f'layout_{layout_id}_*/summary.json'))
    if len(paths) > 1:
        raise RuntimeError(f'Multiple summaries for layout {layout_id}; refusing ambiguous completion')
    status_path = session / 'controller_status.json'
    if not paths or not status_path.exists():
        return None
    status = json.loads(status_path.read_text())
    if status.get('status') not in {'episode_finished', 'blocked_or_error'}:
        return None
    if Path(status.get('folder', '')).resolve() != paths[0].parent.resolve():
        return None
    result = json.loads(paths[0].read_text())
    if result.get('status') not in {'success', 'failed', 'blocked', 'infrastructure_error'}:
        raise RuntimeError('Unrecognized episode summary status')
    if result['status'] == 'success' and not (result.get('official_success') is True and result.get('official_terminal') is True):
        raise RuntimeError('Success summary lacks official terminal success')
    return dict(summary_path=str(paths[0]), summary=result)


def terminate_process_group(child, role, term_timeout=15, kill_timeout=15):
    report = dict(role=role, pid=child.pid, forced=False, signals=[])
    if child.poll() is None:
        report['forced'] = True
        for signum, timeout in [(signal.SIGTERM, term_timeout), (signal.SIGKILL, kill_timeout)]:
            try:
                os.killpg(child.pid, signum)
                report['signals'].append(signal.Signals(signum).name)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=timeout)
                break
            except subprocess.TimeoutExpired:
                if signum == signal.SIGKILL:
                    raise RuntimeError(f'{role} process group did not exit after SIGKILL')
    report['returncode'] = child.poll()
    return report


def wait_layout(simulator, server, session, task, seed, layout_id, on_terminal=None,
                teardown_grace_s=30):
    terminal, deadline = None, None
    while True:
        if terminal is None:
            terminal = completed_layout_result(session, task, seed, layout_id)
            if terminal is not None:
                deadline = time.monotonic() + teardown_grace_s
                if on_terminal:
                    on_terminal(terminal)
        rc = simulator.poll()
        if rc is not None:
            # A terminal artifact can have been written between the first read and poll.
            terminal = terminal or completed_layout_result(session, task, seed, layout_id)
            if terminal is None:
                raise RuntimeError(f'Simulator exited {rc} without a finalized layout result')
            return dict(**terminal, native_returncode=rc, teardown=dict(forced=False, signals=[]))
        if terminal is not None:
            if time.monotonic() >= deadline:
                cleanup = terminate_process_group(simulator, 'simulator')
                return dict(**terminal, native_returncode=cleanup['returncode'], teardown=cleanup)
        elif server.poll() is not None:
            raise RuntimeError(f'Policy server exited {server.returncode} before a layout result')
        time.sleep(0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', required=True, type=Path)
    p.add_argument('--gpu', type=int)
    p.add_argument('--port', type=int)
    p.add_argument('--layout-id', type=int)
    p.add_argument('--attempt-id', help='Explicit retry identifier for an existing layout')
    args = p.parse_args()
    session = args.session.resolve()
    runtime = session / 'runtime/RoboDojo'
    manifest = json.loads((session / 'run_manifest.json').read_text())
    job_id = runtime_job_id()
    allocation = allocated_gpu_ids()
    layout_lock = None
    if args.layout_id is None:
        run_folder, status_path, record = session, session / 'run_manifest.json', manifest
    else:
        layout_lock = (session / 'layout_runner.lock').open('a+')
        try:
            fcntl.flock(layout_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run_folder, status_path, record = reserve_layout(
                session, manifest, args.layout_id, args.attempt_id)
        except BaseException:
            layout_lock.close()
            raise
    try:
        run_worker(args, session, runtime, manifest, run_folder, status_path, record)
    except BaseException as exc:
        record.update(status='runtime_error', error=f'{type(exc).__name__}: {exc}', finished_at=time.time())
        atomic_json(status_path, record)
        raise
    finally:
        if layout_lock is not None:
            layout_lock.close()


def run_worker(args, session, runtime, manifest, run_folder, status_path, record):
    allocation = allocated_gpu_ids()
    job_id = runtime_job_id()
    task = manifest['initial_smoke_task']
    eval_num = 1 if args.layout_id is not None else int(manifest.get('eval_num', 1))
    layouts = [args.layout_id] if args.layout_id is not None else manifest.get('layout_ids', list(range(eval_num)))
    run_id = record.get('run_id', session.name) if args.layout_id is not None else session.name
    demo = json.loads((session / 'demos' / task / 'demo_context.json').read_text())
    waived = demo.get('status') == 'waived_by_user' and task in manifest.get('authorized_demo_waivers', [])
    if demo.get('status') != 'reviewed' and not waived:
        raise RuntimeError('Review the training demonstration before running')
    if args.port is None:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
    else:
        port = args.port
    base = os.environ.copy()
    base.update(ASTRA_SESSION=str(session), PYTHONDONTWRITEBYTECODE='1',
                TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0', OMNI_KIT_ACCEPT_EULA='YES',
                SETUPTOOLS_SCM_PRETEND_VERSION='0.0.0', PYTHONUNBUFFERED='1',
                ROBODOJO_RUN_ID=run_id, EVAL_NUM=str(eval_num),
                ROBODOJO_LAYOUT_IDS=','.join(map(str, layouts)),
                ROBODOJO_WS_REQUEST_TIMEOUT_S='900', ROBODOJO_MAX_BASH_RETRIES='1',
                TOKENIZERS_PARALLELISM='false', IMAGEWAM_SKIP_ACTION_INIT='1',
                ROBODOJO_SAVE_VIDEO='1', EVAL_RUN_NAME='astra_liberai_test')
    for key, leaf in [('TMPDIR', 'tmp'), ('HF_HOME', 'cache/huggingface'),
                      ('XDG_CACHE_HOME', 'cache/xdg'), ('TORCHINDUCTOR_CACHE_DIR', 'cache/inductor'),
                      ('TRITON_CACHE_DIR', 'cache/triton'), ('CUDA_CACHE_PATH', 'cache/cuda')]:
        dest = session / leaf
        dest.mkdir(parents=True, exist_ok=True)
        base[key] = str(dest)
    model_cfg = session / 'model.yaml'
    import yaml
    cfg = yaml.safe_load(model_cfg.read_text())
    from astra.conditioning import mode_for
    if mode_for(manifest) != mode_for(cfg):
        raise ValueError('Manifest and model conditioning modes differ; refusing launch')
    base['ASTRA_CONDITIONING_AUDIT_LOG'] = str(run_folder / 'model_conditioning.jsonl')
    base['PYTHONPATH'] = ':'.join(map(str, [PROJECT / 'wam/model', PROJECT,
                                runtime, runtime / 'XPolicyLab', ROOT,
                                Path(cfg['flux2_src_path']), Path(cfg['flux2_src_path']) / 'src', RUNTIME_DEPS]))
    policy = base.copy()
    if args.gpu is not None:
        allocated = allocation
        if args.gpu not in allocated:
            raise RuntimeError('Worker GPU is not allocated to this job')
        policy['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        policy['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    policy['PATH'] = str(POLICY_ENV / 'bin') + ':' + base['PATH']
    policy['CONDA_PREFIX'] = str(POLICY_ENV)
    # The policy uses CUDA 13; do not mix its runtime libraries into Isaac Sim.
    import site
    nvrtc = next((Path(s) / 'nvidia/cu13/lib' for s in site.getsitepackages()
                  if (Path(s) / 'nvidia/cu13/lib/libnvrtc.so.13').exists()), None)
    if nvrtc is None:
        raise RuntimeError('CUDA 13 NVRTC library not found in policy environment')
    policy['LD_LIBRARY_PATH'] = str(nvrtc) + ':' + str(POLICY_ENV / 'lib') + ':' + base.get('LD_LIBRARY_PATH', '')
    sim = base.copy()
    sim_gpu = 0
    if args.gpu is not None:
        if len(allocated) == 8:
            sim.pop('CUDA_VISIBLE_DEVICES', None)
            sim['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            sim_gpu = args.gpu
        elif len(allocated) == 1:
            sim_gpu = 0
            if os.environ.get('SLURM_GPUS_ON_NODE') == '1':
                sim.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            # The pool may expose several physical GPUs to its parent, but
            # Isaac Sim accepts only one visible GPU per worker. Restrict this
            # child to its assigned physical GPU and use worker-local index 0.
            sim['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
            sim['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            sim_gpu = 0
    sources = [runtime / 'third_party/IsaacLab/source' / name for name in
               ('isaaclab', 'isaaclab_assets', 'isaaclab_tasks', 'isaaclab_mimic', 'isaaclab_rl', 'isaaclab_contrib')]
    sim['PYTHONPATH'] = ':'.join(map(str, [runtime, runtime / 'XPolicyLab', ROOT,
                              *[v for v in sources if v.is_dir()], runtime / 'third_party/curobo']))
    sim['PATH'] = str(session / 'bin') + ':' + str(SIM_ENV / 'bin') + ':' + base['PATH']
    sim['CONDA_PREFIX'] = str(SIM_ENV)
    sim['CUDA_HOME'] = str(SIM_ENV)
    sim['LD_LIBRARY_PATH'] = str(SIM_ENV / 'lib') + ':/usr/lib/x86_64-linux-gnu:' + base.get('LD_LIBRARY_PATH', '')
    # Reference RoboDojo launcher requires Vulkan to see the isolated single GPU.
    if os.environ.get('SLURM_GPUS_ON_NODE') == '1':
        sim.pop('CUDA_VISIBLE_DEVICES', None)
    bootstrap = ROOT / 'astra/bootstrap.py'
    server_cmd = [str(POLICY_ENV / 'bin/python'), str(bootstrap), str(runtime / 'XPolicyLab/setup_policy_server.py'),
                  '--config_path', str(model_cfg), '--overrides', f'port={port}', 'host=127.0.0.1',
                  f'task_name={task}', 'bench_name=RoboDojo', 'ckpt_name=step_072000']
    sim_cmd = [str(SIM_ENV / 'bin/python'), str(bootstrap), str(runtime / 'src/eval_client/main.py'),
               '--task_name', task, '--env_cfg_type', 'arx_x5', '--num_envs', '1',
               '--device_id', str(sim_gpu), '--device', f'cuda:{sim_gpu}',
               '--policy_name', 'AstraImageWAM', '--port', str(port),
               '--host', '127.0.0.1', '--protocol', 'ws', '--additional_info', run_id,
               '--seed', '0', '--headless', '--enable_cameras',
               '--kit_args', '--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera --/renderer/multiGpu/enabled=false']
    record.update(status='starting_layout' if args.layout_id is not None else 'starting_smoke', job_id=job_id,
                    hostname=socket.gethostname(), allocated_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    job_name=os.environ.get('SLURM_JOB_NAME'), server_command=server_cmd, simulator_command=sim_cmd,
                    model_config=cfg, controller_timeout_s=900, assigned_gpu=args.gpu,
                    policy_cuda_visible_devices=policy.get('CUDA_VISIBLE_DEVICES'), simulator_device=sim_gpu)
    atomic_json(status_path, record)
    children = []
    def stop(signum, frame):
        raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with (run_folder / 'policy.log').open('w') as policy_log, (run_folder / 'simulator.log').open('w') as sim_log:
            server = subprocess.Popen(server_cmd, cwd=runtime, env=policy, stdout=policy_log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(server)
            deadline = time.monotonic() + 1500
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Policy server exited {server.returncode}; see policy.log')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=1):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Policy server startup exceeded 1500 seconds')
                    time.sleep(2)
            record['status'] = 'running_layout' if args.layout_id is not None else 'running_smoke'
            record['policy_pid'] = server.pid
            atomic_json(status_path, record)
            simulator = subprocess.Popen(sim_cmd, cwd=runtime, env=sim, stdout=sim_log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(simulator)
            record['simulator_pid'] = simulator.pid
            atomic_json(status_path, record)
            if args.layout_id is not None:
                def terminal_found(terminal):
                    record.update(status='terminal_waiting_teardown', result=terminal,
                                  terminal_observed_at=time.time(), teardown_grace_s=30)
                    atomic_json(status_path, record)
                result = wait_layout(simulator, server, session, task, manifest.get('seed', 0),
                                     args.layout_id, on_terminal=terminal_found)
                record.update(status='layout_finished', result=result,
                              outcome=result['summary']['status'],
                              official_success=result['summary'].get('official_success'),
                              episode_summaries=[result['summary_path']])
                return
            rc = simulator.wait()
            if rc:
                raise RuntimeError(f'Simulator exited {rc}; see simulator.log')
            summaries = list((session / task).glob('seed_*/*/summary.json'))
            if not summaries:
                raise RuntimeError('Simulator returned without an audited rollout')
            record['status'] = 'smoke_finished_runtime_review_required'
            record['episode_summaries'] = [str(v) for v in summaries]
    except BaseException as exc:
        record.update(status='runtime_error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        cleanup, cleanup_errors = [], []
        for child in reversed(children):
            try:
                cleanup.append(terminate_process_group(child, 'policy' if child is children[0] else 'simulator'))
            except Exception as exc:
                cleanup_errors.append(str(exc))
        record.update(cleanup=cleanup, cleanup_errors=cleanup_errors, finished_at=time.time())
        atomic_json(status_path, record)
        if cleanup_errors:
            raise RuntimeError('Child cleanup incomplete: ' + '; '.join(cleanup_errors))


if __name__ == '__main__':
    main()
