#!/usr/bin/env python3
"""Own Codex supervision on the control host; never submit robot actions."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from astra.task_run import make_prompt, read_thread_id, write_json
from astra.config import POLICY_ENV

POOL_TERMINAL = {'complete_or_blocked_review_results', 'stopped_or_error', 'cancelled', 'failed', 'interrupted', 'finished'}
JOB_TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
WAITING_FOR_SUPERVISOR = {'waiting_for_supervisor', 'waiting_for_supervisor_ready'}


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def unit_id(worker):
    return worker.get('worker_id', worker['task'])


def supervisor_executable():
    return 'wangzihang_codex'


def supervisor_command(executable, args):
    if executable == 'wangzihang_codex':
        # Resolve the user's configured command in an interactive shell.  Do
        # not substitute the wrapper script or vendor ``codex`` executable:
        # the alias owns authentication, routing, and CODEX_HOME.
        command = shlex.join(['wangzihang_codex', *args])
        script = f"eval {shlex.quote(command)}"
        return ['bash', '-ic', script]
    # Explicit injected executables are used by isolated process/lock tests.
    return [executable, *args]


def preflight(batch):
    """Render every runnable prompt before claiming ownership or waiting for GPUs."""
    batch = Path(batch).resolve()
    config = read(batch / 'parallel_manifest.json')
    if config is None:
        raise ValueError('Prepared batch manifest is required')
    tasks = []
    for worker in config.get('workers', []):
        if not worker.get('runnable', True):
            continue
        folder = Path(worker['session'])
        manifest = read(folder / 'run_manifest.json')
        if manifest is None:
            raise ValueError(f'Missing worker manifest: {folder}')
        if (config.get('scheduling_unit') == 'one_layout_per_worker'
                and manifest.get('layout_ids') != worker.get('layout_ids')):
            raise ValueError(f"{unit_id(worker)}: worker and runtime layouts differ")
        make_prompt(folder, batch, worker['task'], manifest['layout_ids'], 'preflight', manifest)
        tasks.append(worker['task'])
    return dict(status='preflight_passed', python=sys.executable, tasks=tasks)


def plan_supervisors(workers, schedules, started, active, pool_terminal=False, job_terminal=False, capacity=8):
    """Pure scheduling: only allocated waiting workers can acquire a Codex owner."""
    by_task = {unit_id(worker): worker for worker in workers}
    stop = {}
    for task in active:
        worker, schedule = by_task.get(task, {}), schedules.get(task, {})
        if worker.get('returncode') not in (None, 0) or worker.get('status') in {'task_error', 'pool_stopped'}:
            stop[task] = 'Task manager exited with an error or was stopped'
        elif (pool_terminal or job_terminal or worker.get('finalized') or worker.get('returncode') is not None) and not schedule.get('all_layouts_settled'):
            stop[task] = 'Allocation/task ended before all layouts settled'
    launch = []
    if not pool_terminal and not job_terminal:
        slots = max(0, capacity-len(active))
        launch = [unit_id(worker) for worker in workers
                  if unit_id(worker) not in started and unit_id(worker) not in active
                  and worker.get('status') == 'running' and worker.get('returncode') is None
                  and schedules.get(unit_id(worker), {}).get('status') in WAITING_FOR_SUPERVISOR][:slots]
    return launch, stop


def finished_waiting(config, job_terminal, active):
    return not active and (config.get('status') in POOL_TERMINAL or job_terminal)


def slurm_state(job_id):
    if not job_id:
        return None
    try:
        result = subprocess.run(['squeue', '-h', '-j', str(job_id), '-o', '%T'], capture_output=True, text=True, timeout=10)
        if result.returncode:
            return None
        states = [line.strip().split()[0] for line in result.stdout.splitlines() if line.strip()]
        if states:
            return states[0]
        # Empty squeue alone is not proof that a newly submitted job has finished.
        result = subprocess.run(['sacct', '-X', '-n', '-j', str(job_id), '--format=State', '--parsable2'],
                                capture_output=True, text=True, timeout=10)
        if not result.returncode:
            for line in result.stdout.splitlines():
                state = line.split('|')[0].strip().split(' ')[0].rstrip('+')
                if state:
                    return state
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def refresh(entry):
    state, child, folder = entry['state'], entry['child'], entry['folder']
    state.update(returncode=child.poll(), updated_at=time.time(), heartbeat_at=time.time())
    ready = read(folder / 'supervisor_ready.json', {})
    state['ready'] = bool(ready and ready.get('owner_id') == state['owner_id'])
    if not state['thread_id']:
        state['thread_id'] = read_thread_id(Path(state['events']))
    if state['returncode'] is not None:
        state.update(status='exited', finished_at=state.get('finished_at', time.time()))
    else:
        state['status'] = 'supervising' if state['ready'] else 'starting'
    write_json(folder / 'supervisor_status.json', state)


def stop_child(child):
    if child.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            if sig == signal.SIGKILL:
                raise RuntimeError('Codex child did not exit after SIGKILL')


def terminate(entry, reason):
    entry['state']['stop_reason'] = reason
    stop_child(entry['child'])
    refresh(entry)


def release(entry):
    if entry['child'].poll() is None:
        raise RuntimeError('Cannot release supervisor lock while owned child is alive')
    refresh(entry)
    owner = read(entry['folder'] / 'supervisor_owner.json', {})
    if owner.get('owner_id') == entry['state']['owner_id']:
        owner.update(status='exited', finished_at=time.time(), returncode=entry['state']['returncode'])
        write_json(entry['folder'] / 'supervisor_owner.json', owner)
    entry['log'].close()
    entry['lock'].close()


def launch(worker, batch, executable, manager_id, restart=False, fresh_attempt=False,
           layout_override=None):
    folder, task = Path(worker['session']).resolve(), worker['task']
    unit = unit_id(worker)
    lock = (folder / 'supervisor.lock').open('a+')
    log = child = None
    owned = False
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archive = None
        prior = read(folder / 'supervisor_status.json', {})
        if restart:
            from scripts.restart_supervisors import restart_eligibility
            if not fresh_attempt:
                eligible, reason = restart_eligibility(folder)
                if not eligible:
                    raise RuntimeError(reason)
            archive = folder / 'supervision_handoffs' / ('auto_' + uuid.uuid4().hex[:12])
            archive.mkdir(parents=True)
            for name in ('supervisor_owner.json', 'supervisor_status.json', 'supervisor_ready.json',
                         'supervisor.prompt.txt', 'supervisor.final.md'):
                if (folder / name).exists():
                    shutil.copy2(folder / name, archive / ('prior_' + name))
        elif (folder / 'supervisor_owner.json').exists():
            raise RuntimeError(f'{task}: prior ownership exists; explicit handoff required')
        owner_id = f'{manager_id}:{unit}'
        hostname = socket.gethostname()
        owner = dict(owner_id=owner_id, manager_pid=os.getpid(), hostname=hostname,
                     manager_hostname=hostname, assigned_at=time.time(), status='assigned', external=True)
        if restart:
            owner['handoff_record'] = str(archive / 'handoff.json')
            write_json(folder / 'supervisor_owner.json', owner)
        else:
            with (folder / 'supervisor_owner.json').open('x') as stream:
                json.dump(owner, stream)
        owned = True
        manifest = read(folder / 'run_manifest.json')
        output = archive or folder
        prompt = output / 'supervisor.prompt.txt'
        layouts = layout_override or manifest['layout_ids']
        text = make_prompt(folder, batch, task, layouts, owner_id, manifest)
        if restart:
            controller = read(folder / 'controller_status.json', {})
            request = read(Path(controller['request']), {}) if controller.get('request') else {}
            if not request and controller.get('status') == 'paused_for_main':
                paths = sorted((folder / task / 'seed_0' / controller.get('episode', '') / 'requests').glob('*/request.json'))
                if paths:
                    request = read(paths[-1], {})
                    controller['last_request_for_review'] = str(paths[-1])
            handoff = dict(status='assigned', authorized_by='main_auto_restart', at=time.time(),
                           old_supervisor=prior, new_owner_id=owner_id, controller=controller,
                           request_id=request.get('request_id'), episode=request.get('episode'),
                           step=request.get('step'), scene_reset=False)
            write_json(archive / 'handoff.json', handoff)
            text = text.replace('本次是全新隔离测试，不是接手旧局；', '本次是断线后的原场景监督交接；')
            text = (f'自动重启监督授权：读取 {archive / "handoff.json"}。只替换监督者，模型/仿真/runner不重启。'
                    '保留当前episode、step、条件版本、预算和干预历史，不提交初始条件来覆盖已有局。'
                    '先读既有interventions和示范记录，实际查看示范片段、当前三视角及全局原图，再写新owner的supervisor_ready.json。'
                    '重新核实已有L2/cue/bbox，正确复用或显式replan。不可继承旧会话“工具不存在”的结论。'
                    '若controller因监督超时处于paused_for_main，读取交接记录last_request_for_review直接查看图像，'
                    '完成接手审阅后写ready并等待管理器授权重试；其他接口故障不能自动放行。'
                    '只有全部计划layout达到官方终态并完成报告才结束；故障保持等待。'
                    f'本worker计划分母为{len(manifest["layout_ids"])}。\n\n' + text)
            if fresh_attempt:
                text = (f'基础设施恢复 attempt {worker.get("recovery_attempt_id", "unknown")}：上一仿真已退出，当前将启动新的关联 episode。'
                        '旧场景不再存在，不能声称恢复旧场景。启动后必须查看新 episode step 0 的多视角图像，'
                        '重新审阅并建立完整 L3+L2 cue/框条件，再放行任何动作。保留旧 attempt 的 summary、请求和干预记录。'
                        f'本次仅运行 layout={layouts}，审计 worker 目录保持为 {folder}。\n\n') + text
        prompt.write_text(text)
        events = output / 'supervisor.events.jsonl'
        log = events.open('x')
        env = os.environ.copy()
        env.update(ASTRA_SUPERVISOR_ID=owner_id, PYTHONDONTWRITEBYTECODE='1')
        env.update(PATH=str(POLICY_ENV / 'bin') + os.pathsep + env.get('PATH', ''),
                   CONDA_PREFIX=str(POLICY_ENV))
        # Run the user's configured interactive command directly. The prompt
        # is supplied on stdin; shlex.join keeps paths and arguments exact.
        codex_args = ['exec', '--json', '--skip-git-repo-check', '--sandbox', 'workspace-write',
                      '-C', str(folder), '-m', 'gpt-6-astra',
                      '-c', 'model_reasoning_effort="high"',
                      '-o', str(output / 'supervisor.final.md'), '-']
        command = supervisor_command(executable, codex_args)
        with prompt.open() as inp:
            child = subprocess.Popen(command, cwd=folder, env=env, stdin=inp,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        activity_stat = events.stat()
        state = dict(task=task, worker_id=unit, session=str(folder), owner_id=owner_id, pid=child.pid,
                     thread_id=None, returncode=None, ready=False, events=str(events), prompt=str(prompt),
                     model='gpt-6-astra', reasoning_effort='high', manager_pid=os.getpid(),
                     hostname=hostname, manager_hostname=hostname, external=True,
                     command=command, started_at=time.time(), updated_at=time.time(), heartbeat_at=time.time(),
                     last_activity_at=time.time(), activity_size=activity_stat.st_size,
                     activity_mtime_ns=activity_stat.st_mtime_ns, status='starting')
        entry = dict(child=child, state=state, folder=folder, lock=lock, log=log)
        if restart:
            state['handoff_record'] = str(archive / 'handoff.json')
            handoff.update(status='started', new_pid=child.pid, new_events=str(events))
            write_json(archive / 'handoff.json', handoff)
        refresh(entry)
        return entry
    except BaseException as exc:
        if child:
            stop_child(child)
        if owned:
            write_json(folder / 'supervisor_status.json', dict(task=task, session=str(folder),
                owner_id=owner_id, pid=child.pid if child else None, thread_id=None, returncode=127, ready=False,
                events=str(locals().get('events', folder / 'supervisor.events.jsonl')), model='gpt-6-astra', reasoning_effort='high',
                manager_pid=os.getpid(), hostname=socket.gethostname(), manager_hostname=socket.gethostname(),
                external=True, command=locals().get('command'), status='launch_error',
                error=repr(exc), updated_at=time.time(), heartbeat_at=time.time()))
        if log:
            log.close()
        lock.close()
        raise


def launch_detached(batch):
    """Detach the manager from the launching terminal, retaining an auditable log."""
    batch = Path(batch).resolve()
    if read(batch / 'parallel_manifest.json') is None:
        raise ValueError('Prepared batch manifest is required')
    preflight(batch)
    # Exclusive creation prevents accidental double launches, including retries.
    with (batch / 'external_supervisor_launcher.json').open('x') as record:
        command = [sys.executable, str(Path(__file__).resolve()), '--session', str(batch)]
        with (batch / 'external_supervisor_manager.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        state = dict(pid=child.pid, hostname=socket.gethostname(), command=command,
                     started_at=time.time(), launch_mode='detached_session',
                     log=str(batch / 'external_supervisor_manager.log'))
        json.dump(state, record, indent=2)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f'Detached manager exited {child.returncode}; inspect its log')
        current = read(batch / 'external_supervisor_manager.json', {})
        if current.get('pid') == child.pid and current.get('heartbeat_at'):
            return state
        time.sleep(0.2)
    raise RuntimeError('Manager heartbeat not confirmed within 30 seconds; inspect before retrying')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--detach', action='store_true',
                        help='Start a separate background session with persistent logs')
    parser.add_argument('--preflight-only', action='store_true',
                        help='Render all task prompts in this interpreter without starting processes')
    args = parser.parse_args()
    batch = args.session.resolve()
    checked = preflight(batch)
    if args.preflight_only:
        print(json.dumps(checked, indent=2))
        return
    if args.detach:
        print(json.dumps(launch_detached(batch), indent=2))
        return
    config = read(batch / 'parallel_manifest.json')
    if config is None:
        raise ValueError('Prepared batch manifest is required')
    executable = supervisor_executable()
    manager_lock = (batch / 'external_supervisor_manager.lock').open('a+')
    fcntl.flock(manager_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from scripts.restart_supervisors import ensure_watchdog
    ensure_watchdog(batch)
    manager = dict(manager_id='external_' + uuid.uuid4().hex[:12], pid=os.getpid(),
                   hostname=socket.gethostname(), session=str(batch), started_at=time.time(),
                   status='waiting_for_pool', model='gpt-6-astra', reasoning_effort='high',
                   maximum_active=config.get('concurrency', config.get('max_workers', 8)),
                   requested_max_workers=config.get('max_workers', 8), workers={}, alerts=[])
    active, started = {}, set()
    job_state, last_job_check = None, 0
    def stopped(signum, frame):
        raise InterruptedError(f'External supervisor manager received signal {signum}')
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    def persist():
        manager.update(updated_at=time.time(), heartbeat_at=time.time(), active=list(active),
                       started=sorted(started), job_state=job_state)
        write_json(batch / 'external_supervisor_manager.json', manager)
    try:
        while True:
            config = read(batch / 'parallel_manifest.json', config)
            workers = read(batch / 'workers_status.json', [])
            definitions = {unit_id(worker): worker for worker in config['workers']}
            schedules = {unit: read(Path(worker['session']) / 'task_scheduler_status.json', {})
                         for unit, worker in definitions.items()}
            if time.monotonic()-last_job_check >= 30:
                job_state = slurm_state(config.get('job_id'))
                last_job_check = time.monotonic()
            for task, entry in list(active.items()):
                refresh(entry)
                manager['workers'][task] = dict(entry['state'])
                if entry['child'].poll() is not None:
                    release(entry)
                    del active[task]
            pool_done, job_done = config.get('status') in POOL_TERMINAL, job_state in JOB_TERMINAL
            capacity = min(manager['maximum_active'], config.get('concurrency', manager['maximum_active']))
            launches, stops = plan_supervisors(workers, schedules, started, active,
                                               pool_done, job_done, capacity=capacity)
            for task, reason in stops.items():
                terminate(active[task], reason)
                manager['workers'][task] = dict(active[task]['state'])
                release(active.pop(task))
            for task in launches:
                started.add(task)
                active[task] = launch(definitions[task], batch, executable, manager['manager_id'])
                manager['workers'][task] = dict(active[task]['state'])
            # Normal all-layout completion gets time to write the task's final report.
            if pool_done or job_done:
                manager.setdefault('pool_terminal_observed_at', time.time())
                if time.time()-manager['pool_terminal_observed_at'] > 300:
                    for task, entry in list(active.items()):
                        terminate(entry, 'Final reporting exceeded 300 seconds after pool termination')
                        manager['workers'][task] = dict(entry['state'])
                        release(active.pop(task))
            manager['status'] = 'supervising' if active else 'waiting_for_pool'
            persist()
            if finished_waiting(config, job_done, active):
                manager['status'] = 'finished'
                break
            time.sleep(3)
    except BaseException as exc:
        manager.update(status='stopped' if isinstance(exc, InterruptedError) else 'manager_error', error=repr(exc))
        raise
    finally:
        # A repeated stop request must not interrupt cleanup and leave child owners alive.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for task, entry in list(active.items()):
            try:
                terminate(entry, 'External supervisor manager stopping')
                manager['workers'][task] = dict(entry['state'])
                release(entry)
                del active[task]
            except Exception as exc:
                manager['alerts'].append(dict(task=task, cleanup_error=repr(exc)))
        manager['finished_at'] = time.time()
        persist()
        manager_lock.close()


if __name__ == '__main__':
    main()
