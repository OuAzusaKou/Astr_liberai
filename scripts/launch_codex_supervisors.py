"""Attach one persistent Codex CLI session to each of eight paused workers.

This launcher never selects or approves robot actions. Each Codex process must
inspect images and submit its own request-bound decisions with respond_live.py.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def make_prompt(worker, batch, owner_id):
    task, folder = worker['task'], worker['session']
    prompt = f'''你是任务 {task} 的唯一高层机器人监督 Codex 会话。用户明确要求8个独立会话并行，每任务3局。你只负责 {folder}，其他7个worker由别的会话负责。主会话已停止向你的worker提交动作。owner_id={owner_id} 已通过环境 ASTRA_SUPERVISOR_ID 设置。

必须持续闭环监督直到你的任务3个布局(0,1,2)，seed=0全部得到终态，或仿真/必要接口真正阻塞。不要只做启动检查或给建议后结束，不要等待用户说继续。以 AGENTS.md 为契约。当前Slurm10176单节点8GPU已经运行，不要提交/取消Slurm，不要重启模型或仿真，不要reset场景，不要修改冻结的source_snapshot/runtime/config/checkpoint。你只写自己worker下的记录、响应和必要的小型分析脚本。其他worker/root共享代码只读。

先读取 {ROOT}/AGENTS.md、自己的 run_manifest.json、demos/{task}/demo_context.json，以及 {ROOT}/astra/bridge.py 和 scripts/respond_live.py 理解协议。主会话已查看训练示范，但你仍须实际 view_image 查看缓存的示范时序拼图/帧（demo_context中含路径，可在旧session相应demos目录找到），建立自己视觉上下文。align_blocks 和 classify_objects_by_language 已经用户授权无示范运行，其他任务必须查看示范。不要把训练示范绝对关节轨迹重放当前场景。

这是接手现有episode，主会话已推进约28步(倒球约24步)，所有动作目前在文件请求处暂停。读取当前episode的demo_context和interventions.jsonl了解既有审查，不重跑初始状态。当前场景提示：charger在左而插排在右；organize先alarm；fill_pen_holder笔筒右/笔左；cover_blocks布局0左红中蓝右绿；language布局0 cars左白篮/watches中蓝篮/wooden_toy右红篮。其他布局须重新读取指令/图像。

每轮：
1. 用 python3 {ROOT}/scripts/inspect_live.py {folder} 获取最新请求、动作和生成自己worker/live_contact.png；必须用 view_image 实际查看新图像，多视角不足可查看原始PNG。不能用循环自动approve、不能基于文字状态代替看图、不能调用另一个LLM替你监督。
2. 根据图像和机器人状态判断，执行前审查每次1..4个动作。用 python3 {ROOT}/scripts/respond_live.py {folder} REQUEST_ID MODE COUNT REASON --segment SEGMENT 发送唯一响应。支持approve、correct、takeover、hand_back、replan、blocked、abort；以实际脚本参数为准。连续TAKEOVER必须用 --continue-attempt 延续当前恢复，或 --new-attempt 开始失败后新的恢复尝试，不能漏记或把每个短动作都计为新尝试。每个新episode step0必须提供 --scene-review；切换子目标先显式查看示范对应片段和时间范围。真实干预及时记录原因、证据和改变。首次接手后写 supervisor_ready.json 包含owner_id、reviewed_demo_images、request_id、当前场景理解与时间。
3. 每步动作之后检查最新状态。成功以官方信号为准。任何有限夹爪超出[0,1]沿用原生限幅，不裁剪、不计干预。NaN/Inf/维度/关节越界仍拒绝。128次动作无进展仅诊断/接管，不能终止。恢复预算读取实际manifest/请求：新版6次不同依据的恢复/子目标，10接管/episode；已加载旧桥的运行仍实际3次，不能假称6次生效。重规划不能清零无进展计数。自己在supervisor_progress.json记录子目标进展证据和计数。最终操作目标视觉完成时，在响应附加 --completion-candidate --completion-evidence TEXT；1秒有效仿真推进仍未收到官方成功必须诊断、改变规划。普通中间子目标不触发此窗口；重复宣告完成不能重置窗口。若存在supervisor_directive.json，先读并按其说明确认最新用户规则。
4. 接管要提交显式动作字典，双臂关节绝对弧度各6维+双夹爪各1维，或双臂世界xyz+wxyz EE姿态+双夹爪，代码使用原生IK验证且每步关节变化≤0.15rad。不能把像素差当关节差；不能读仿真物体真值来控制。小动作闭环，恢复后hand_back要求 --recovery-evidence 并清除旧chunk。接口真的不足则blocked基础设施/接口，不伪装任务失败。不要用一次失败/没想到抓法作为不可恢复证据。
5. 等待模型/下个布局时用≤20秒sleep并读状态，不要提前结束会话。每个请求900秒监督超时；及时处理。若自己会话被限流/工具阻塞，写清楚阻塞信息，禁止自动放行。你有真实持续执行职责，不是仅检查日志。

完成条件：检查 {batch}/workers_status.json 中你任务的returncode和自己任务seed_0/layout_*/summary.json。controller_status=episode_finished只是单局结束；需要布局0,1,2各有唯一终态且runner退出才算该任务3局结束。若worker错误退出，报告已跑/未跑数及基础设施原因，不自行重启。最后写自己 supervisor_summary.json 和 .md，区分官方零干预成功、辅助成功、失败、基础设施错误及未运行，分母3；保留所有现有证据。最终回答具体结果与路径。

现在开始查看示范和当前请求，持续执行。
'''
    if worker.get('continuation'):
        layouts = worker['layout_ids']
        prompt = prompt.replace('3个布局(0,1,2)，seed=0全部得到终态',
                                f'剩余{len(layouts)}个布局{layouts}，seed=0全部得到终态')
        start = prompt.index('这是接手现有episode，')
        end = prompt.index('\n每轮：', start)
        prompt = prompt[:start] + (
            f'这是修复恢复计数接口后的新运行，只执行布局{layouts}。旧布局0阻塞记录保留在主批次，'
            '不能重跑或覆盖。模型/仿真正在启动；未生成请求时，每次等待不超过20秒，'
            '检查launcher.log、policy.log、simulator.log及continuation_status.json。'
            '准备阶段先实际查看训练示范。新运行恢复预算为6次/子目标。\n') + prompt[end:]
        start = prompt.index('完成条件：')
        end = prompt.index('\n现在开始', start)
        prompt = prompt[:start] + (
            f'完成条件：检查自己的 continuation_status.json 中 worker_returncode，以及'
            f'任务 seed_0/layout_*/summary.json。只需要本次布局{layouts}各有唯一终态且runner退出。'
            'controller_status=episode_finished只表示单局结束，必须继续监督后续布局。'
            '若进程错误退出或必要接口阻塞，记录已跑/未跑数，不自行重启。最后写'
            'supervisor_summary.json和.md，明确本次2局与主任务总共3局的区别。\n') + prompt[end:]
    return prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--session', required=True, type=Path)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    batch = args.session.resolve()
    config = json.loads((batch / 'parallel_manifest.json').read_text())
    if len(config['workers']) != 8:
        raise ValueError('Expected exactly eight workers')
    root = batch / 'codex_supervision'
    root.mkdir(exist_ok=True)
    locks, children, streams = [], [], []
    state = dict(supervision_session_id='codex8_' + uuid.uuid4().hex[:12], started_at=time.time(),
                 runtime_session=str(batch), previous_supervisor='single interactive Codex',
                 model='gpt-6-astra', reasoning_effort='high', workers=[], status='prepared')
    try:
        for worker in config['workers']:
            folder = Path(worker['session'])
            lock = (folder / 'supervisor.lock').open('a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
            owner_id = state['supervision_session_id'] + ':' + worker['task']
            prompt = root / (worker['task'] + '.prompt.txt')
            prompt.write_text(make_prompt(worker, batch, owner_id))
            state['workers'].append(dict(task=worker['task'], session=str(folder), owner_id=owner_id,
                                         prompt=str(prompt), thread_id=None, pid=None, returncode=None))
        if args.prepare_only:
            write_json(root / 'prepared.json', state)
            print(root / 'prepared.json')
            return
        # All workers are already waiting for explicit decisions. No reset occurs.
        for worker in state['workers']:
            folder = Path(worker['session'])
            owner_path = folder / 'supervisor_owner.json'
            if owner_path.exists():
                raise RuntimeError(f'Existing ownership must be reviewed before restart: {owner_path}')
        for worker in state['workers']:
            folder = Path(worker['session'])
            write_json(folder / 'supervisor_owner.json', dict(owner_id=worker['owner_id'],
                       assigned_at=time.time(), manager_pid=os.getpid(), status='assigned',
                       supervision_session_id=state['supervision_session_id']))
            log_path = root / (worker['task'] + '.events.jsonl')
            log = log_path.open('w'); streams.append(log)
            env = os.environ.copy()
            env['ASTRA_SUPERVISOR_ID'] = worker['owner_id']
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            cmd = ['codex', 'exec', '--json', '--skip-git-repo-check', '--sandbox', 'workspace-write',
                   '-C', str(folder), '-m', 'gpt-6-astra', '-c', 'model_reasoning_effort="high"',
                   '-o', str(root / (worker['task'] + '.final.md')), '-']
            with Path(worker['prompt']).open() as inp:
                child = subprocess.Popen(cmd, stdin=inp, stdout=log, stderr=subprocess.STDOUT,
                                         env=env, cwd=folder, start_new_session=True)
            children.append(child)
            worker.update(pid=child.pid, events=str(log_path), command=cmd)
        state['status'] = 'eight_codex_processes_started_verification_pending'
        write_json(root / 'status.json', state)
        def stop(signum, frame):
            raise InterruptedError(f'Supervisor manager signal {signum}')
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        while any(child.poll() is None for child in children):
            for worker, child in zip(state['workers'], children):
                worker['returncode'] = child.poll()
                if not worker['thread_id']:
                    with Path(worker['events']).open() as stream:
                        for _ in range(30):
                            line = stream.readline()
                            if not line: break
                            try: event = json.loads(line)
                            except ValueError: continue
                            if event.get('type') == 'thread.started':
                                worker['thread_id'] = event['thread_id']; break
                worker['ready'] = (Path(worker['session']) / 'supervisor_ready.json').exists()
            state['updated_at'] = time.time()
            write_json(root / 'status.json', state)
            time.sleep(3)
        state['status'] = 'codex_processes_exited_review_episode_summaries'
    except BaseException as exc:
        state.update(status='supervisor_manager_error', error=repr(exc))
        raise
    finally:
        for child in children:
            if child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        for worker, child in zip(state['workers'], children): worker['returncode'] = child.returncode
        for stream in streams: stream.close()
        if not args.prepare_only:
            state['finished_at'] = time.time()
            write_json(root / 'status.json', state)
        for lock in locks: lock.close()


if __name__ == '__main__':
    main()
