"""Own one task's Codex supervisor and run each planned layout exactly once.

This scheduler never publishes robot responses. A exited runner settles its
layout even if startup failed; it never retries a layout or rewrites a summary.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import uuid

from astra.config import POLICY_ENV, ROOT
from astra.allocation import allocated_gpu_ids, runtime_job_id


def write_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def remaining_layouts(planned, started):
    """Only never-started layouts are eligible; a failure is not a retry."""
    if len(set(planned)) != len(planned):
        raise ValueError('Duplicate planned layouts')
    if not set(started).issubset(planned):
        raise ValueError('Started layout outside plan')
    return [layout for layout in planned if layout not in started]


def supervisor_unavailable(returncode, planned, settled, runner_active):
    return returncode is not None and (runner_active or set(settled) != set(planned))



def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def external_supervisor_available(status, now):
    heartbeat = status.get('updated_at')
    return bool(status.get('pid') and status.get('owner_id')
                and status.get('returncode') is None
                # Scene readiness requires a running simulator to publish images.
                # Here check process liveness; the bridge gates every action on
                # the supervisor's visual review and conditioning response.
                and (status.get('ready') is True or bool(status.get('thread_id')))
                and isinstance(heartbeat, (int, float))
                and 0 <= now - heartbeat <= 120)


def cleanup_verified(record, runner_returncode, runner_started=True):
    if not runner_started:
        return True
    if runner_returncode is None or runner_returncode < 0:
        return False
    if not record.get('finished_at') or record.get('cleanup_errors'):
        return False
    children = {record[key] for key in ('policy_pid', 'simulator_pid') if record.get(key)}
    cleanup = record.get('cleanup')
    if cleanup is None:
        return not children and record.get('status') == 'runtime_error'
    if not isinstance(cleanup, list):
        return False
    cleaned = {row.get('pid') for row in cleanup if row.get('returncode') is not None}
    return children.issubset(cleaned) and all(row.get('returncode') is not None for row in cleanup)


def project_root(session):
    for path in session.parents:
        if (path / 'AGENTS.md').is_file() and (path / 'scripts/respond_live.py').is_file():
            return path
    raise ValueError('Cannot locate project AGENTS.md and control tools')


def make_prompt(session, batch, task, layouts, owner, manifest=None):
    root = project_root(session)
    if manifest is None:
        path = session / 'run_manifest.json'
        manifest = json.loads(path.read_text()) if path.is_file() else {}
    try:
        batch_manifest = json.loads((Path(batch) / 'parallel_manifest.json').read_text())
    except (OSError, json.JSONDecodeError):
        batch_manifest = {}
    execution_mode = batch_manifest.get('execution_mode')
    if execution_mode == 'bash_direct':
        allocation_rule = ('当前资源由主会话通过 bash 分配，节点/GPU 以本批次 manifest 为准；'
                           '不得提交 Slurm 作业、取消或扩容资源、重启服务。')
    elif execution_mode == 'slurm':
        allocation_rule = ('当前资源由 Slurm 分配，job name wangzihang_astra；'
                           '不得提交、取消、扩容作业或重启服务。')
    else:
        allocation_rule = '资源由主会话统一分配；不得提交、取消、扩容作业或重启服务。'
    if task in manifest.get('authorized_demo_waivers', []):
        demo_rule = (f'本轮用户已明确授权仅任务 {task} 无训练示范执行，依据本 worker manifest 的 '
                     'authorized_demo_waivers。不得把豁免泛化到其他任务。必须实际 view_image 查看初始多视角场景，'
                     '读取原任务指令并识别指定目标、目标关系与成功要求；每个子目标依据当前场景建立观察证据。'
                     'supervisor_ready.json 中记录 demo_waiver=true、waiver_task、waiver_basis、'
                     'reviewed_demo_images=[]、实际 reviewed_scene_images；不能声称已看训练视频。')
    else:
        demo_rule = ('必须实际 view_image 查看示范 contact_sheets 或逐帧图片，记录已看的帧、视角和各子目标；'
                     '仅读文字不算。每个独立子目标先显式查看对应片段及时间范围。'
                     '其他任务获得示范豁免不代表本任务可以跳过示范。')
    conditioning_mode = manifest.get('conditioning_mode', 'pure_l3')
    if conditioning_mode == 'l3_l2':
        conditioning_rule = '''本次 conditioning_mode=l3_l2，conditioning_provider=astra_visual。首次请求状态 PREPARE_CONDITIONING，必须先完成当前多视角及示范审阅，再实际 view_image 查看该请求 observation.images.cam_head 的原始 RGB 图（runtime 全局相机键；若实际键为 cam_high 则使用 cam_high）（不是 live_contact 拼图），根据该图建立 L2 条件。提交 condition 前不得要求模型生成动作或 approve。
在自己 worker 保存 conditioning JSON，字段为 subgoal（与 --segment 一致）、l2_instruction（非空当前操作子目标）、cue_rgb_path（该请求全局相机 cam_head/cam_high 原图路径）、l2_bboxes_xyxy（像素坐标 [N,4]）、box_source（只允许 astra_visual 或 validated_visual_locator，按实际视觉来源选择）、evidence（观察依据）、change_reason（本次建立/更新原因）；只有回位等明确无框子目标可传空列表，并写 unboxed_reason。框须满足 0 <= xmin < xmax <= W、0 <= ymin < ymax <= H。不可用训练示范帧、归一化框、腕部/拼图坐标或仿真真值。cue 必须是未经画框的本局当前原图，服务端负责画框；不能把已有框的图再传入。
L2 严格遵守本次 AGENTS.md 简洁规则：l2_instruction 只写一行简短英文祈使指令，表达当前一个子目标，例如 pick up the yellow mug、hang the yellow mug on the mug rack、pick up the cup with the right arm、move the cup to the vase and pour all balls into it。只保留动作、物体和必要目标/机械臂；不得加入观察、坐标、失败历史、恢复解释、逐步动作细节或冗长注意事项。详细依据单独写入 evidence/change_reason/recovery_evidence/reason，不拼接入 L2。每次只发当前一条，replan/hand_back/恢复时同样简短；参阅 AGENTS.md 的 episode 1000 和 2300 示例，结合当前布局选取指令，不照抄整段序列。
初次运行 respond_live.py SESSION REQUEST_ID condition 0 REASON --segment SEGMENT --scene-review "当前场景审阅证据" --conditioning-file FILE；条件校验后模型才开始推理。condition 只准备条件，不执行动作；下一请求仍需实际看图审查候选动作。L3 instruction/task_instruction 保留整局原始目标，不用 --instruction 把它替换成 L2；子目标放入 l2_instruction。
同一子目标沿用固定起始 cue、框和版本，每次重新推理同时携带这些条件和最新三路观测。replan 与 hand_back 必须明确给出 --conditioning-version VERSION（来自 request.conditioning.conditioning_version，已重新核对仍适用）或 --conditioning-file FILE（新子目标/恢复条件）。切换子目标、滑落换目标或原条件失效时，基于当前请求全局相机原图建立新的 cue/框/L2，用 replan 新条件原子替换并丢弃旧 chunk；不能仅 --segment 改名继续 approve 旧动作。接管后 hand_back 必须核对条件、清旧 chunk/模型历史，若已完成当前子目标则提交下一子目标新条件。不能仅改名重置停滞/恢复计数。条件缺失/非法时保持待审并修正输入；接口错误暂停报告主会话，不能降级 pure_l3。
审计核对每次 policy 请求的 L3/L2、conditioning_version、起始 step、cue 路径/哈希/尺寸、框及来源、无框理由，并核对模型实际接收记录。Astra 提供 L2/框属于高层条件辅助，即使没有 correct/takeover，成功也不能归为 low policy 独立成功。'''
        allowed_modes = 'condition/approve/correct/takeover/replan/hand_back'
    elif conditioning_mode == 'pure_l3':
        conditioning_rule = ('本 worker 历史 manifest 的 conditioning_mode=pure_l3（缺失字段同样按 pure_l3 解释），'
                             '保持本局实际模式，不得宣称已有 L2/cue/框接入，不修改冻结配置；'
                             '新 L3+L2 规则需由主会话建立新关联 session 后生效。')
        allowed_modes = 'approve/correct/takeover/replan/hand_back'
    else:
        raise ValueError(f'Unsupported conditioning_mode: {conditioning_mode!r}')
    from astra.monitoring_config import monitoring_config
    cadence = monitoring_config(manifest)
    interval = cadence['high_level_monitor_interval_actions']
    takeover_interval = cadence.get('max_takeover_actions_per_decision', interval)
    capacity = batch_manifest.get('max_workers', 8)
    if len(layouts) == 1:
        execution_scope = (f'当前 worker 只负责 layout={layouts[0]} 这一局；不要等待、启动或监督同任务的其他 layout。'
                          '调度器会在本局终态和清理完成后释放 GPU 槽位，再从全局队列分配下一个独立 worker。')
        completion_scope = ('自己的 task_scheduler_status.json 显示 all_layouts_settled=true、当前 runner 已退出，且核对本布局官方终态/启动错误证据后，'
                            '写 supervisor_summary.json 和 supervisor_summary.md 并结束。分母为 1；当前局结束不代表同任务其他布局结束。'
                            '重试须由主会话作为显式新 attempt 计划，不能自行重复。')
    else:
        execution_scope = ('当前 worker 负责本 manifest 中列出的布局；只有官方终态后停止新增动作并等调度器启动下一 layout，不自行 reset。'
                           '调度器只负责按计划依次启动，不会替你批准动作。')
        completion_scope = ('自己的 task_scheduler_status.json 显示 all_layouts_settled=true、当前 runner 已退出，且核对所有布局终态/启动错误证据后，'
                            f'写 supervisor_summary.json 和 supervisor_summary.md 并结束。分母 {len(layouts)}。'
                            '本局 done 不能提前退出，否则调度器会保护性停止后续测试。')
    return f'''你是 {task} 的唯一机器人高层执行会话。本次是全新隔离测试，不是接手旧局；只负责 {session}，计划 seed=0、layout={layouts}。owner_id={owner}，环境 ASTRA_SUPERVISOR_ID 已设置。你有持续执行职责，不能只检查或提建议后结束。
先读 {root}/AGENTS.md、{session}/run_manifest.json、demos/{task}/demo_context.json 及 {root}/astra/bridge.py、{root}/scripts/respond_live.py。{demo_rule}查看当前请求图像后才写 supervisor_ready.json：owner_id、request_id、reviewed_demo_images、current_scene_understanding、reviewed_at。未看图不得声称 ready。新场景须核对对象、布局、方向；若有示范，不能重放其绝对轨迹。
所有任务由主会话统一管理，本批最多 {capacity} 个单 episode worker 独立并行；同一任务的不同 layout 使用不同 worker/session/Codex 会话。{allocation_rule}禁止更改 manifest、冻结 runtime/source_snapshot、checkpoint、成功条件、对象位姿和 seed。只在自己的 worker 内写记录、控制响应和必要分析文件；其他目录只读。不得读取旧局兼容接管追加额度作为本次授权。
每轮运行 python3 {root}/scripts/inspect_live.py {session} 读取 request 并生成 live_contact.png，必须 view_image 实际看当前多视角图像后审查动作，必要时查看原始相机帧。不能自动 approve 循环、不能仅读状态代替看图。使用 python3 {root}/scripts/respond_live.py {session} REQUEST_ID MODE COUNT REASON --segment SEGMENT；先 --help 核对真实参数。本会话只可提交 {allowed_modes}；禁止提出任务终止，禁止提交 blocked/abort/reset、停止 runner/仿真或通过退出监督会话结束当前局。即使脚本帮助仍列出旧模式，也不构成本次授权。本 worker 实际高层监控间隔为 {interval} 次环境动作；low policy 和修正默认每批 {interval} 步；完全接管时默认一次提交 {takeover_interval} 步连续动作，不逐步输出，每批后重新观察。核对 request 的 high_level_monitor_interval_actions/max_actions_per_decision/max_takeover_actions_per_decision 与 manifest 一致（历史 manifest 缺少接管字段时沿用原批次限制）。已知接触风险或精细恢复可缩短本批，用 --short-batch-reason 或 reason 记录具体证据；可用 chunk 不足时提前审查，不拼接未经审阅的新 chunk。replan_steps=16 保持独立配置。底层逐动作检查官方终态、预算和合法性，不能为凑满批次继续执行。step 0 必须 --scene-review "当前场景审阅证据"。保持单一 owner、request ID 绑定、原子提交，不覆盖响应。
上述控制工具命令中的 python3 必须替换为 {POLICY_ENV}/bin/python 的绝对路径，避免 shell 激活环境变化。推理服务与 Astra 图像/响应工具使用 POLICY_ENV；仿真进程由调度器使用独立 SIM_ENV 启动，不自行混用或安装依赖。
{conditioning_rule}
本次恢复尝试和接管事件次数均无限（unlimited_recovery=true，两个次数上限为 null），allow_agent_termination=false；所有动作共同计入原生任务动作上限×3；核对每次 request 的 original_step_limit/step_limit_multiplier/step_limit，缺失或不符则暂停动作并报告主会话核对，不能提交结束响应。一次接管无额外动作总上限，尽量少用有效动作，可连续多个短批。连续 TAKEOVER 必须 --continue-attempt（当前尝试继续）或 --new-attempt（失败后改变方法再试），不能漏记，也不能每短批当新尝试；重命名子目标不能清零次数。replan 没有硬次数上限，但须新证据、调整和后续执行验证。
low policy 停滞、空抓、抓错、滑落时及时诊断，不能只反复改提示。明确失败可直接接管，用已验证的显式动作：双臂各 6 维绝对关节弧度加双夹爪，或双臂世界 xyz+wxyz EE 姿态加双夹爪，通过原生 IK，单步关节变化≤0.15 rad。不得把像素偏差直接当关节增量、不得用仿真物体真值控制。接管动作后观察验证，再 hand_back --recovery-evidence，清旧 chunk 和历史。有限夹爪超出标称 [0,1] 交原生适配器处理，不因它干预；NaN/Inf、错维度或关节越界仍拒绝。
128 次环境动作无有效进展仅触发诊断/接管，不能单独结束；重规划不能清零停滞计数。将进展证据、恢复变化、累计次数与剩余环境动作预算记入 supervisor_progress.json/intervention 相关记录。最终操作视觉完成用 --completion-candidate --completion-evidence TEXT；1.0 秒有效仿真推进仍无官方成功必须诊断并改变规划。普通中间子目标不触发；不能重复宣告重置窗口。无逐项官方日志时写具体失败项未知。成功以官方信号为准。
只有官方终态到达后停止新增动作，不自行 reset。{execution_scope}不得因停滞、次数或恢复困难提出终止、提交结束响应或自行退出。各局独立 runner，日志在 runs/layout_N/。调度器不会替你批准动作。必要接口或基础设施故障时暂停动作、保存场景与具体证据，在自己 worker 写 supervisor_incident.json 并报告主会话，保持监督会话等待协调；不得自行 blocked/abort、reset、停进程或退出。若自己工具/限流失效同样只暂停上报，不自动批准。运行中规则更新须确认带版本的主会话授权，保留历史计数，不改写已结束局。
完成条件：{completion_scope}本局 done 不得早于官方终态或以结束会话替代结果核对。启动阶段暂无 request 时继续等待而不是结束。
主批次 {batch} 仅用于关联记录，不能写别的 worker。现在开始，持续监督本 worker 的唯一布局。
'''


def stop_group(child, grace=60):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def read_thread_id(events):
    with events.open() as stream:
        for _, line in zip(range(100), stream):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('type') == 'thread.started':
                return event.get('thread_id')
    return None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--external-supervisor', action='store_true')
    parser.add_argument('--retry-layout', type=int,
                        help='Retry one failed/interrupted layout in this existing worker directory')
    parser.add_argument('--attempt-id', help='Audit identifier for --retry-layout')
    args = parser.parse_args(argv)
    session, batch = args.session.resolve(), args.batch.resolve()
    manifest = json.loads((session / 'run_manifest.json').read_text())
    if len(manifest['tasks']) != 1:
        raise ValueError('Exactly one task per supervisor')
    job_id = runtime_job_id()
    if not os.environ.get('SLURM_JOB_ID'):
        allocated_gpu_ids()
    task, layouts = manifest['tasks'][0], manifest['layout_ids']
    retry_layout = args.retry_layout
    if retry_layout is not None:
        if retry_layout not in layouts:
            raise ValueError(f'Retry layout {retry_layout} is not in the worker plan {layouts}')
        if not args.attempt_id:
            raise ValueError('--attempt-id is required with --retry-layout')
        run_layouts = [retry_layout]
        previous = read_json(session / 'task_scheduler_status.json')
        if previous:
            with (session / 'task_scheduler_history.jsonl').open('a') as history:
                history.write(json.dumps(dict(recorded_at=time.time(), status=previous), ensure_ascii=False) + '\n')
    else:
        remaining_layouts(layouts, [])
        run_layouts = layouts
    if not layouts or any(not isinstance(x, int) or x < 0 for x in layouts):
        raise ValueError('Nonempty nonnegative layout IDs required')
    codex_executable = shutil.which('codex') if not args.external_supervisor else None
    if not args.external_supervisor and not codex_executable:
        raise RuntimeError('codex executable missing from inherited PATH; no runner has started')
    lock = None
    if not args.external_supervisor:
        lock = (session / 'supervisor.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.external_supervisor and (session / 'supervisor_owner.json').exists():
        raise ValueError('Existing supervisor ownership requires explicit handoff; no automatic relaunch')
    if retry_layout is None and any((session / 'runs' / f'layout_{x}').exists() or
           list((session / task / 'seed_0').glob(f'layout_{x}_*')) for x in layouts):
        raise ValueError('Planned layout was already started; no automatic retry')
    owner = f'core9_{uuid.uuid4().hex[:12]}:{task}'
    if not args.external_supervisor:
        write_json(session / 'supervisor_owner.json', dict(owner_id=owner, assigned_at=time.time(),
                   manager_pid=os.getpid(), status='assigned'))
    prompt = session / 'supervisor.prompt.txt'
    if not args.external_supervisor:
        prompt.write_text(make_prompt(session, batch, task, layouts, owner, manifest))
    events = session / 'supervisor.events.jsonl'
    state = dict(task=task, session=str(session), batch=str(batch), gpu=args.gpu,
                 job_id=job_id, planned_layouts=layouts, run_layouts=run_layouts,
                 recovery_of=str(session) if retry_layout is not None else None,
                 attempt_id=args.attempt_id,
                 started_layouts=[], settled_layouts=[], remaining_layouts=run_layouts[:],
                 not_run_layouts=[], layout_results=[],
                 layouts=[dict(layout_id=x, state='not_started', returncode=None,
                               run_status_path=str(session / 'runs' /
                                                   (f'layout_{x}_attempt_{args.attempt_id}'
                                                    if retry_layout == x else f'layout_{x}') /
                                                   'run_status.json'))
                          for x in layouts], current_layout=None, runner_pid=None,
                 runner_returncode=None, all_layouts_settled=False, started_at=time.time(),
                 status='starting_supervisor', manager_pid=os.getpid(),
                 external_supervisor=args.external_supervisor, gpu_quarantined=False)
    supervisor = dict(task=task, session=str(session), owner_id=owner, model='gpt-6-astra',
                      reasoning_effort='high', manager_pid=os.getpid(), pid=None, thread_id=None, returncode=None,
                      ready=False, events=str(events), prompt=str(prompt), started_at=time.time())
    runner = codex = log = None
    streams = []
    def supervisor_poll():
        if args.external_supervisor:
            return read_json(session / 'supervisor_status.json').get('returncode')
        return codex.poll() if codex else None
    def supervisor_lost():
        if args.external_supervisor:
            status = read_json(session / 'supervisor_status.json')
            # A recovery attempt starts the runner and replacement supervisor
            # together.  Their startup handshake may briefly be ``starting``;
            # action requests remain gated by the bridge until ready/visual
            # review is published.
            if (args.attempt_id and status.get('pid') and status.get('returncode') is None
                    and isinstance(status.get('updated_at'), (int, float))
                    and time.time() - status['updated_at'] <= 120):
                return False
            return not external_supervisor_available(status, time.time())
        return codex.poll() is not None
    def persist():
        nonlocal supervisor
        state['updated_at'] = time.time()
        if args.external_supervisor:
            # Preserve the remote writer's heartbeat; refreshing it here hides failures.
            supervisor = read_json(session / 'supervisor_status.json')
        else:
            supervisor['updated_at'] = time.time()
            supervisor['returncode'] = supervisor_poll()
            supervisor['ready'] = (session / 'supervisor_ready.json').exists()
            if not supervisor['thread_id'] and events.exists():
                supervisor['thread_id'] = read_thread_id(events)
        for item in state['layouts']:
            if item['layout_id'] in state['not_run_layouts']:
                item['state'] = state.get('not_run_reason', 'not_run_scheduler_stopped')
        state['codex'] = dict(supervisor)
        if not args.external_supervisor:
            write_json(session / 'supervisor_status.json', supervisor)
        write_json(session / 'task_scheduler_status.json', state)
    def stop(signum, frame):
        raise InterruptedError(f'Task scheduler received signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    code = 0
    try:
        if args.external_supervisor:
            state['status'] = 'waiting_for_supervisor'
            deadline = time.time() + 180
            while supervisor_lost() and time.time() < deadline:
                persist()
                time.sleep(2)
        else:
            codex_log = events.open('w'); streams.append(codex_log)
            env = os.environ.copy()
            env.update(ASTRA_SUPERVISOR_ID=owner, PYTHONDONTWRITEBYTECODE='1')
            cmd = [codex_executable, 'exec', '--json', '--skip-git-repo-check', '--sandbox', 'workspace-write',
                   '-C', str(session), '-m', 'gpt-6-astra', '-c', 'model_reasoning_effort="high"',
                   '-o', str(session / 'supervisor.final.md'), '-']
            with prompt.open() as inp:
                codex = subprocess.Popen(cmd, stdin=inp, stdout=codex_log, stderr=subprocess.STDOUT,
                                         env=env, cwd=session, start_new_session=True)
            supervisor.update(pid=codex.pid, command=cmd)
        persist()
        while state['remaining_layouts']:
            if supervisor_lost():
                state.update(status='supervisor_unavailable', not_run_layouts=state['remaining_layouts'][:],
                             not_run_reason='not_run_supervisor_unavailable')
                code = 2
                break
            layout = state['remaining_layouts'][0]
            layout_status = next(x for x in state['layouts'] if x['layout_id'] == layout)
            layout_status['state'] = 'starting'
            state.update(current_layout=layout, runner_pid=None, runner_returncode=None,
                         status='starting_layout')
            # Mark before spawning: even a startup failure is never silently retried.
            state['started_layouts'].append(layout)
            state['remaining_layouts'] = remaining_layouts(run_layouts, state['started_layouts'])
            write_json(session / 'controller_status.json', dict(status='starting_layout',
                       task=task, layout_id=layout, updated_at=time.time()))
            persist()
            log = (session / f'layout_{layout}.launcher.log').open('w'); streams.append(log)
            env = os.environ.copy()
            env.update(PYTHONPATH=os.pathsep.join((str(ROOT), str(session / 'source_snapshot'))),
                       PYTHONDONTWRITEBYTECODE='1')
            cmd = [str(POLICY_ENV / 'bin/python'), '-m', 'astra.run', '--session', str(session),
                   '--gpu', str(args.gpu), '--layout-id', str(layout)]
            started = time.time()
            if retry_layout is not None:
                cmd.extend(['--attempt-id', str(args.attempt_id)])
            result = dict(layout_id=layout, started_at=started, command=cmd,
                          attempt_id=args.attempt_id)
            runner_started = False
            runner = None
            try:
                # Recovery attempts must use the live patched scheduler/runtime
                # from this checkout.  The worker snapshot predates attempt-id
                # support and would reject the recovery CLI before rollout.
                runner = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                runner_started = True
                state.update(runner_pid=runner.pid, status='running_layout')
                layout_status.update(state='running', pid=runner.pid)
                supervisor_pause = None
                while runner.poll() is None:
                    if supervisor_lost():
                        if supervisor_pause is None:
                            supervisor_pause = dict(
                                event='supervisor_lost', status='paused_for_main',
                                pause_id=uuid.uuid4().hex, created_at=time.time(),
                                task=task, layout_id=layout, runner_pid=runner.pid,
                                error='Supervisor unavailable; live runner retained awaiting main coordination',
                                controller=read_json(session / 'controller_status.json'))
                            with (session / 'task_supervisor_incidents.jsonl').open('a') as incident_log:
                                incident_log.write(json.dumps(supervisor_pause) + '\n')
                            write_json(session / 'supervisor_loss_incident.json', supervisor_pause)
                        # The control bridge waits for a reviewed response at its next
                        # request boundary. Do not kill/reset the live scene or publish
                        # responses; any conditioning timeout also waits for main.
                        state.update(status='paused_for_main', supervisor_pause=supervisor_pause)
                        layout_status['state'] = 'paused_for_main'
                    elif supervisor_pause is not None:
                        resumed = dict(supervisor_pause, event='supervisor_available',
                                       status='supervision_restored', resumed_at=time.time())
                        with (session / 'task_supervisor_incidents.jsonl').open('a') as incident_log:
                            incident_log.write(json.dumps(resumed) + '\n')
                        write_json(session / 'supervisor_loss_incident.json', resumed)
                        supervisor_pause = None
                        state.pop('supervisor_pause', None)
                        state['status'] = 'running_layout'
                        layout_status['state'] = 'running'
                    persist()
                    time.sleep(2)
                result['runner_returncode'] = runner.poll()
                state['runner_returncode'] = runner.poll()
            except OSError as exc:
                result.update(runner_returncode=127, infrastructure_error=repr(exc))
                state['runner_returncode'] = 127
            result.update(finished_at=time.time(), summaries=[str(p) for p in
                          (session / task / 'seed_0').glob(f'layout_{layout}_*/summary.json')])
            if not result['summaries']:
                result.setdefault('infrastructure_error', 'Runner exited without episode summary')
            # A runner can exit zero after writing an audited infrastructure
            # error. Keep that error visible to the pool instead of treating
            # the layout as a successful worker completion.
            if (result.get('infrastructure_error')
                    or result.get('runner_returncode') not in (0, None)):
                code = max(code, 1)
            runtime_status = read_json(Path(layout_status['run_status_path']))
            result['cleanup_verified'] = cleanup_verified(runtime_status, result['runner_returncode'], runner_started)
            if not result['cleanup_verified']:
                state.update(status='cleanup_unverified', gpu_quarantined=True,
                             not_run_reason='not_run_gpu_quarantined')
                result['infrastructure_error'] = 'Runner cleanup missing, incomplete or terminated by signal'
                code = 4
            layout_status.update(state='settled' if result['cleanup_verified'] else 'cleanup_unverified', returncode=result['runner_returncode'],
                                 summaries=result['summaries'],
                                 infrastructure_error=result.get('infrastructure_error'))
            state['layout_results'].append(result)
            state['settled_layouts'].append(layout)
            state['all_layouts_settled'] = set(state['settled_layouts']) == set(run_layouts)
            state['current_layout'] = None
            log.close()
            persist()
            if code:
                if state.get('status') != 'cleanup_unverified':
                    state['status'] = 'infrastructure_error'
                state['not_run_layouts'] = state['remaining_layouts'][:]
                break
        if state['all_layouts_settled'] and not code:
            state['status'] = 'awaiting_supervisor_summary'
            deadline = time.time() + 300
            while supervisor_poll() is None and time.time() < deadline:
                if deadline - time.time() <= 180:
                    state['warning'] = 'Supervisor still finalizing more than 120 seconds after all layouts settled'
                persist()
                time.sleep(2)
            if supervisor_poll() is None:
                state.update(status='supervisor_finalization_timeout', error='No supervisor exit within 300 seconds')
                code = 3
            else:
                state['status'] = 'finished'
                if supervisor_poll() != 0 or not (session / 'supervisor_summary.json').exists():
                    state.update(status='supervisor_summary_incomplete', error='Supervisor failed or did not write summary')
                    code = 3
    except BaseException as exc:
        state.update(status='scheduler_interrupted' if isinstance(exc, InterruptedError) else 'scheduler_error',
                     error=repr(exc), not_run_layouts=state['remaining_layouts'][:])
        code = 130 if isinstance(exc, InterruptedError) else 1
    finally:
        # First stop decisions, then runner; astra.run cleans its detached server/sim groups.
        if not args.external_supervisor:
            stop_group(codex, grace=10)
        stop_group(runner)
        state['finished_at'] = time.time()
        if not args.external_supervisor:
            supervisor['finished_at'] = state['finished_at']
        if runner:
            state['runner_returncode'] = runner.poll()
        if state['current_layout'] is not None:
            active = next(x for x in state['layouts'] if x['layout_id'] == state['current_layout'])
            active.update(state='interrupted', returncode=state['runner_returncode'],
                          infrastructure_error=state.get('error', state['status']))
            if not cleanup_verified(read_json(Path(active['run_status_path'])), state['runner_returncode'], runner is not None):
                active['state'] = 'cleanup_unverified'
                state.update(status='cleanup_unverified', gpu_quarantined=True,
                             not_run_reason='not_run_gpu_quarantined')
                code = 4
        state['returncode'] = code
        persist()
        for stream in streams:
            stream.close()
        if lock is not None:
            lock.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
