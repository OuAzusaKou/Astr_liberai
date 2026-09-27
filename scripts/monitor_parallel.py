"""Collect independent supervisor/rollout status; never submits robot actions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra.live_recovery import overlay_request, completed_episode_metrics
from astra.runtime_control import (overlay_request as overlay_runtime_control,
                                   completed_episode_metrics as runtime_episode_metrics)


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def classify(result):
    status = result.get('status')
    if status in {'blocked', 'infrastructure_error'}:
        return status
    if status == 'success' and result.get('official_terminal') is True and result.get('official_success') is True:
        if result.get('high_level_conditioning_assistance') is True:
            return 'assisted_success'
        counts = (result.get('logical_takeovers', result.get('takeovers')),
                  result.get('logical_corrections', result.get('corrections')))
        if all(type(count) is int and count >= 0 for count in counts):
            return 'zero_intervention_success' if sum(counts) == 0 else 'assisted_success'
        return 'success_intervention_unknown'
    if status == 'failed':
        if str(result.get('failure') or '').startswith('agent_aborted:'):
            return 'agent_aborted_failure'
        if result.get('official_terminal') is True and result.get('official_success') is False:
            return 'official_failure'
    return 'unclassified'


def episode_budget(record, manifest):
    """Actual request/summary evidence wins over requested future configuration."""
    limit = record.get('step_limit')
    multiplier = record.get('step_limit_multiplier', 1 if limit is not None else None)
    original = record.get('original_step_limit', limit if multiplier == 1 else None)
    return dict(original_step_limit=original, step_limit_multiplier=multiplier, step_limit=limit,
                configured_step_limit_multiplier=manifest.get('step_limit_multiplier', 1))


def snapshot(batch):
    now = time.time()
    config = read(batch / 'parallel_manifest.json')
    supervisors = read(batch / 'codex_supervision/status.json', {})
    sessions = {w.get('worker_id', w['task']): w for w in supervisors.get('workers', [])}
    processes = {w.get('worker_id', w['task']): w for w in read(batch / 'workers_status.json', [])}
    continuations = read(batch / 'continuations.json', {}).get('workers', [])
    extension_history = read(batch / 'recovery_extensions.json', {})
    rows, alerts = [], []
    for worker in config['workers']:
        task, folder = worker['task'], Path(worker['session'])
        worker_id = worker.get('worker_id', task)
        manifest = read(folder / 'run_manifest.json', {})
        source_folders = [folder]
        owner = sessions.get(worker_id, sessions.get(task, {}))
        rc = processes.get(worker_id, processes.get(task, {})).get('returncode')
        execution_history = [dict(session=str(folder), worker_returncode=rc)]
        for continuation in [entry for entry in continuations if entry['task'] == task]:
            folder = Path(continuation['session'])
            if folder in source_folders:
                alerts.append(dict(task=task, reason='duplicate_continuation_session', session=str(folder)))
                continue
            source_folders.append(folder)
            status = read(Path(continuation['status_file']), {})
            execution_history.append({**status, 'session': str(folder), 'status_file': continuation['status_file']})
            rc = status.get('worker_returncode')
            owner = dict(thread_id=status.get('codex_thread_id'), pid=status.get('codex_pid'),
                         returncode=status.get('codex_returncode'), ready=status.get('ready', False), events=status.get('events'))
        state = read(folder / 'controller_status.json', {})
        task_schedule = read(folder / 'task_scheduler_status.json', {})
        request = read(Path(state['request']), {}) if state.get('request') else {}
        if request:
            request = overlay_request(folder, request)
            request = overlay_runtime_control(folder, request)
        waiting = state.get('status') == 'waiting_for_agent' and not Path(state.get('response', '/nonexistent')).exists()
        configured_layouts = manifest.get('layout_ids', worker.get('layout_ids', [0, 1, 2]))
        expected = list(dict.fromkeys(configured_layouts))
        if len(expected) != len(configured_layouts):
            alerts.append(dict(task=task, reason='duplicate_planned_layouts'))
        seed = manifest.get('seed', config.get('seed', 0))
        attempts, results, unexpected = [], [], []
        by_layout = {layout: [] for layout in expected}
        for episode in sorted(path for source in source_folders for path in (source / task).glob('seed_*/*')):
            if not episode.is_dir():
                continue
            match = re.fullmatch(r'layout_(\d+)_.+', episode.name)
            layout = int(match[1]) if match else None
            valid = episode.parent.name == f'seed_{seed}' and layout in by_layout
            summary = read(episode / 'summary.json')
            attempt = dict(path=str(episode), episode=episode.name, layout_id=layout,
                           seed=episode.parent.name, planned_layout=valid, has_summary=isinstance(summary, dict))
            attempts.append(attempt)
            if valid:
                by_layout[layout].append(attempt)
            else:
                unexpected.append(str(episode))
            if isinstance(summary, dict):
                result = dict(**summary, path=str(episode / 'summary.json'), episode=episode.name,
                              layout_id=layout, planned_layout=valid)
                evidence = summary
                if 'step_limit' not in summary:
                    saved_requests = sorted((episode / 'requests').glob('*/request.json'))
                    evidence = read(saved_requests[-1], {}) if saved_requests else {}
                result['step_budget'] = episode_budget(evidence, read(episode.parents[2] / 'run_manifest.json', {}))
                for extension in extension_history.values():
                    if extension['task'] == task and extension['episode_path'] == str(episode):
                        result['compatibility_recovery'] = extension
                        result['logical_takeovers'] = extension['logical_takeovers']
                        result['logical_retries'] = extension['logical_retries']
                        result['logical_corrections'] = (summary.get('corrections', 0)
                            - extension['native_correct_transports'] - extension['native_replan_handbacks'])
                final_extension = completed_episode_metrics(episode.parents[2], episode)
                if final_extension:
                    result['compatibility_recovery'] = final_extension
                    result['logical_takeovers'] = final_extension['logical_takeovers']
                    result['logical_retries'] = final_extension['logical_retries']
                    result['logical_corrections'] = (summary.get('corrections', 0)
                        - final_extension['native_correct_transports'] - final_extension['native_replan_handbacks'])
                runtime_metrics = runtime_episode_metrics(episode.parents[2], episode)
                if runtime_metrics:
                    result['runtime_control'] = runtime_metrics
                    result['logical_takeovers'] = runtime_metrics['takeovers']
                    result['logical_retries'] = runtime_metrics['retries']
                    native_corrections = summary.get('corrections')
                    result['logical_corrections'] = (
                        native_corrections - runtime_metrics['transport_corrections']
                        - runtime_metrics['transport_handbacks']
                        if type(native_corrections) is int else None)
                    result['unlimited_recovery'] = True
                    result['allow_agent_termination'] = False
                    result['max_retries_per_subgoal'] = None
                    result['max_takeovers'] = None
                results.append(result)
        duplicates = [layout for layout, values in by_layout.items() if len(values) > 1]
        started = [layout for layout, values in by_layout.items() if values]
        finished = [layout for layout, values in by_layout.items() if any(v['has_summary'] for v in values)]
        missing = [layout for layout in expected if layout not in finished]
        classifications = {}
        for layout in finished:
            candidates = [r for r in results if r['planned_layout'] and r['layout_id'] == layout]
            classifications[layout] = 'ambiguous' if layout in duplicates else classify(candidates[0])
        counts = {category: sum(value == category for value in classifications.values()) for category in
                  ('zero_intervention_success', 'assisted_success', 'success_intervention_unknown',
                   'official_failure', 'agent_aborted_failure', 'blocked', 'infrastructure_error',
                   'unclassified', 'ambiguous')}
        active = [layout for layout in started if layout not in finished]
        running = [layout for layout in active if rc is None and
                   any(Path(attempt['path']).is_relative_to(folder) for attempt in by_layout[layout])]
        if task_schedule.get('layouts'):
            executing = {item['layout_id'] for item in task_schedule['layouts'] if item.get('state') in {'running', 'starting'}}
            running = [layout for layout in running if layout in executing]
        interrupted = [layout for layout in active if layout not in running]
        planned, tested = len(expected), len(finished)
        successes = sum(counts[k] for k in ('zero_intervention_success', 'assisted_success', 'success_intervention_unknown'))
        failures = counts['official_failure'] + counts['agent_aborted_failure']
        blocked = counts['blocked'] + counts['infrastructure_error']
        event_path = Path(owner['events']) if owner.get('events') else None
        latest_result = max((r for r in results if r['planned_layout']),
                            key=lambda r: Path(r['path']).stat().st_mtime, default={})
        evidence = {}
        evidence_sources = {}
        unlimited_recovery = (request.get('unlimited_recovery', False) if request
                              else latest_result.get('unlimited_recovery', False))
        for key, summary_key in [('takeover_count', 'takeovers'), ('retries', 'retries'),
                                 ('max_retries_per_subgoal', 'max_retries_per_subgoal'), ('max_takeovers', 'max_takeovers')]:
            if unlimited_recovery and key in {'max_retries_per_subgoal', 'max_takeovers'}:
                evidence[key] = None
                evidence_sources[key] = (str(folder / 'runtime_control.json') if request
                                         else latest_result.get('path'))
                continue
            logical_summary_key = {'takeovers': 'logical_takeovers', 'retries': 'logical_retries'}.get(summary_key)
            if request.get(key) is not None:
                evidence[key], evidence_sources[key] = request[key], state.get('request')
                if request.get('live_recovery_extension'):
                    evidence_sources[key] = dict(native_request=state.get('request'),
                        authorization=str(folder / 'live_recovery_extension.json'),
                        ledger=str(Path(state['request']).parents[2] / 'requests/*/response.json'))
                if request.get('runtime_control_authorization'):
                    evidence_sources[key] = dict(native_request=state.get('request'),
                        authorization=str(folder / 'runtime_control.json'),
                        ledger=str(Path(state['request']).parents[2] / 'requests/*/response.json'))
            elif logical_summary_key and latest_result.get(logical_summary_key) is not None:
                evidence[key], evidence_sources[key] = latest_result[logical_summary_key], latest_result['path']
            elif latest_result.get(summary_key) is not None:
                evidence[key], evidence_sources[key] = latest_result[summary_key], latest_result['path']
            else:
                evidence[key], evidence_sources[key] = None, None
        row = dict(task=task, worker_id=worker_id, gpu=worker['gpu'], codex_thread_id=owner.get('thread_id'),
                   scheduling_status=task_schedule.get('status', processes.get(worker_id, processes.get(task, {})).get('status', 'unknown')),
                   preflight_blocked_reason=worker.get('blocked_reason') if not worker.get('runnable', True) else None,
                   blocked_before_start=len(expected) if not worker.get('runnable', True) else 0,
                   layout_execution_status=task_schedule.get('layouts', []),
                   active_run=read(folder / 'active_run.json'),
                   codex_pid=owner.get('pid'), codex_returncode=owner.get('returncode'),
                   supervisor_ready=owner.get('ready', False),
                   latest_codex_event_age_s=round(now-event_path.stat().st_mtime, 1) if event_path and event_path.exists() else None,
                   controller_status=state.get('status'), episode=request.get('episode'), step=request.get('step'),
                   request_id=request.get('request_id'), waiting=waiting,
                   pending_seconds=round(now-state.get('updated_at', now), 1) if waiting else 0,
                   **evidence, recovery_evidence_sources=evidence_sources,
                   unlimited_recovery=unlimited_recovery,
                   allow_agent_termination=(request.get('allow_agent_termination') if request
                                            else latest_result.get('allow_agent_termination')),
                   runtime_control_authorization=(request.get('runtime_control_authorization') if request
                                                  else latest_result.get('runtime_control', {}).get('authorization_id')),
                   live_recovery_extension=request.get('live_recovery_extension'),
                   step_budget=episode_budget(request, read(folder / 'run_manifest.json', {})) if request
                               else latest_result.get('step_budget', episode_budget({}, read(folder / 'run_manifest.json', {}))),
                   native_takeover_count=request.get('native_takeover_count'), native_retries=request.get('native_retries'),
                   active_worker_session=str(folder), source_sessions=list(map(str, source_folders)),
                   execution_history=execution_history,
                   worker_returncode=rc, expected_layouts=expected, missing_layouts=missing,
                   duplicate_layouts=duplicates, unexpected_attempts=unexpected, episodes=results, attempts=attempts,
                   attempts_started=len(attempts), attempts_finished=len(results), summary_count=len(results),
                   planned_episodes=planned, requested_episodes=planned, tested_episodes=tested,
                   finished_episodes=tested, completed_episodes=tested, success_episodes=successes,
                   zero_intervention_successes=counts['zero_intervention_success'], assisted_successes=counts['assisted_success'],
                   success_intervention_unknown=counts['success_intervention_unknown'],
                   failure_episodes=failures, official_failures=counts['official_failure'],
                   agent_aborted_failures=counts['agent_aborted_failure'], blocked_episodes=blocked,
                   interface_blocked_episodes=counts['blocked'], infrastructure_errors=counts['infrastructure_error'],
                   ambiguous_episodes=counts['ambiguous'], unclassified_episodes=counts['unclassified'],
                   started_episodes=len(started), running_episodes=len(running),
                   interrupted_episodes=len(interrupted),
                   not_started_episodes=planned-len(started), not_run_episodes=planned-len(started),
                   layout_outcomes=classifications,
                   success_rate_planned=successes/planned if planned else None,
                   success_rate_finished=successes/tested if tested else None,
                   fully_evaluated=not missing and not duplicates and not unexpected and rc == 0
                   and all(category in {'zero_intervention_success', 'assisted_success', 'success_intervention_unknown', 'official_failure'}
                           for category in classifications.values()))
        if duplicates: alerts.append(dict(task=task, reason='duplicate_layout_attempts', layouts=duplicates))
        if unexpected: alerts.append(dict(task=task, reason='unexpected_layout_or_seed', attempts=unexpected))
        if counts['unclassified']: alerts.append(dict(task=task, reason='unclassified_summary'))
        if interrupted: alerts.append(dict(task=task, reason='started_layout_without_summary_after_exit', layouts=interrupted))
        if waiting and row['pending_seconds'] > 300: alerts.append(dict(task=task, reason='pending_decision_over_300s'))
        if state.get('status') == 'paused_for_main':
            alerts.append(dict(task=task, reason='conditioning_paused_for_main',
                               pause_id=state.get('pause_id'), error=state.get('error')))
        if owner.get('returncode') is not None and rc is None: alerts.append(dict(task=task, reason='supervisor_exited_while_worker_active'))
        if rc not in (None, 0): alerts.append(dict(task=task, reason='worker_error', returncode=rc))
        if row['preflight_blocked_reason']:
            alerts.append(dict(task=task, reason=row['preflight_blocked_reason'], not_run_episodes=row['blocked_before_start']))
        rows.append(row)
    aggregate_keys = ('planned_episodes', 'requested_episodes', 'tested_episodes', 'finished_episodes',
                      'completed_episodes', 'success_episodes', 'zero_intervention_successes', 'assisted_successes',
                      'success_intervention_unknown', 'failure_episodes', 'official_failures', 'agent_aborted_failures',
                      'blocked_episodes', 'interface_blocked_episodes', 'infrastructure_errors', 'running_episodes',
                      'interrupted_episodes', 'not_started_episodes', 'not_run_episodes', 'ambiguous_episodes',
                      'unclassified_episodes', 'attempts_started', 'attempts_finished', 'summary_count', 'blocked_before_start')
    return dict(updated_at=now, job_id=config.get('job_id'), monitor='status collection only; no robot commands or automatic recovery',
                count_definition='Episode counts are distinct planned seed/layouts; tested/finished means a summary exists, not success. Duplicate attempts are ambiguous and never credited as success.',
                supervision_session_id=supervisors.get('supervision_session_id'), tasks=rows, alerts=alerts,
                **{key: sum(r[key] for r in rows) for key in aggregate_keys})


def output_basename(batch, output_name=None):
    name = output_name or ('monitor_runtime_status' if (batch / 'runtime_control.json').exists() else 'monitor_status')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
        raise ValueError('Monitor output name must be a simple filename stem')
    return name


def save(batch, value, output_name=None):
    output_name = output_basename(batch, output_name)
    extensions = read(batch / 'recovery_extensions.json', {})
    for row in value['tasks']:
        ext = row.get('live_recovery_extension')
        if ext:
            native_request = row['recovery_evidence_sources']['takeover_count']['native_request']
            extensions[ext['authorization_id']] = dict(**ext, task=row['task'], episode=row['episode'],
                episode_path=str(Path(native_request).parents[2]), logical_takeovers=row['takeover_count'],
                logical_retries=row['retries'], native_takeovers=row['native_takeover_count'],
                native_retries=row['native_retries'], updated_at=value['updated_at'],
                count_basis='published responses; rejected direct actions can reserve an attempt; execution in native interventions log')
    if extensions:
        path = batch / 'recovery_extensions.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(extensions, ensure_ascii=False, indent=2)+'\n')
        tmp.replace(path)
    target = batch / (output_name + '.json')
    tmp = target.with_suffix('.tmp'); tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n'); tmp.replace(target)
    lines = ['# 并行评测状态', '',
             f"Job {value['job_id']}；计划 {value['planned_episodes']} 局，已测/已结束 {value['tested_episodes']} 局，官方成功 {value['success_episodes']} 局。", '',
             '已测指不同计划布局已产生终态记录，包含失败与阻塞，不代表成功。成功率分别以计划局数、已测局数为分母。', '',
             '| 任务 | GPU | 调度状态 | 计划 | 已测 | 成功（零干预/辅助/待核） | 失败（官方/接管结束） | 阻塞（接口/设施） | 运行中 | 未运行 | 结果待核 | 成功/计划 | 成功/已测 | 当前局/步 | 实际单局上限 | 恢复/接管次数上限 | 等待秒 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in value['tasks']:
        unresolved = r['ambiguous_episodes'] + r['unclassified_episodes'] + r['interrupted_episodes']
        label = f"{r['task']} ({r['worker_id'].split(':layout_')[-1]})" if ':layout_' in r['worker_id'] else r['task']
        lines.append(f"| {label} | {r['gpu']} | {r['scheduling_status']} | {r['planned_episodes']} | {r['tested_episodes']} | "
                     f"{r['success_episodes']}（{r['zero_intervention_successes']}/{r['assisted_successes']}/{r['success_intervention_unknown']}） | "
                     f"{r['failure_episodes']}（{r['official_failures']}/{r['agent_aborted_failures']}） | "
                     f"{r['blocked_episodes']}（{r['interface_blocked_episodes']}/{r['infrastructure_errors']}） | "
                     f"{r['running_episodes']} | {r['not_run_episodes']} | {unresolved} | "
                     f"{r['success_episodes']}/{r['planned_episodes']} | {str(r['success_episodes'])+'/'+str(r['tested_episodes']) if r['tested_episodes'] else '—'} | "
                     f"{r['episode']} / {r['step']} | "
                     f"{str(r['step_budget']['step_limit'])+'（×'+str(r['step_budget']['step_limit_multiplier'])+'）' if r['step_budget']['step_limit'] is not None else '待加载'} | "
                     f"{'无限/无限' if r['unlimited_recovery'] else str(r['max_retries_per_subgoal'])+'/'+str(r['max_takeovers'])} | {r['pending_seconds']} |")
    groups = {}
    for row in value['tasks']:
        for result in row['episodes']:
            if not result['planned_layout'] or result['layout_id'] in row['duplicate_layouts']:
                continue
            multiplier = result['step_budget']['step_limit_multiplier']
            key = 'unknown' if multiplier is None else str(multiplier)
            group = groups.setdefault(key, dict(tested=0, success=0))
            group['tested'] += 1
            group['success'] += int(classify(result) in {'zero_intervention_success', 'assisted_success', 'success_intervention_unknown'})
    lines.extend(['', '实际步数预算分组（已结束局）：' + json.dumps(groups, ensure_ascii=False),
                  '倍率不同属于不同评测条件；旧局保留实际预算，不因新版默认×3而追改。'])
    lines.extend(['', f"原始 attempts：开始 {value['attempts_started']}，summary {value['summary_count']}；重复布局不重复计数，异常布局/seed 不计入计划结果。",
                  '', '告警：' + json.dumps(value['alerts'], ensure_ascii=False), '',
                  '此监控只读状态，不提交机器人动作。各 Codex 会话负责独立监督。'])
    for row in value['tasks']:
        if row.get('unlimited_recovery'):
            lines.append(f"\n{row['task']} 已授权恢复/接管次数无上限；逻辑接管事件 {row['takeover_count']}、恢复尝试 {row['retries']}。"
                         '子会话不得提出终止；实际单局动作总上限仍按本局记录。兼容传输的 correction/replan 不另计为逻辑修正。')
        ext = row.get('live_recovery_extension')
        if ext:
            lines.append(f"\n{row['task']} 显式兼容接管：历史审计 {ext['audited_attempts_before']} 次；"
                         f"追加 {ext['additional_attempts']} 次，已提交 {ext['additional_used']} 次，剩余 {ext['additional_remaining']} 次。"
                         f"逻辑接管事件 {row['takeover_count']}；旧桥原始接管事件 {row['native_takeover_count']}、重试 {row['native_retries']}，分别保留。")
    (batch / (output_name + '.md')).write_text('\n'.join(lines)+'\n')


def source_signature():
    source = Path(__file__).resolve()
    return source.stat().st_mtime_ns, hashlib.sha256(source.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--session', type=Path, required=True); p.add_argument('--watch', action='store_true')
    p.add_argument('--output-name', help='Output filename stem; runtime authorization defaults to monitor_runtime_status')
    args=p.parse_args()
    batch=args.session.resolve()
    signature = source_signature()
    output_name = output_basename(batch, args.output_name)
    pid_name = output_name.removesuffix('_status') + '_pid.json'
    pid_path = batch / pid_name
    pid_path.write_text(json.dumps(dict(pid=os.getpid(), source=str(Path(__file__).resolve()), argv=sys.argv,
                                        started_at=time.time(), status='running'))+'\n')
    try:
        while True:
            if source_signature() != signature:
                os.execv(sys.executable, [sys.executable, *sys.argv])
            value=snapshot(batch); save(batch,value,output_name)
            if not args.watch or all(r['worker_returncode'] is not None or r['preflight_blocked_reason'] for r in value['tasks']):
                print(json.dumps(dict(completed=value['completed_episodes'], successes=value['success_episodes'], alerts=value['alerts']))); return
            time.sleep(10)
    finally:
        owner = read(pid_path, {})
        if owner.get('pid') == os.getpid():
            owner.update(status='exited', ended_at=time.time())
            pid_path.write_text(json.dumps(owner)+'\n')


if __name__ == '__main__':
    main()
