#!/usr/bin/env python3
"""Prepare an auditable Core9 queue without starting GPU or Codex processes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astra.config import CORE9

DEFAULT_DEMO_CACHE = ROOT / 'outputs/astra_liberai_test/core9_demo_cache_20260921'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def plan_workers(session, demo_cache, eval_num=3, waive_demo_tasks=(), tasks=CORE9,
                 task_layout_ids=None):
    """Keep every requested unit in the plan, including missing demonstrations."""
    if eval_num < 1:
        raise ValueError('eval_num must be positive')
    tasks = list(tasks)
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(CORE9):
        raise ValueError('tasks must be a nonempty unique subset of Core9')
    task_layout_ids = task_layout_ids or {}
    if set(task_layout_ids) - set(tasks):
        raise ValueError('Layout selections may only name requested tasks')
    waivers = set(waive_demo_tasks)
    if waivers - {'general_pickup'} or waivers - set(tasks):
        raise ValueError('This batch only authorizes an explicit general_pickup demonstration waiver')
    session, demo_cache = Path(session).resolve(), Path(demo_cache).resolve()
    workers = []
    for task in tasks:
        layouts = list(task_layout_ids.get(task, range(eval_num)))
        if (not layouts or len(set(layouts)) != len(layouts)
                or any(not isinstance(layout, int) or layout < 0 or layout >= eval_num for layout in layouts)):
            raise ValueError(f'{task}: layout IDs must be unique integers in [0, {eval_num})')
        cached = demo_cache / 'demos' / task / 'demo_context.json'
        reason = None
        context = None
        if task not in waivers:
            try:
                context = json.loads(cached.read_text())
            except FileNotFoundError:
                reason = 'No reviewed demonstration context exists in the selected cache'
            except (OSError, ValueError) as exc:
                reason = f'Cannot read demonstration context: {type(exc).__name__}: {exc}'
            if context is None and reason is None:
                reason = 'Demonstration context is null, not a reviewed JSON object'
            if context is not None:
                if not isinstance(context, dict):
                    reason = 'Demonstration context is not a JSON object'
                elif context.get('status') != 'reviewed':
                    reason = context.get('reason') or f'Demonstration status is {context.get("status")!r}, not reviewed'
                elif context.get('task') != task:
                    reason = 'Reviewed demonstration belongs to a different execution unit'
                elif not isinstance(context.get('segments'), dict) or not context['segments']:
                    reason = 'Demonstration context has no executable segment dictionary'
        runnable = reason is None
        blocked_reason = None if runnable else (
            'blocked_demo_decode' if isinstance(context, dict) and context.get('status') == 'blocked_demo_decode'
            else 'blocked_missing_demo')
        for layout in layouts:
            index = len(workers)
            worker_id = f'{task}:layout_{layout}'
            workers.append(dict(worker_id=worker_id, task=task,
                                session=str(session / 'workers' / f'w{index:02d}_{task}_layout_{layout}'),
                                gpu=None, layout_ids=[layout], runnable=runnable,
                                demo_waived=task in waivers, blocked_reason=blocked_reason,
                                blocked_detail=reason,
                                demo_context_path=None if task in waivers else str(cached), queue_index=index))
    return workers


def snapshot_source(session):
    for name in ('astra', 'scripts', 'tests'):
        shutil.copytree(ROOT / name, session / 'source_snapshot' / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'AGENTS.md', session / 'source_snapshot/AGENTS.md')


def prepare_batch(session, demo_cache=DEFAULT_DEMO_CACHE, eval_num=3, waive_demo_tasks=(),
                  tasks=CORE9, task_layout_ids=None, recovery_of=None, max_workers=None):
    session, demo_cache = Path(session).resolve(), Path(demo_cache).resolve()
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
        raise ValueError('max_workers must be a positive integer')
    workers = plan_workers(session, demo_cache, eval_num, waive_demo_tasks, tasks, task_layout_ids)
    waivers = [task for task in tasks if task in waive_demo_tasks]
    # Existing experiments must never be reset or silently overwritten.
    session.mkdir(parents=True, exist_ok=False)
    created_at = datetime.now(timezone.utc).isoformat()
    common = dict(run_name='astra_liberai_test', seed=0, eval_num=eval_num,
                  layout_ids=list(range(eval_num)), step_limit_multiplier=3,
                  max_retries_per_subgoal=None, max_takeovers=None,
                  unlimited_recovery=True, allow_agent_termination=False,
                  conditioning_mode='l3_l2', conditioning_schema_version=1,
                  conditioning_provider='astra_visual', high_level_conditioning_assistance=True,
                  max_actions_per_decision=15, high_level_monitor_interval_actions=15,
                  max_takeover_actions_per_decision=30,
                  isaac_infrastructure_auto_restart=True,
                  isaac_infrastructure_restart_limit=None,
                  authorized_demo_waivers=waivers,
                  model='gpt-6-astra', reasoning_effort='high')
    manifest = dict(**common, created_at=created_at, status='preparing', job_id=None,
                    max_workers=max_workers, concurrency=max_workers,
                    requested_nodes=1, requested_gpus=max_workers,
                  tasks=list(tasks), workers=workers, demo_cache=str(demo_cache),
                  planned_layouts={task: [layout for worker in workers if worker['task'] == task
                                          for layout in worker['layout_ids']] for task in tasks},
                  scheduling_unit='one_layout_per_worker',
                  recovery_of=str(Path(recovery_of).resolve()) if recovery_of else None)
    write_json(session / 'parallel_manifest.json', manifest)
    try:
        for worker in workers:
            folder = Path(worker['session'])
            if worker['runnable']:
                command = ['python3', '-m', 'astra.prepare', '--session', str(folder),
                           '--tasks', worker['task'], '--eval-num', str(eval_num),
                           '--step-limit-multiplier', '3', '--unlimited-recovery',
                           '--no-allow-agent-termination']
                command += ['--waive-demo'] if worker['demo_waived'] else ['--demo-cache', str(demo_cache)]
                subprocess.run(command, cwd=ROOT, check=True)
                runtime_manifest_path = folder / 'run_manifest.json'
                runtime_manifest = json.loads(runtime_manifest_path.read_text())
                runtime_manifest.update(layout_ids=worker['layout_ids'],
                                        eval_num=1,
                                        planned_episodes=len(worker['layout_ids']),
                                        recovery_of=str(Path(recovery_of).resolve()) if recovery_of else None)
                write_json(runtime_manifest_path, runtime_manifest)
                actual = json.loads((folder / 'demos' / worker['task'] / 'demo_context.json').read_text())
                if worker['demo_waived']:
                    if actual.get('status') != 'waived_by_user' or worker['task'] not in waivers:
                        raise RuntimeError(f'{worker["task"]}: prepared demonstration waiver is not recorded')
                elif actual.get('status') != 'reviewed':
                    raise RuntimeError(f'{worker["task"]}: prepared demonstration is not reviewed')
            else:
                folder.mkdir(parents=True, exist_ok=False)
                blocking = dict(task=worker['task'], status=worker['blocked_reason'],
                                reason=worker['blocked_detail'], runnable=False,
                                layout_ids=worker['layout_ids'], created_at=created_at,
                                execution_started=False, source_context=worker['demo_context_path'])
                context = dict(status=worker['blocked_reason'], task=worker['task'],
                               reason=worker['blocked_detail'], segments={}, episode_id=None,
                               source_context=worker['demo_context_path'])
                write_json(folder / 'demos' / worker['task'] / 'demo_context.json', context)
                write_json(folder / 'demo_index.json', {worker['task']: context})
                write_json(folder / 'blocking.json', blocking)
                blocked_manifest = dict(**common, tasks=[worker['task']], created_at=created_at,
                    status=worker['blocked_reason'], runnable=False,
                    blocked_reason=worker['blocked_reason'], blocked_detail=worker['blocked_detail'],
                    execution_started=False)
                blocked_manifest.update(layout_ids=worker['layout_ids'], eval_num=1, planned_episodes=1)
                write_json(folder / 'run_manifest.json', blocked_manifest)
        snapshot_source(session)
        statuses = [dict(task=w['task'], worker_id=w['worker_id'], session=w['session'], gpu=None, pid=None, returncode=None,
                         status='queued' if w['runnable'] else 'blocked', runnable=w['runnable'],
                         blocked_reason=w['blocked_reason'], layout_ids=w['layout_ids']) for w in workers]
        write_json(session / 'workers_status.json', statuses)
        write_json(session / 'codex_supervision/status.json', dict(
            status='prepared_not_started', model=common['model'], reasoning_effort=common['reasoning_effort'],
            workers=[dict(task=w['task'], worker_id=w['worker_id'], session=w['session'], pid=None, thread_id=None,
                          returncode=None, ready=False, status='queued' if w['runnable'] else 'blocked')
                     for w in workers]))
        manifest['status'] = 'prepared'
    except BaseException as exc:
        manifest.update(status='preparation_error', error=f'{type(exc).__name__}: {exc}')
        write_json(session / 'parallel_manifest.json', manifest)
        raise
    write_json(session / 'parallel_manifest.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--demo-cache', type=Path, default=DEFAULT_DEMO_CACHE)
    parser.add_argument('--eval-num', type=int, default=3,
                        help='Layout ID universe size; task layout selections must be below this value')
    parser.add_argument('--max-workers', type=int, required=True,
                        help='User-selected maximum simultaneous single-layout workers/Codex sessions')
    parser.add_argument('--tasks', nargs='+', choices=CORE9, default=CORE9)
    parser.add_argument('--task-layout', action='append', default=[], metavar='TASK=ID,ID',
                        help='Select specific layout IDs for a task; repeat for multiple tasks')
    parser.add_argument('--recovery-of', type=Path,
                        help='Prior session whose unfinished layouts this batch continues as new attempts')
    parser.add_argument('--waive-demo-tasks', nargs='+', default=[], choices=['general_pickup'],
                        help='Explicit current-batch user-authorized waiver; never assumed from cache')
    parser.add_argument('--validate-only', action='store_true', help='Inspect cache and print plan; write nothing')
    args = parser.parse_args(argv)
    if args.eval_num < 1:
        parser.error('--eval-num must be positive')
    if args.max_workers < 1:
        parser.error('--max-workers must be positive')
    task_layout_ids = {}
    try:
        for item in args.task_layout:
            task, value = item.split('=', 1)
            if task in task_layout_ids:
                raise ValueError(f'duplicate layout selection for {task}')
            task_layout_ids[task] = [int(layout) for layout in value.split(',')]
    except ValueError as exc:
        parser.error(f'invalid --task-layout: {exc}')
    if args.recovery_of and not args.recovery_of.is_dir():
        parser.error('--recovery-of must name an existing session directory')
    if args.validate_only:
        result = dict(status='validation_only', authorized_demo_waivers=args.waive_demo_tasks,
                      max_workers=args.max_workers,
                      workers=plan_workers(args.session, args.demo_cache, args.eval_num,
                                           args.waive_demo_tasks, args.tasks, task_layout_ids),
                      recovery_of=str(args.recovery_of.resolve()) if args.recovery_of else None)
    else:
        result = prepare_batch(args.session, args.demo_cache, args.eval_num, args.waive_demo_tasks,
                               args.tasks, task_layout_ids, args.recovery_of, args.max_workers)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
