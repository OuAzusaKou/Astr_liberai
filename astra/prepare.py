"""Create an isolated runtime, model config and auditable demonstration index."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from astra.config import CHECKPOINT, CORE9, DATA, DEMO_TASK_INDEX, PROJECT, ROOT, RUN, SOURCE, RUNTIME_DEPS
from astra.runtime import fail_closed_eval_source, strict_eval_cli_source


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--tasks', nargs='+', default=CORE9)
    parser.add_argument('--eval-num', type=int, default=1)
    parser.add_argument('--demo-cache', type=Path)
    parser.add_argument('--waive-demo', action='store_true')
    parser.add_argument('--max-retries-per-subgoal', type=int, default=6)
    parser.add_argument('--max-takeovers', type=int, default=10)
    parser.add_argument('--step-limit-multiplier', type=int, default=3)
    parser.add_argument('--unlimited-recovery', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--allow-agent-termination', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.max_retries_per_subgoal < 1 or args.max_takeovers < 1:
        parser.error('Recovery limits must be positive integers')
    if args.step_limit_multiplier < 1:
        parser.error('Step limit multiplier must be a positive integer')
    session = args.session.resolve()
    session.mkdir(parents=True, exist_ok=False)
    for name in ('astra', 'scripts', 'tests'):
        shutil.copytree(ROOT / name, session / 'source_snapshot' / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'AGENTS.md', session / 'source_snapshot/AGENTS.md')
    (session / 'bin').mkdir()
    ffmpeg = Path('/share-2/home/vla/.local/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
    assert ffmpeg.is_file()
    (session / 'bin/ffmpeg').symlink_to(ffmpeg)
    runtime = session / 'runtime/RoboDojo'
    runtime.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '.git', '*.pyc', '.ruff_cache')
    # Copy executable/configuration layers; large immutable assets stay shared.
    for name in ('env', 'env_cfg', 'src', 'task', 'utils', 'scripts'):
        shutil.copytree(SOURCE / name, runtime / name, ignore=ignore)
    entrypoint = runtime / 'src/eval_client/main.py'
    entrypoint.write_text(strict_eval_cli_source(fail_closed_eval_source(entrypoint.read_text())))
    for name in ('Assets', 'third_party'):
        (runtime / name).symlink_to(SOURCE / name, target_is_directory=True)
    xpl = runtime / 'XPolicyLab'
    xpl.mkdir()
    for item in (SOURCE / 'XPolicyLab').iterdir():
        if item.is_file():
            shutil.copy2(item, xpl / item.name)
    for name in ('utils', 'client_server'):
        shutil.copytree(SOURCE / 'XPolicyLab' / name, xpl / name, ignore=ignore)
    policy = xpl / 'policy/AstraImageWAM'
    policy.mkdir(parents=True)
    (policy / '__init__.py').write_text('')
    (policy / 'deploy.py').write_text('from astra.bridge import eval_one_episode\n')
    (policy / 'deploy.yml').write_text('policy_name: AstraImageWAM\nprotocol: ws\neval_batch: false\n')
    (policy / 'model.py').write_text(
        'from astra.conditioned_model import Model\n')
    train = yaml.safe_load((RUN / 'config.yaml').read_text())
    assert train['model']['variant'] == 'klein-base-9b'
    assert train['data']['train']['dataset_dirs'] == [str(DATA)]
    assert train['data']['train']['train_episode_fraction'] == 1.0
    assert train['data']['train']['val_set_proportion'] == 0.0
    assert train['data']['train']['episode_manifest_path'] is None
    assert CHECKPOINT.is_file()
    for name in ('config.yaml', 'dataset_stats.json'):
        shutil.copy2(RUN / name, session / name)
    model_cfg = yaml.safe_load((PROJECT / 'policy_server/imagewam/xpolicylab_policy/deploy.yml').read_text())
    m = train['model']
    model_cfg.update(policy_name='AstraImageWAM', eval_batch=False,
                     project_root=str(PROJECT), checkpoint_path=str(CHECKPOINT),
                     config_path=str(RUN / 'config.yaml'), dataset_stats_path=str(RUN / 'dataset_stats.json'),
                     seed=0, model_seed=0, qwen_context_len=m['qwen_context_len'],
                     flux2_variant=m['variant'], flux2_src_path=m['flux2_src_path'],
                     flux2_model_path=m['flux2_model_path'], ae_model_path=m['ae_model_path'],
                     qwen3_model_spec=m['qwen3_model_spec'], robotwin_camera_layout='four_grid_native_352x480',
                     conditioning_mode='l3_l2', conditioning_schema_version=1,
                     conditioning_provider='astra_visual', replan_steps=16, num_inference_steps=10,
                     ws_ping_interval_s=None, ws_ping_timeout_s=None)
    for key in ('flux2_src_path', 'flux2_model_path', 'ae_model_path', 'qwen3_model_spec'):
        assert Path(model_cfg[key]).exists(), key
    (session / 'model.yaml').write_text(yaml.safe_dump(model_cfg, sort_keys=False))
    tasks = {r['task_index']: r['task'] for r in map(json.loads, (DATA / 'meta/tasks.jsonl').read_text().splitlines())}
    episodes = list(map(json.loads, (DATA / 'meta/episodes.jsonl').read_text().splitlines()))
    info = json.loads((DATA / 'meta/info.json').read_text())
    start, stop = map(int, info['splits']['train'].split(':'))
    demos = {}
    for task in args.tasks:
        if args.waive_demo:
            if task not in {'align_blocks', 'classify_objects_by_language', 'general_pickup'}:
                raise ValueError('Demo waiver is authorized only for the requested open tasks')
            demos[task] = dict(status='waived_by_user', task=task, episode_id=None,
                               authorization=('2026-09-21 user explicitly permits general_pickup without demonstrations in the Core9 restart'
                                              if task == 'general_pickup' else '2026-09-21 user permits requested open-category tasks without demonstrations'),
                               segments={'observe': {'source': 'current observation and task instruction'},
                                         'operate': {'source': 'current observation and task instruction'},
                                         'recover': {'source': 'current observation and task instruction'}},
                               selection_reason='No matching training demonstration; explicit user exception',
                               current_scene_differences='Must inspect every new episode before approving actions')
            write_json(session / 'demos' / task / 'demo_context.json', demos[task])
            continue
        if args.demo_cache:
            cached = args.demo_cache / 'demos' / task / 'demo_context.json'
            if cached.is_file():
                context = json.loads(cached.read_text())
                if context.get('status') == 'reviewed':
                    context['reused_from'] = str(cached)
                    context['current_scene_differences'] = 'Inspect current episode before first action; prior layout conclusions do not transfer'
                    demos[task] = context
                    write_json(session / 'demos' / task / 'demo_context.json', context)
                    continue
        if task not in DEMO_TASK_INDEX:
            demos[task] = {'status': 'blocked_missing_demo', 'reason': 'No exact semantic task in training metadata; a valid pickup segment must be reviewed first.'}
            continue
        instruction = tasks[DEMO_TASK_INDEX[task]]
        episode = next(r for r in episodes if instruction in r['tasks'] and start <= r['episode_index'] < stop)
        idx = episode['episode_index']
        videos = {key: str(DATA / info['video_path'].format(episode_chunk=idx // info['chunks_size'],
                  episode_index=idx, video_key=key)) for key, value in info['features'].items() if value['dtype'] == 'video' and key.endswith(('cam_high', 'cam_left_wrist', 'cam_right_wrist'))}
        for path in videos.values():
            assert Path(path).is_file(), path
        demos[task] = dict(status='awaiting_visual_review', dataset=str(DATA), episode_id=idx,
                           task=task, instruction=instruction, videos=videos, length=episode['length'],
                           fps=info['fps'], train_split=info['splits']['train'],
                           selection_reason='Exact task semantics; first training episode',
                           variant_difference='Random test layout; standard training demonstration' if task.endswith('_random') else None)
        write_json(session / 'demos' / task / 'demo_context.json', demos[task])
    write_json(session / 'demo_index.json', demos)
    stat = CHECKPOINT.stat()
    write_json(session / 'run_manifest.json', dict(
        run_name='astra_liberai_test', created_at=datetime.now(timezone.utc).isoformat(),
        status='prepared_runtime_unverified', source=str(SOURCE), project=str(PROJECT),
        checkpoint={'path': str(CHECKPOINT), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                    'fingerprint_type': 'path-size-mtime; not a content hash'},
        train_config_sha256=hashlib.sha256((RUN / 'config.yaml').read_bytes()).hexdigest(),
        tasks=args.tasks, initial_smoke_task=args.tasks[0], smoke_eval_num=args.eval_num,
        eval_num=args.eval_num, authorized_demo_waivers=args.tasks if args.waive_demo else [],
        layout_ids=list(range(args.eval_num)),
        seed=0, env_cfg='arx_x5', action_type='joint', replan_steps=16, inference_steps=10,
        conditioning_mode='l3_l2', conditioning_schema_version=1,
        conditioning_provider='astra_visual', high_level_conditioning_assistance=True,
        unlimited_recovery=args.unlimited_recovery, allow_agent_termination=args.allow_agent_termination,
        max_retries_per_subgoal=None if args.unlimited_recovery else args.max_retries_per_subgoal,
        max_takeovers=None if args.unlimited_recovery else args.max_takeovers,
        controller='interactive Codex via atomic JSON request/response; no unattended fallback',
        max_actions_per_decision=15, high_level_monitor_interval_actions=15,
        max_takeover_actions_per_decision=30, controller_timeout_s=900,
        isaac_infrastructure_auto_restart=True, isaac_infrastructure_restart_limit=None,
        step_limit_multiplier=args.step_limit_multiplier,
        step_limit_source='original task class limit times step_limit_multiplier; recorded per episode', requested_nodes=1, requested_gpus=1,
        gripper_handling='Native EvalEnv.take_action np.clip; no agent intervention for finite nominal-range overshoot',
        policy_runtime_deps=str(RUNTIME_DEPS), source_version='no git metadata; source_snapshot and SHA256',
        local_code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for subdir in ('astra', 'scripts') for p in (ROOT / subdir).glob('*') if p.is_file()},
        stagnation_check={'window_actions': 128, 'automatic_abort': False,
                          'method': 'agent assesses approach, grasp, alignment and object progress; no claimed pixel tracker'},
        completion_confirmation={'window_s': 1.0, 'clock': 'effective_simulation_time',
                                 'scope': 'visually completed final operation; not intermediate subgoals',
                                 'on_expiry': 'diagnose_and_adjust_plan', 'automatic_abort': False},
        observation_channels=['RGB', 'joint state', 'world end-effector state', 'official success/termination'],
        privileged_object_state=False, runtime=str(runtime)))
    print(session)


if __name__ == '__main__':
    main()
