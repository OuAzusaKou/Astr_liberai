"""Fail-closed, file-based action arbitration for an actively supervising agent.

No LLM is hidden in this module: Codex reads request images and writes a decision.
The simulator does not advance while waiting. Every decision is episode/request bound.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image
from astra.monitoring_config import monitoring_config, decision_action_limit

from astra.conditioning_pause import pause_for_main
from astra.conditioning import (ConditioningError, mode_for, validate_decision,
                                freeze_condition, build_observation, read_model_receipt)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append_event(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(plain(value), ensure_ascii=False, allow_nan=False) + '\n')


class AgentTerminationDisallowed(ValueError):
    """Rejected supervision command; this must never terminate an episode."""


def recovery_limits(config):
    unlimited = config.get('unlimited_recovery', False)
    allow_termination = config.get('allow_agent_termination', True)
    if type(unlimited) is not bool or type(allow_termination) is not bool:
        raise ValueError('Recovery/termination policy flags must be booleans')
    limits = {key: None if unlimited else config.get(key, default) for key, default in
              (('max_retries_per_subgoal', 6), ('max_takeovers', 10))}
    if not unlimited:
        for key, value in limits.items():
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
    return dict(**limits, unlimited_recovery=unlimited, allow_agent_termination=allow_termination)


def configure_step_limit(env, config):
    """Scale the task's original action budget once, preserving legacy manifests."""
    multiplier = config.get('step_limit_multiplier', 1)
    if type(multiplier) is not int or multiplier < 1:
        raise ValueError('step_limit_multiplier must be a positive integer')
    original = getattr(env, '_astra_original_step_limit', env.step_lim)
    if isinstance(original, bool) or not isinstance(original, (int, np.integer)) or original < 1:
        raise ValueError('Original task step limit must be a positive integer')
    original = int(original)
    env._astra_original_step_limit = original
    env.step_lim = original * multiplier
    return dict(original_step_limit=original, step_limit_multiplier=multiplier,
                step_limit=env.step_lim)


def batch_audit(request, response):
    interval = monitoring_config(request)['high_level_monitor_interval_actions']
    mode = response['mode']
    decision_interval = decision_action_limit(request, mode) if mode == 'takeover' else interval
    count = (response.get('execute', 1) if mode == 'approve' else
             len(response.get('actions', [])) if mode in {'correct', 'takeover'} else 0)
    reason = None
    if 0 < count < decision_interval:
        reason = response.get('short_batch_reason')
        if not reason:
            reason = ('remaining reviewed chunk has fewer actions than monitoring interval'
                      if mode == 'approve' and count == len(request['proposed_actions'])
                      else response['reason'])
    return dict(high_level_monitor_interval_actions=interval, decision_monitor_interval_actions=decision_interval, selected_actions=count,
                short_batch_reason=reason)


def decision_for(request, response):
    if response.get('request_id') != request['request_id']:
        raise ValueError('Stale or mismatched request_id')
    if response.get('mode') not in {'condition', 'approve', 'correct', 'takeover', 'hand_back', 'replan', 'abort', 'blocked'}:
        raise ValueError('Unknown decision mode')
    if response['mode'] in {'abort', 'blocked'} and request.get('allow_agent_termination', True) is False:
        raise AgentTerminationDisallowed('Agent termination is disabled; continue supervised recovery until an official terminal state')
    for key in ('reason', 'subgoal', 'demo_segment'):
        if not isinstance(response.get(key), str) or not response[key].strip():
            raise ValueError(f'Missing {key}')
    if response['demo_segment'] not in request['demo_segments']:
        raise ValueError('Subgoal must reference a reviewed demonstration segment')
    validate_decision(request, response)
    maximum = decision_action_limit(request, response['mode'])
    if response['mode'] == 'approve':
        count = response.get('execute', 1)
        if type(count) is not int or not 1 <= count <= min(maximum, len(request['proposed_actions'])):
            raise ValueError(f'Approve requires 1..{maximum} available actions')
    if response['mode'] in {'correct', 'takeover'}:
        if not isinstance(response.get('actions'), list) or not 1 <= len(response['actions']) <= maximum:
            raise ValueError(f'Correction/takeover requires 1..{maximum} explicit actions')
    if 'new_attempt' in response and (response['mode'] != 'takeover' or type(response['new_attempt']) is not bool):
        raise ValueError('new_attempt must be a boolean on a takeover decision')
    if response['mode'] == 'takeover' and request.get('state') == 'TAKEOVER' and 'new_attempt' not in response:
        raise ValueError('Continuous takeover requires explicit new_attempt=true or false')
    if request.get('step') == 0 and 'worker_id' in request and response['mode'] in {'approve', 'correct', 'takeover'}:
        if not response.get('scene_review'):
            raise ValueError('First action requires an explicit review of this episode scene')
    if response['mode'] in {'abort', 'blocked'}:
        allowed = {'infrastructure_or_interface'} if response['mode'] == 'blocked' else {'unrecoverable_with_evidence', 'recovery_budget_exhausted'}
        if response.get('termination_condition') not in allowed or not response.get('termination_evidence'):
            raise ValueError('Stopping requires a documented termination condition; stagnation alone is insufficient')
        if response.get('termination_condition') == 'recovery_budget_exhausted':
            limits = recovery_limits(request)
            exhausted = ((limits['max_takeovers'] is not None
                          and request.get('takeover_count', 0) >= limits['max_takeovers'])
                         or (limits['max_retries_per_subgoal'] is not None
                             and request.get('retries', {}).get(response['subgoal'], 0) >= limits['max_retries_per_subgoal']))
            if not exhausted:
                raise ValueError('Recovery budget has not been exhausted')
    return response


def validated_action(env, action, direct=False):
    action = {key: np.asarray(value, dtype=np.float64) for key, value in action.items()}
    env.validate_action_dict(action)
    if not all(np.isfinite(v).all() for v in action.values()):
        raise ValueError('Non-finite action')
    manager = env.robot_manager
    targets = [r for r in manager.robot_list if r.type == 'target']
    joint_keys = {manager.process_name(r.arm_name) for r in targets}
    gripper_keys = {manager.process_name(r.gripper_name) for r in targets}
    ee_keys = {f'{r.arm_name.split("_")[0]}_ee_pose' for r in targets}
    if set(action) not in (joint_keys | gripper_keys, ee_keys | gripper_keys):
        raise ValueError('Action must specify both arms and grippers, with one coordinate representation')
    # Finite normalized gripper overshoot is handled by native EvalEnv.take_action.
    # Preserve the raw proposal; it is not an agent correction.
    if set(action) == ee_keys | gripper_keys:
        converted = {key: action[key] for key in gripper_keys}
        for robot in targets:
            key = f'{robot.arm_name.split("_")[0]}_ee_pose'
            if not np.isclose(np.linalg.norm(action[key][3:]), 1, atol=1e-3):
                raise ValueError('EE quaternion must be normalized, wxyz, in world coordinates')
            # Resolve both IK solutions before moving either arm. No silent IK failures.
            result = manager.solve_ik(target_pose=action[key].tolist(), env_idx=0, robot=robot)
            if result['status'] != 'Success':
                raise ValueError(f'IK failed: {key}')
            converted[manager.process_name(robot.arm_name)] = np.asarray(result['joint_value'])
        action = converted
    for robot in targets:
        key = manager.process_name(robot.arm_name)
        articulation = manager.robot_key[manager.robot_list.index(robot)]
        limits = articulation.data.soft_joint_pos_limits[0, robot.arm_joint_indices].detach().cpu().numpy()
        if not np.isfinite(action[key]).all() or (action[key] < limits[:, 0]).any() or (action[key] > limits[:, 1]).any():
            raise ValueError(f'Joint limit violation: {key}')
        if direct:
            current = np.asarray(manager.get_joint(robot, env_idx_list=[0])[0])
            if np.max(np.abs(action[key] - current)) > 0.15:
                raise ValueError('Direct action exceeds 0.15 rad per-step joint change; subdivide trajectory')
    env.validate_action_dict(action)
    return action


def save_observation(folder, step, obs):
    record = {k: plain(v) for k, v in obs.items() if k != 'vision'}
    record['images'] = {}
    for camera, values in obs['vision'].items():
        if 'color' not in values:
            continue
        rgb = np.asarray(values['color'])
        if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4) or rgb.dtype != np.uint8:
            raise ValueError(f'Unsupported RGB shape/type for {camera}: {rgb.shape}, {rgb.dtype}')
        path = folder / 'rollout' / f'{step:06d}_{camera}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb[..., :3]).save(path)
        record['images'][camera] = str(path)
    atomic_json(folder / 'rollout' / f'{step:06d}.json', record)
    return record


def eval_one_episode(TASK_ENV, model_client):
    env = TASK_ENV
    if env.num_envs != 1:
        raise RuntimeError('Interactive smoke validation supports one environment only')
    session = Path(os.environ['ASTRA_SESSION'])
    demo_path = session / 'demos' / env.task_name / 'demo_context.json'
    demo = json.loads(demo_path.read_text())
    manifest_path = session / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    conditioning_mode = mode_for(manifest)
    l2_enabled = conditioning_mode == 'l3_l2'
    condition, condition_count = None, 0
    conditioning_summary = dict(conditioning_mode=conditioning_mode,
                                high_level_conditioning_assistance=l2_enabled)
    limits = recovery_limits(manifest)
    monitoring = monitoring_config(manifest)
    step_limits = configure_step_limit(env, manifest)
    waived = demo.get('status') == 'waived_by_user' and env.task_name in manifest.get('authorized_demo_waivers', [])
    if (demo.get('status') != 'reviewed' and not waived) or not demo.get('segments'):
        raise RuntimeError('No visually reviewed demonstration context; refusing rollout')
    episode = f'layout_{env.env_seeds[0]}_{uuid.uuid4().hex[:8]}'
    folder = session / env.task_name / f'seed_{env.eval_seed}' / episode
    folder.mkdir(parents=True)
    atomic_json(folder / 'demo_context.json', demo)
    log = folder / 'interventions.jsonl'
    def reset_policy():
        while True:
            try:
                model_client.call(func_name='reset')
                return
            except Exception as exc:
                if not l2_enabled:
                    raise
                pause_for_main(session, folder, int(env.take_action_cnt[0]), f'{episode}:reset', exc)
    reset_policy()
    actions, takeover, takeovers, retries = [], False, 0, {}
    takeover_subgoal = None
    correction_count = 0
    request_index, replan_index, inference_attempt = 0, 0, 0
    instruction = None
    started = time.time()
    failure = None
    rejected = None
    try:
        while not env.is_episode_end():
            obs = env.get_obs()
            step = int(env.take_action_cnt[0])
            observation = save_observation(folder, step, obs)
            request_id = f'{episode}:{request_index}'
            needs_condition = l2_enabled and condition is None
            if not actions and not takeover and not needs_condition:
                inference_attempt += 1
                try:
                    inference_id = f'{episode}:inference:{inference_attempt}'
                    conditioned = (build_observation(obs, condition, request_id, inference_id)
                                   if l2_enabled else deepcopy(obs))
                    if instruction is not None:
                        conditioned['instruction'] = instruction
                    payload = f'{env.eval_seed}:{env.env_seeds[0]}:{replan_index}'.encode()
                    conditioned['_imagewam_request_seed'] = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'little') & ((1 << 63) - 1)
                    model_client.call(func_name='update_obs', obs=conditioned)
                    actions = list(model_client.call(func_name='get_action'))
                    if l2_enabled:
                        receipt = read_model_receipt()
                        expected = dict(inference_id=inference_id, request_id=request_id,
                                        conditioning_version=condition['conditioning_version'],
                                        actual_conditioning_mode='l3_l2',
                                        actual_instruction=f"{condition['l3_instruction']}\nCurrent subtask instruction: {condition['l2_instruction']}",
                                        actual_cue_pixel_sha256=condition['cue_pixel_sha256'],
                                        actual_bboxes_xyxy=condition['l2_bboxes_xyxy'])
                        if (not isinstance(receipt, dict) or receipt.get('verified') is not True
                                or receipt.get('status') != 'inference_completed'
                                or any(plain(receipt.get(k)) != v for k, v in expected.items())):
                            raise ConditioningError('Model conditioning receipt missing or mismatched; no action released')
                        append_event(folder / 'inference_conditions.jsonl', dict(
                            event='inference_verified', step=step, condition=condition, receipt=receipt))
                    replan_index += 1
                    if not actions:
                        raise RuntimeError('Empty low-policy chunk')
                except Exception as exc:
                    if not l2_enabled:
                        raise
                    actions = []
                    pause_for_main(session, folder, step, request_id, exc)
                    reset_policy()
                    continue
            request = dict(request_id=request_id, task=env.task_name, seed=env.eval_seed,
                           episode=episode, step=step, **step_limits,
                           state='PREPARE_CONDITIONING' if needs_condition else ('TAKEOVER' if takeover else 'LOW_POLICY'),
                           conditioning_mode=conditioning_mode, conditioning=condition,
                           demo_context=str(demo_path), demo_segments=list(demo['segments']),
                           observation=observation, proposed_actions=actions,
                           rejected_action=rejected, takeover_count=takeovers, retries=retries,
                           active_recovery_subgoal=takeover_subgoal,
                           **limits, **monitoring,
                           worker_id=session.name, demo_waived=waived,
                           stagnation_window_actions=128, stagnation_is_terminal=False)
            request_dir = folder / 'requests' / f'{request_index:06d}'
            request_file = request_dir / 'request.json'
            response_file = request_dir / 'response.json'
            atomic_json(request_file, request)
            atomic_json(session / 'controller_status.json', dict(status='waiting_for_agent', request=str(request_file), response=str(response_file), updated_at=time.time()))
            deadline = time.monotonic() + float(os.environ.get('ASTRA_CONTROLLER_TIMEOUT', '900'))
            while not response_file.exists():
                if time.monotonic() > deadline:
                    error = TimeoutError('Agent decision timeout; no unsupervised fallback')
                    if not l2_enabled:
                        raise error
                    pause_for_main(session, folder, step, request_id, error)
                    atomic_json(session / 'controller_status.json', dict(status='waiting_for_agent', request=str(request_file), response=str(response_file), updated_at=time.time()))
                    deadline = time.monotonic() + float(os.environ.get('ASTRA_CONTROLLER_TIMEOUT', '900'))
                time.sleep(0.2)
            submitted = json.loads(response_file.read_text())
            request_index += 1
            try:
                command = decision_for(request, submitted)
                if l2_enabled and command.get('conditioning') is not None:
                    next_condition = freeze_condition(folder, request, command, obs, condition_count + 1)
                    condition = next_condition
                    condition_count += 1
                    atomic_json(folder / 'conditioning' / f'{condition_count:06d}.json', condition)
                    append_event(log, dict(event='conditioning_updated', time=time.time(),
                                          request_id=request_id, step=step, **condition))
            except ValueError as exc:
                if not l2_enabled and not isinstance(exc, (AgentTerminationDisallowed, ConditioningError)):
                    raise
                rejected = str(exc)
                append_event(log, dict(event='decision_rejected', request_id=request_id,
                                      step=step, reason=rejected, submitted_command=submitted,
                                      control_state=request['state'], **limits))
                continue
            mode = command['mode']
            append_event(log, dict(event='decision', time=time.time(), step=step,
                                  task=env.task_name, seed=env.eval_seed, episode=episode,
                                  monitoring_batch=batch_audit(request, command), **command))
            if command.get('scene_review'):
                demo['current_scene_differences'] = command['scene_review']
                demo['scene_review_observation'] = str(folder / 'rollout' / f'{step:06d}.json')
                atomic_json(folder / 'demo_context.json', demo)
            if mode in {'abort', 'blocked'}:
                failure = ('blocked: ' if mode == 'blocked' else 'agent_aborted: ') + command['reason']
                break
            if mode == 'condition':
                actions = []
                rejected = None
                continue
            if mode == 'hand_back':
                if not takeover or not command.get('recovery_evidence'):
                    raise ValueError('Hand-back requires takeover and observed recovery evidence')
                takeover, actions = False, []
                takeover_subgoal = None
                instruction = None if l2_enabled else command.get('instruction')
                reset_policy()
                continue
            if mode == 'replan':
                if takeover:
                    raise ValueError('Use hand_back to leave takeover')
                correction_count += 1
                instruction = None if l2_enabled else command.get('instruction')
                actions = []
                reset_policy()
                continue
            if mode == 'approve':
                if takeover:
                    raise ValueError('Must hand back before approving low policy')
                selected = actions[:command.get('execute', 1)]
                actions = actions[len(selected):]
            else:
                if mode == 'correct' and takeover:
                    raise ValueError('Use takeover mode while agent owns control')
                if mode == 'takeover':
                    key = command['subgoal']
                    if takeover and key != takeover_subgoal and not command['new_attempt']:
                        rejected = 'Changed recovery subgoal requires an explicit new_attempt; no action executed'
                        append_event(log, dict(event='intervention_rejected', request_id=request_id,
                                              step=step, reason=rejected, subgoal=key))
                        continue
                    new_attempt = not takeover or command.get('new_attempt', False)
                    next_takeovers = takeovers + int(not takeover)
                    next_retries = retries.get(key, 0) + int(bool(new_attempt))
                    if ((limits['max_takeovers'] is not None and next_takeovers > limits['max_takeovers'])
                            or (limits['max_retries_per_subgoal'] is not None
                                and next_retries > limits['max_retries_per_subgoal'])):
                        rejected = 'Recovery budget exhausted; additional intervention rejected, episode remains available for supervised continuation'
                        append_event(log, dict(event='intervention_rejected', request_id=request_id,
                                              step=step, reason=rejected, subgoal=key,
                                              takeover_count=takeovers, retries=dict(retries), **limits))
                        continue
                    takeovers = next_takeovers
                    if new_attempt:
                        retries[key] = next_retries
                        append_event(log, dict(event='recovery_attempt_started', request_id=request_id,
                                              step=step, subgoal=key, attempt_number=next_retries,
                                              takeover_event=next_takeovers, reason=command['reason']))
                    takeover = True
                    takeover_subgoal = key
                else:
                    correction_count += 1
                selected = command['actions']
                actions = []
            rejected = None
            for action in selected:
                if env.is_episode_end():
                    break
                try:
                    final_action = validated_action(env, action, direct=mode != 'approve')
                except ValueError as exc:
                    rejected = str(exc)
                    actions = []
                    append_event(log, dict(event='action_rejected', step=int(env.take_action_cnt[0]), reason=rejected, action=action))
                    break
                before = int(env.take_action_cnt[0])
                env.take_action(final_action)
                next_obs = env.get_obs()
                save_observation(folder, int(env.take_action_cnt[0]), next_obs)
                append_event(log, dict(event='action', request_id=request_id, step=before,
                                      mode=mode, original_action=action, executed_action=final_action,
                                      conditioning_version=condition['conditioning_version'] if condition else None,
                                      subgoal=command['subgoal'], reason=command['reason']))
        ended = env.is_episode_end()
        success = bool(ended and env.success[0])
        blocked = failure is not None and failure.startswith('blocked: ')
        atomic_json(folder / 'summary.json', dict(status='blocked' if blocked else ('success' if success else 'failed'),
                    official_success=None if blocked else success, official_terminal=ended, failure=failure,
                    steps=int(env.take_action_cnt[0]), takeovers=takeovers, corrections=correction_count,
                    retries=retries, **limits, **step_limits, **conditioning_summary, **monitoring,
                    conditioning_versions=condition_count,
                    elapsed_s=time.time() - started, agent_decisions=request_index))
        if failure:
            raise RuntimeError(failure)
        atomic_json(session / 'controller_status.json', dict(status='episode_finished', folder=str(folder), success=success))
    except Exception as exc:
        atomic_json(folder / 'error.json', dict(error=type(exc).__name__, message=str(exc), steps=int(env.take_action_cnt[0])))
        if not (folder / 'summary.json').exists():
            atomic_json(folder / 'summary.json', dict(status='infrastructure_error',
                        official_success=None, official_terminal=None,
                        failure=f'{type(exc).__name__}: {exc}',
                        steps=int(env.take_action_cnt[0]), takeovers=takeovers,
                        corrections=correction_count, retries=retries,
                        **limits, **step_limits, **conditioning_summary, **monitoring,
                        conditioning_versions=condition_count,
                        elapsed_s=time.time() - started, agent_decisions=request_index))
        atomic_json(session / 'controller_status.json', dict(status='blocked_or_error', folder=str(folder), error=str(exc)))
        raise
