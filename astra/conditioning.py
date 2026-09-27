"""Validated, versioned visual subgoals supplied before low-policy inference."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


class ConditioningError(ValueError):
    """Reject a condition without executing actions or ending the episode."""


def mode_for(manifest):
    mode = manifest.get('conditioning_mode', 'pure_l3')
    if mode not in {'pure_l3', 'l3_l2'}:
        raise ConditioningError(f'Unsupported conditioning_mode: {mode}')
    return mode


def text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ConditioningError(f'{field} must be a nonempty string')
    return value.strip()


def l3_from_obs(obs):
    for key in ('task_instruction', 'instruction', 'instructions'):
        if key in obs:
            value = obs[key]
            if isinstance(value, np.ndarray):
                value = value.reshape(-1).tolist()
            if isinstance(value, (tuple, list)):
                value = value[0] if value else None
            if isinstance(value, bytes):
                value = value.decode()
            return text(value, 'L3 instruction')
    raise ConditioningError('L3 instruction missing from current observation')


def head_image(images):
    for camera in ('cam_head', 'cam_high'):
        if camera in images:
            return camera, Path(images[camera]).resolve()
    raise ConditioningError('Current head-camera image missing')


def pixel_hash(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def read_model_receipt():
    """The standard WS client has no generic RPC; read its server's atomic receipt."""
    log = os.environ.get('ASTRA_CONDITIONING_AUDIT_LOG')
    if not log:
        raise ConditioningError('ASTRA_CONDITIONING_AUDIT_LOG missing')
    try:
        return json.loads(Path(log + '.receipt.json').read_text())
    except (OSError, ValueError) as exc:
        raise ConditioningError(f'Model conditioning receipt unavailable: {exc}') from exc


def validate_new(request, data, subgoal):
    if not isinstance(data, dict):
        raise ConditioningError('conditioning must be a JSON object')
    result = {key: text(data.get(key), key) for key in
              ('subgoal', 'l2_instruction', 'cue_rgb_path', 'box_source', 'evidence')}
    if result['subgoal'] != subgoal:
        raise ConditioningError('Condition subgoal differs from the decision segment')
    if result['box_source'] not in {'astra_visual', 'validated_visual_locator'}:
        raise ConditioningError('Only observed RGB based box sources are allowed')
    camera, expected = head_image(request['observation']['images'])
    if Path(result['cue_rgb_path']).resolve() != expected:
        raise ConditioningError('New cue must be the pending request head-camera image')
    try:
        with Image.open(expected) as im:
            if im.mode != 'RGB':
                raise ConditioningError('Cue must be an unboxed RGB image')
            rgb = np.array(im, dtype=np.uint8)
    except OSError as exc:
        raise ConditioningError(f'Cannot read cue: {exc}') from exc
    height, width = rgb.shape[:2]
    try:
        raw = np.asarray(data['l2_bboxes_xyxy'])
        if (raw.dtype.kind not in 'iuf'
                or (raw.shape != (0,) and (raw.ndim != 2 or raw.shape[1] != 4))):
            raise ValueError('numeric [N,4] required')
        boxes = np.asarray(raw, dtype=np.float32).reshape(-1, 4)
    except (KeyError, ValueError, TypeError) as exc:
        raise ConditioningError('l2_bboxes_xyxy must be a numeric [N,4] array') from exc
    if len(boxes):
        if not (np.isfinite(boxes).all() and (boxes[:, 0] >= 0).all()
                and (boxes[:, 1] >= 0).all() and (boxes[:, 2] <= width).all()
                and (boxes[:, 3] <= height).all() and (boxes[:, 2] > boxes[:, 0]).all()
                and (boxes[:, 3] > boxes[:, 1]).all()):
            raise ConditioningError('Boxes outside cue pixel bounds or degenerate/nonfinite')
        result['unboxed_reason'] = None
    else:
        result['unboxed_reason'] = text(data.get('unboxed_reason'), 'unboxed_reason')
    if request.get('conditioning'):
        result['change_reason'] = text(data.get('change_reason'), 'change_reason')
    else:
        result['change_reason'] = data.get('change_reason') or 'initial visually reviewed subgoal'
    result.update(cue_rgb_path=str(expected), camera=camera, width=width, height=height,
                  cue_pixel_sha256=pixel_hash(rgb), l2_bboxes_xyxy=boxes.tolist())
    return result


def validate_decision(request, command):
    mode = command['mode']
    new, version = command.get('conditioning'), command.get('conditioning_version')
    active = request.get('conditioning')
    if request.get('conditioning_mode', 'pure_l3') != 'l3_l2':
        if mode == 'condition' or new is not None or version is not None:
            raise ConditioningError('L2 conditions require a prepared l3_l2 runtime')
        return
    if command.get('instruction') is not None:
        raise ConditioningError('Keep L3 unchanged; update l2_instruction through conditioning')
    if request['state'] == 'PREPARE_CONDITIONING' and mode != 'condition':
        raise ConditioningError('First supply condition; no low-policy inference or actions yet')
    if mode == 'condition' and request['state'] != 'PREPARE_CONDITIONING':
        raise ConditioningError('Use replan or hand_back to change an active condition')
    if mode == 'condition' and not command.get('scene_review'):
        raise ConditioningError('Initial conditioning requires a current scene review')
    if mode in {'condition', 'replan', 'hand_back'}:
        if mode == 'replan' and request['state'] == 'TAKEOVER':
            raise ConditioningError('Use hand_back to leave takeover')
        if mode == 'hand_back' and (request['state'] != 'TAKEOVER' or not command.get('recovery_evidence')):
            raise ConditioningError('Hand-back requires takeover and recovery evidence')
        if (new is None) == (version is None):
            raise ConditioningError('Supply exactly one new conditioning or conditioning_version')
        if new is not None:
            validate_new(request, new, command['subgoal'])
        elif not active or version != active['conditioning_version'] or command['subgoal'] != active['subgoal']:
            raise ConditioningError('Stale conditioning version or mismatched subgoal')
    elif new is not None or version is not None:
        raise ConditioningError('Conditions may change only through condition/replan/hand_back')
    if mode in {'approve', 'correct'} and (not active or command['subgoal'] != active['subgoal']):
        raise ConditioningError('Low-policy action segment must match its conditioning subgoal')
    if mode in {'approve', 'correct'} and request['state'] == 'TAKEOVER':
        raise ConditioningError('Hand back before approving or correcting low policy')


def freeze_condition(folder, request, command, obs, index):
    data = validate_new(request, command['conditioning'], command['subgoal'])
    rgb = np.asarray(obs['vision'][data['camera']]['color'])[..., :3].copy()
    if pixel_hash(rgb) != data['cue_pixel_sha256']:
        raise ConditioningError('Cue file differs from the actual current observation')
    l3 = l3_from_obs(obs)
    previous = request.get('conditioning')
    if previous and l3 != previous['l3_instruction']:
        raise ConditioningError('L3 changed within the episode')
    version = f"{request['episode']}:c{index}"
    target = Path(folder) / 'conditioning' / f'{index:06d}.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        Image.fromarray(rgb).save(stream, format='PNG')
    data.update(conditioning_version=version, start_step=request['step'],
                created_request_id=request['request_id'], source_cue_rgb_path=data['cue_rgb_path'],
                cue_rgb_path=str(target), cue_file_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                l3_instruction=l3, schema_version=1)
    return data


def build_observation(obs, condition, request_id, inference_id):
    conditioned = deepcopy(obs)
    path = Path(condition['cue_rgb_path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != condition['cue_file_sha256']:
        raise ConditioningError('Frozen cue was modified')
    with Image.open(path) as im:
        rgb = np.array(im, dtype=np.uint8)
    if pixel_hash(rgb) != condition['cue_pixel_sha256']:
        raise ConditioningError('Frozen cue pixels differ')
    conditioned.update(instruction=condition['l3_instruction'],
                       task_instruction=condition['l3_instruction'],
                       l2_instruction=condition['l2_instruction'], l2_visual_cue_rgb=rgb,
                       l2_bboxes_xyxy=np.asarray(condition['l2_bboxes_xyxy'], dtype=np.float32).reshape(-1, 4))
    conditioned['_astra_conditioning'] = dict(condition, request_id=request_id, inference_id=inference_id)
    return conditioned
