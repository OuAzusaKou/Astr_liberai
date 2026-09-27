"""Verify and audit the conditions actually accepted by the model adapter.

This mixin deliberately has no model/checkpoint imports. It observes adapter
inputs and successful inference, not attention or learned use of a condition.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def pixel_sha256(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


class ConditioningAuditMixin:
    def _audit_enabled(self):
        return getattr(self, 'conditioning_mode', None) == 'l3_l2'

    def _write_conditioning_audit(self, record):
        value = os.environ.get('ASTRA_CONDITIONING_AUDIT_LOG')
        if not value:
            raise ValueError('l3_l2 requires ASTRA_CONDITIONING_AUDIT_LOG')
        path = Path(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
        # WsModelClient exposes only its fixed infer/reset protocol, so the
        # bridge reads this local receipt after get_action returns. Publish it
        # before returning the actions and never expose a partially written JSON.
        receipt_path = Path(str(path) + '.receipt.json')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                    dir=path.parent, prefix=receipt_path.name + '.', suffix='.tmp',
                    delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(record, stream, ensure_ascii=False, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, receipt_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _verify_actual_conditioning(self, metadata):
        if not self._audit_enabled():
            raise ValueError('Actual model conditioning_mode is not l3_l2')
        actual = getattr(self, 'last_obs', None)
        if not isinstance(actual, Mapping):
            raise ValueError('Model adapter did not retain conditioned observation')
        expected_instruction = (metadata['l3_instruction'].strip() + '\n'
                                'Current subtask instruction: ' + metadata['l2_instruction'].strip())
        if getattr(self, 'last_instruction', None) != expected_instruction:
            raise ValueError('Actual model instruction differs from L3/L2 metadata')
        cue = np.asarray(actual.get('visual_cue_rgb'))
        if cue.ndim != 3 or cue.shape[-1] != 3 or cue.dtype != np.uint8 or min(cue.shape) <= 0:
            raise ValueError('Actual model cue must be nonempty uint8 HWC RGB')
        digest = pixel_sha256(cue)
        if digest != metadata['cue_pixel_sha256']:
            raise ValueError('Actual model cue digest differs from conditioning metadata')
        boxes = np.asarray(actual.get('visual_cue_bboxes_xyxy'))
        expected_boxes = np.asarray(metadata['l2_bboxes_xyxy'], dtype=np.float32)
        if expected_boxes.size == 0:
            expected_boxes = expected_boxes.reshape(0, 4)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.dtype != np.float32:
            raise ValueError('Actual model boxes must be float32 [N,4]')
        if not np.isfinite(boxes).all() or not np.array_equal(boxes, expected_boxes):
            raise ValueError('Actual model boxes differ from conditioning metadata')
        height, width = cue.shape[:2]
        if not bool(((boxes[:, 0] >= 0) & (boxes[:, 1] >= 0)
                     & (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])
                     & (boxes[:, 2] <= width) & (boxes[:, 3] <= height)).all()):
            raise ValueError('Actual model boxes are outside cue image or degenerate')
        return dict(actual_conditioning_mode=self.conditioning_mode,
                    actual_instruction=self.last_instruction,
                    actual_cue_pixel_sha256=digest, actual_cue_shape=list(cue.shape),
                    actual_bboxes_xyxy=boxes.tolist(),
                    processing_stage='adapter_input_before_policy_preprocessing')

    def update_obs(self, obs):
        self._astra_conditioning_receipt = None
        self._astra_conditioning_metadata = None
        if not self._audit_enabled():
            if isinstance(obs, Mapping) and '_astra_conditioning' in obs:
                raise ValueError('Conditioning metadata received by a model not in l3_l2 mode')
            return super().update_obs(obs)
        if not isinstance(obs, Mapping) or not isinstance(obs.get('_astra_conditioning'), Mapping):
            raise ValueError('l3_l2 requires observation._astra_conditioning metadata')
        metadata = copy.deepcopy(dict(obs['_astra_conditioning']))
        required = ('inference_id', 'request_id', 'conditioning_version', 'l3_instruction',
                    'l2_instruction', 'cue_pixel_sha256', 'cue_rgb_path', 'subgoal',
                    'start_step', 'l2_bboxes_xyxy')
        if any(key not in metadata for key in required):
            raise ValueError('Incomplete _astra_conditioning metadata')
        for key in ('inference_id', 'request_id', 'l3_instruction', 'l2_instruction',
                    'cue_pixel_sha256', 'cue_rgb_path', 'subgoal'):
            if not isinstance(metadata[key], str) or not metadata[key].strip():
                raise ValueError(f'_astra_conditioning.{key} must be nonempty text')
        if isinstance(metadata['start_step'], bool) or not isinstance(metadata['start_step'], int) or metadata['start_step'] < 0:
            raise ValueError('_astra_conditioning.start_step must be a nonnegative integer')
        version = metadata['conditioning_version']
        if isinstance(version, bool) or not isinstance(version, (int, str)) or not str(version).strip():
            raise ValueError('Invalid conditioning_version')
        for key, expected in (('instruction', metadata['l3_instruction']),
                              ('l2_instruction', metadata['l2_instruction'])):
            if obs.get(key) != expected:
                raise ValueError(f'observation.{key} differs from conditioning metadata')
        if 'task_instruction' in obs and obs['task_instruction'] != metadata['l3_instruction']:
            raise ValueError('task_instruction would override the L3 instruction')
        for key in ('l2_visual_cue_rgb', 'l2_bboxes_xyxy'):
            if key not in obs:
                raise ValueError(f'l3_l2 requires observation.{key}')
        # Refuse inference if the audit cannot be preserved, even when the
        # upstream adapter itself accepts an observation.
        if not os.environ.get('ASTRA_CONDITIONING_AUDIT_LOG'):
            raise ValueError('l3_l2 requires ASTRA_CONDITIONING_AUDIT_LOG')
        result = super().update_obs(obs)
        actual = self._verify_actual_conditioning(metadata)
        receipt = dict(metadata, **actual, status='observation_verified', verified=True,
                       recorded_at=time.time())
        self._write_conditioning_audit(receipt)
        self._astra_conditioning_metadata = metadata
        self._astra_conditioning_receipt = receipt
        return result

    def get_action(self):
        if not self._audit_enabled():
            return super().get_action()
        metadata = getattr(self, '_astra_conditioning_metadata', None)
        if metadata is None:
            raise ValueError('No verified L3/L2 observation; call update_obs first')
        actual = self._verify_actual_conditioning(metadata)
        receipt = dict(metadata, **actual, status='inference_started', verified=True,
                       recorded_at=time.time())
        self._write_conditioning_audit(receipt)
        self._astra_conditioning_receipt = receipt
        try:
            actions = super().get_action()
        except Exception as exc:
            receipt = dict(receipt, status='inference_failed', error=f'{type(exc).__name__}: {exc}',
                           recorded_at=time.time())
            self._astra_conditioning_receipt = receipt
            self._write_conditioning_audit(receipt)
            raise
        receipt = dict(receipt, status='inference_completed', action_count=len(actions),
                       recorded_at=time.time())
        self._write_conditioning_audit(receipt)
        self._astra_conditioning_receipt = receipt
        return actions

    def get_conditioning_receipt(self):
        return copy.deepcopy(getattr(self, '_astra_conditioning_receipt', None))

    def reset(self):
        self._astra_conditioning_receipt = None
        self._astra_conditioning_metadata = None
        return super().reset()

    def update_obs_batch(self, *args, **kwargs):
        if self._audit_enabled():
            raise ValueError('Astra L3/L2 audit supports one environment per worker; batch disabled')
        return super().update_obs_batch(*args, **kwargs)

    def get_action_batch(self, *args, **kwargs):
        if self._audit_enabled():
            raise ValueError('Astra L3/L2 audit supports one environment per worker; batch disabled')
        return super().get_action_batch(*args, **kwargs)
