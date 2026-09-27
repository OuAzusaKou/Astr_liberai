"""Send one explicitly reviewed decision; requires the exact pending request ID."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra.bridge import decision_for
from astra.supervision import publish_decision, with_runtime_limits

p = argparse.ArgumentParser()
p.add_argument('session', type=Path)
p.add_argument('request_id')
p.add_argument('mode', choices=['condition', 'approve', 'correct', 'takeover', 'hand_back', 'abort', 'blocked', 'replan'])
p.add_argument('count', type=int, help='Reviewed action count; new runs allow up to 15, bounded by the request batch limit')
p.add_argument('reason')
p.add_argument('--segment', default='grasp')
p.add_argument('--instruction')
conditioning = p.add_mutually_exclusive_group()
conditioning.add_argument('--conditioning-file', type=Path,
                          help='JSON with reviewed L2 instruction, current cam_high cue path, pixel boxes, and evidence')
conditioning.add_argument('--conditioning-version',
                          help='Explicitly reuse the pending request\'s still-valid L2 conditioning version')
p.add_argument('--recovery-evidence')
p.add_argument('--actions-file', type=Path)
p.add_argument('--subdivide', action='store_true')
p.add_argument('--scene-review')
p.add_argument('--short-batch-reason', help='Observed risk or review boundary requiring fewer than the configured 15 actions')
p.add_argument('--termination-condition')
p.add_argument('--termination-evidence')
p.add_argument('--completion-candidate', action='store_true')
p.add_argument('--completion-evidence')
attempt = p.add_mutually_exclusive_group()
attempt.add_argument('--new-attempt', action='store_true', help='Begin a distinct recovery retry while retaining takeover control')
attempt.add_argument('--continue-attempt', action='store_true', help='Continue the same recovery retry; do not increment retries')
args = p.parse_args()
status = json.loads((args.session / 'controller_status.json').read_text())
assert status['status'] == 'waiting_for_agent'
request = with_runtime_limits(args.session, json.loads(Path(status['request']).read_text()))
assert request['request_id'] == args.request_id
assert not Path(status['response']).exists()
command = dict(request_id=args.request_id, mode=args.mode, reason=args.reason,
               subgoal=args.segment, demo_segment=args.segment,
               observation_reference=request['observation']['images'])
if args.conditioning_file or args.conditioning_version is not None:
    if args.mode not in ('condition', 'replan', 'hand_back'):
        p.error('Conditioning may only be submitted or confirmed with condition, replan, or hand_back')
    if args.conditioning_file:
        try:
            condition = json.loads(args.conditioning_file.read_text())
        except (OSError, ValueError) as exc:
            p.error(f'Cannot read conditioning JSON: {exc}')
        if not isinstance(condition, dict):
            p.error('Conditioning JSON must be an object')
        command['conditioning'] = condition
    else:
        command['conditioning_version'] = args.conditioning_version
if args.mode == 'condition' and args.count != 0:
    p.error('condition prepares model inputs without executing actions; count must be 0')
if args.mode != 'takeover' and (args.new_attempt or args.continue_attempt):
    p.error('Recovery attempt flags apply only to takeover')
if args.mode == 'takeover':
    if request['state'] == 'TAKEOVER' and not (args.new_attempt or args.continue_attempt):
        p.error('Continuous takeover must explicitly use --continue-attempt for the SAME recovery, '
                'or --new-attempt for a NEW retry after failure. Describe what changed in reason; '
                'one takeover event can contain multiple recovery attempts.')
    command['new_attempt'] = args.new_attempt or request['state'] != 'TAKEOVER'
    key = command['subgoal']
    limit = request.get('max_retries_per_subgoal', 6)
    next_retry = request.get('retries', {}).get(key, 0) + int(command['new_attempt'])
    next_takeover = request.get('takeover_count', 0) + int(request['state'] != 'TAKEOVER')
    takeover_limit = request.get('max_takeovers', 10)
    if ((limit is not None and next_retry > limit)
            or (takeover_limit is not None and next_takeover > takeover_limit)):
        p.error('Recovery budget would be exceeded; no response was published. '
                'Keep the scene, assess safe hand-back/replanning or a documented terminal condition; '
                'do not rename the subgoal or silently start another attempt.')
if args.mode == 'approve':
    command['execute'] = args.count
elif args.mode in ('correct', 'takeover'):
    if args.actions_file:
        command['actions'] = json.loads(args.actions_file.read_text())
    else:
        assert args.mode == 'correct'
        command['original_proposal'] = request['proposed_actions'][:args.count]
        command['actions'] = deepcopy(command['original_proposal'])
        # Preserve finite gripper values. Native execution handles saturation.
        if args.subdivide:
            previous = request['observation']['state']
            short = []
            for action in command['actions']:
                middle = deepcopy(action)
                for key in ('left_arm_joint_state', 'right_arm_joint_state'):
                    middle[key] = [(a + b) / 2 for a, b in zip(previous[key], action[key])]
                short.extend([middle, action])
                previous = action
            command['actions'] = short
if args.instruction:
    command['instruction'] = args.instruction
if args.recovery_evidence:
    command['recovery_evidence'] = args.recovery_evidence
if args.completion_candidate:
    command['completion_candidate'] = True
    command['completion_evidence'] = args.completion_evidence
for key in ('scene_review', 'short_batch_reason', 'termination_condition', 'termination_evidence'):
    if getattr(args, key):
        command[key] = getattr(args, key)
decision_for(request, command)
publish_decision(args.session, status['response'], command)
print('sent', args.request_id, args.mode)
