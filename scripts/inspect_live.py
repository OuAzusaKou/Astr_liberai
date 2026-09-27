"""Render the pending request for human/agent review; never approves actions."""
import json
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra.supervision import with_runtime_limits

session = Path(sys.argv[1]).resolve()
status = json.loads((session / 'controller_status.json').read_text())
deadline = time.monotonic() + 25
while status['status'] == 'waiting_for_agent' and Path(status['response']).exists() and time.monotonic() < deadline:
    time.sleep(0.5)
    status = json.loads((session / 'controller_status.json').read_text())
print(json.dumps(status))
if status['status'] != 'waiting_for_agent':
    raise SystemExit()
request = with_runtime_limits(session, json.loads(Path(status['request']).read_text()))
print('response_exists', Path(status['response']).exists())
for key in ('request_id', 'step', 'state', 'rejected_action', 'takeover_count', 'retries'):
    print(key, request[key])
for key in ('unlimited_recovery', 'allow_agent_termination', 'max_retries_per_subgoal', 'max_takeovers', 'runtime_control_authorization'):
    if key in request:
        print(key, request[key])
print('conditioning_mode', request.get('conditioning_mode', 'pure_l3'))
if request.get('conditioning_mode') == 'l3_l2':
    print('conditioning', json.dumps(request.get('conditioning'), ensure_ascii=False))
    if request.get('state') == 'PREPARE_CONDITIONING':
        print('conditioning_review_required', 'Review current images, then submit condition with complete L2/cue/boxes before the first inference; no actions are available yet.')
    images = request['observation']['images']
    cue_camera = 'cam_head' if 'cam_head' in images else 'cam_high'
    cue_path = images.get(cue_camera)
    if cue_path:
        with Image.open(cue_path) as cue:
            print('current_cue_rgb', json.dumps(dict(path=cue_path, width=cue.width, height=cue.height,
                                                     mode=cue.mode, camera=cue_camera,
                                                     coordinate_system=f'original {cue_camera} pixels (training cam_high)'), ensure_ascii=False))
        print('conditioning_update', 'For a new condition use this current_cue_rgb path, not live_contact.png. Replan/hand_back may explicitly confirm the existing conditioning_version if it remains valid.')
    else:
        print('conditioning_error', 'Current cam_high cue image is missing; keep paused and report the interface fault.')
if request.get('live_recovery_extension'):
    print('live_recovery_extension', json.dumps(request['live_recovery_extension']))
    print('native_counters', request['native_takeover_count'], request['native_retries'])
def rounded(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [rounded(v) for v in value]
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    return value
print('state', json.dumps(rounded(request['observation']['state'])))
print('actions', json.dumps(rounded(request['proposed_actions'][:4])))
sheet = Image.new('RGB', (960, 258), 'white')
draw = ImageDraw.Draw(sheet)
for i, (cam, path) in enumerate(request['observation']['images'].items()):
    im = Image.open(path).convert('RGB')
    im.thumbnail((320, 240))
    sheet.paste(im, (i * 320, 0))
    draw.text((i * 320, 241), f'{cam} step {request["step"]}', fill='black')
sheet.save(session / 'live_contact.png')
