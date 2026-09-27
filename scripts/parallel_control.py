"""Read eight live requests, or apply explicitly authored per-worker decisions.

This utility never invents approvals or calls an LLM in the background.
"""
import argparse
import json
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra.bridge import atomic_json, decision_for
from astra.supervision import publish_decision, with_runtime_limits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('operation', choices=['inspect', 'respond'])
    p.add_argument('session', type=Path)
    p.add_argument('--decisions', type=Path)
    args = p.parse_args()
    config = json.loads((args.session / 'parallel_manifest.json').read_text())
    workers = {w['task']: w for w in config['workers']}
    if args.operation == 'respond':
        decisions = json.loads(args.decisions.read_text())
        pending = []
        for task, command in decisions.items():
            folder = Path(workers[task]['session'])
            status = json.loads((folder / 'controller_status.json').read_text())
            if status['status'] != 'waiting_for_agent':
                raise ValueError(f'{task}: no pending request')
            response = Path(status['response'])
            if response.exists():
                raise ValueError(f'{task}: response already exists')
            request = with_runtime_limits(folder, json.loads(Path(status['request']).read_text()))
            decision_for(request, command)
            command['observation_reference'] = request['observation']['images']
            command['reviewed_at'] = time.time()
            command['wait_seconds'] = max(0, time.time() - status['updated_at'])
            pending.append((folder, response, command))
        # Validate all IDs before writing any response; each write remains worker-local.
        for folder, path, command in pending:
            publish_decision(folder, path, command)
        print(f'Sent {len(pending)} explicit decisions')
        return
    snapshot = {}
    for task, worker in workers.items():
        folder = Path(worker['session'])
        status_path = folder / 'controller_status.json'
        if not status_path.exists():
            print(task, 'not_ready')
            continue
        status = json.loads(status_path.read_text())
        if status['status'] != 'waiting_for_agent':
            print(task, status['status'])
            continue
        if Path(status['response']).exists():
            print(task, 'executing_or_replanning')
            continue
        request = json.loads(Path(status['request']).read_text())
        snapshot[task] = request
        canvas = Image.new('RGB', (960, 262), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (camera, path) in enumerate(request['observation']['images'].items()):
            if i >= 3:
                break
            canvas.paste(Image.open(path).convert('RGB').resize((320, 240)), (i * 320, 22))
            draw.text((i * 320 + 2, 3), f'{task} step{request["step"]} {camera}', fill='black')
        preview = folder / 'live_contact.png'
        canvas.save(preview)
        print(json.dumps(dict(task=task, request_id=request['request_id'], step=request['step'],
                              state=request['state'], instruction=request['observation'].get('instruction'),
                              demo_segments=request['demo_segments'], retries=request['retries'],
                              robot_state=request['observation'].get('state'),
                              proposed_actions=request['proposed_actions'][:4],
                              rejected=request['rejected_action'], image=str(preview)), ensure_ascii=False))
    atomic_json(args.session / 'pending_snapshot.json', snapshot)


if __name__ == '__main__':
    main()
