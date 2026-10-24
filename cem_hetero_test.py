"""Run heterogeneous CEM on scenes saved by generate_stacking_dataset.py.

Example: python cem_hetero_test.py --data stacking_data --episodes 3 --gui
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image

from generate_stacking_dataset import make_body, camera, COLORS
from cem_hetero import HeteroCEM, find_stack_order


def load_scene(meta):
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE))
    bodies = {}
    for obj in meta['objects']:
        oid = int(obj['id'])
        body = make_body(obj['shape'], obj['dims'], COLORS[oid % len(COLORS)])
        xyz = [*obj['start_xy'], obj['dims'][2] / 2 + 0.001]
        p.resetBasePositionAndOrientation(body, xyz, [0, 0, 0, 1])
        bodies[oid] = body
    for _ in range(120):
        p.stepSimulation()
    return bodies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('stacking_data'))
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--samples', type=int, default=24)
    parser.add_argument('--elites', type=int, default=5)
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--gui', action='store_true', help='Show PyBullet scene (slower)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save-images', action='store_true', help='Save overhead images after placements')
    parser.add_argument('--out', type=Path, default=Path('cem_results'))
    args = parser.parse_args()
    files = sorted(args.data.glob('episode_*/episode.json'))[:args.episodes]
    if not files:
        parser.error(f'No episode_*/episode.json found under {args.data}')
    args.out.mkdir(parents=True, exist_ok=True)
    cid = p.connect(p.GUI if args.gui else p.DIRECT)
    successes = 0
    summary = []
    try:
        if args.gui:
            p.resetDebugVisualizerCamera(cameraDistance=0.82, cameraYaw=35,
                                         cameraPitch=-55, cameraTargetPosition=[0, 0.08, 0.05])
        for idx, file in enumerate(files):
            meta = json.loads(file.read_text())
            objects = meta['objects']
            bodies = load_scene(meta)
            agent = HeteroCEM(objects, bodies, samples=args.samples,
                              elites=args.elites, iterations=args.iters,
                              seed=args.seed+idx, verbose=args.verbose)
            planned = agent.order
            placed, details = [], []
            started = time.perf_counter()
            if args.save_images:
                frame_dir = args.out / file.parent.name
                frame_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(camera()[0]).save(frame_dir / 'before.png')
            print(f'\nEpisode {idx+1}: {len(objects)} objects; order {planned}')
            for step, oid in enumerate(planned):
                action, score = agent.optimize(oid, placed)
                if action is None:
                    print(f'  FAILED: no stable candidate for object {oid}')
                    break
                ok = agent.place(oid, action, placed)
                details.append({'object_id': oid, 'shape': agent.objects[oid]['shape'],
                                'xy_yaw': action.tolist(), 'score': float(score), 'stable': bool(ok)})
                if not ok:
                    print(f'  FAILED: placed object {oid} did not remain stable')
                    break
                placed.append(oid)
                print(f'  {step+1}/{len(planned)} {agent.objects[oid]["shape"]}: stable; score={score:.2f}')
                if args.save_images:
                    Image.fromarray(camera()[0]).save(frame_dir / f'after_{step+1:02d}.png')
                if args.gui:
                    time.sleep(0.4)
            success = len(placed) == len(objects)
            successes += success
            elapsed = time.perf_counter()-started
            print(f'  SUCCESS={success}; stacked={len(placed)}/{len(objects)}; rollouts={agent.rollouts}; {elapsed:.1f}s')
            summary.append({'episode':file.parent.name,'n_objects':len(objects),
                            'n_stacked':len(placed), 'success':bool(success),
                            'rollouts':agent.rollouts,'seconds':elapsed,
                            'order': planned, 'actions': details})
        (args.out / 'results.json').write_text(json.dumps(summary, indent=2))
        print(f'\nSuccess: {successes}/{len(files)}; results saved in {args.out}')
    finally:
        p.disconnect(cid)


if __name__ == '__main__':
    main()
