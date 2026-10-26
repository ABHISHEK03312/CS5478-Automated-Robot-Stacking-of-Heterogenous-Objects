"""CEM tower stacking in the Ravens workspace; no robot or grasp simulation.

Run: python cem_stack_only.py --data stacking_data_validated --episodes 1 --gui

CEM samples the next object ID and placement XY/yaw. It evaluates each
candidate with a PyBullet settling rollout and teleports the selected object.
The episode's recorded stacking order/actions are NOT read by the planner.
Loads Ravens plane/plane.urdf and ur5/workspace.urdf (no UR5 robot).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pybullet as p

from generate_stacking_ravens import make_body, fits, COLORS

CENTER = np.array([0.50, 0.18], dtype=float)
BOUNDS = ((0.25, 0.75), (-0.50, 0.50), (0.0, 0.28))


def settle(n):
    for _ in range(n):
        p.stepSimulation()


def stable_stack(order, objects, bodies, tol_xy=0.018, tol_z=0.018):
    height = 0.0
    for oid in order:
        obj = objects[oid]
        pos, quat = p.getBasePositionAndOrientation(bodies[oid])
        roll, pitch, _ = p.getEulerFromQuaternion(quat)
        expected = height + obj['dims'][2] / 2
        if (not np.all(np.isfinite(pos)) or
                np.linalg.norm(np.asarray(pos[:2]) - CENTER) > tol_xy or
                abs(pos[2] - expected) > tol_z or
                abs(roll) > 0.20 or abs(pitch) > 0.20):
            return False
        height += obj['dims'][2]
    return True


def can_finish(last, remaining, objects):
    """Reject choices that make stacking all remaining objects impossible."""
    if not remaining:
        return True
    for oid in remaining:
        if last is not None and not fits(objects[oid], objects[last], margin=0.001):
            continue
        if can_finish(oid, [j for j in remaining if j != oid], objects):
            return True
    return False


def create_scene(meta, gui, assets):
    client = p.connect(p.GUI if gui else p.DIRECT)
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)
    p.setPhysicsEngineParameter(numSolverIterations=80)
    assets = Path(assets)
    plane_urdf = assets / "plane" / "plane.urdf"
    workspace_urdf = assets / "ur5" / "workspace.urdf"
    for path in (plane_urdf, workspace_urdf):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Ravens asset: {path}")
    p.loadURDF(str(plane_urdf), [0, 0, -0.001], useFixedBase=True)
    p.loadURDF(str(workspace_urdf), [0.5, 0, 0], useFixedBase=True)
    if gui:
        p.resetDebugVisualizerCamera(cameraDistance=1.05, cameraYaw=35,
                                     cameraPitch=-65, cameraTargetPosition=[0.5, 0, 0.04])
    objects = {int(o['id']): o for o in meta['objects']}
    bodies = {}
    for oid, obj in objects.items():
        body = make_body(obj['shape'], obj['dims'], COLORS[oid % len(COLORS)])
        pos = [*obj['start_xy'], obj['dims'][2] / 2 + 0.001]
        p.resetBasePositionAndOrientation(body, pos, [0, 0, 0, 1])
        bodies[oid] = body
    settle(120)
    for oid, body in bodies.items():
        pos, _ = p.getBasePositionAndOrientation(body)
        if not (BOUNDS[0][0] < pos[0] < BOUNDS[0][1] and
                BOUNDS[1][0] < pos[1] < BOUNDS[1][1] and
                0 < pos[2] < 0.28):
            raise RuntimeError(f'Object {oid} invalid at start: {pos}')
    return client, objects, bodies


def evaluate(oid, x, order, objects, bodies, settle_steps):
    if order and not fits(objects[oid], objects[order[-1]], margin=0.001):
        return -1e6
    remaining = [j for j in objects if j not in order and j != oid]
    if not can_finish(oid, remaining, objects):
        return -1e6
    if np.max(np.abs(x[:2])) > 0.025 or abs(x[2]) > 0.20:
        return -1e6
    z = sum(objects[j]['dims'][2] for j in order) + objects[oid]['dims'][2] / 2 + 0.002
    if z > BOUNDS[2][1]:
        return -1e6
    state = p.saveState()
    try:
        body = bodies[oid]
        old_positions = {j: np.asarray(p.getBasePositionAndOrientation(bodies[j])[0]) for j in order}
        target = [float(CENTER[0] + x[0]), float(CENTER[1] + x[1]), float(z)]
        p.resetBasePositionAndOrientation(body, target, p.getQuaternionFromEuler([0, 0, float(x[2])]))
        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
        settle(settle_steps)
        if not stable_stack(order + [oid], objects, bodies):
            return -1e6
        displacement = sum(np.linalg.norm(np.asarray(p.getBasePositionAndOrientation(bodies[j])[0]) - old_positions[j]) for j in order)
        pos, _ = p.getBasePositionAndOrientation(body)
        error = np.linalg.norm(np.asarray(pos[:2]) - CENTER)
        return float(10 - 80 * error - 40 * displacement - 15 * np.linalg.norm(x[:2]) - abs(x[2]))
    finally:
        p.restoreState(state)
        p.removeState(state)


def cem_plan(order, objects, bodies, rng, samples, iters, elites, settle_steps, debug):
    ids = [i for i in objects if i not in order]
    probs = np.ones(len(ids)) / len(ids)
    means = {i: np.zeros(3) for i in ids}
    stds = {i: np.array([0.008, 0.008, 0.08]) for i in ids}
    best = (-1e6, None, None)
    for iteration in range(iters):
        batch = []
        for k in range(samples):
            oid = int(rng.choice(ids, p=probs))
            x = means[oid] + rng.normal(size=3) * stds[oid]
            if k == 0:
                valid = [i for i in ids if (not order or fits(objects[i], objects[order[-1]], margin=0.001))
                         and can_finish(i, [j for j in ids if j != i], objects)]
                if valid:
                    oid, x = valid[0], np.zeros(3)
            score = evaluate(oid, x, order, objects, bodies, settle_steps)
            batch.append((score, oid, x.copy()))
            if score > best[0]:
                best = (score, oid, x.copy())
        batch.sort(key=lambda t: t[0], reverse=True)
        elite = [b for b in batch[:elites] if b[0] > -1e5]
        if elite:
            counts = np.array([sum(e[1] == i for e in elite) for i in ids], dtype=float)
            probs = (counts + 0.5) / (len(elite) + 0.5 * len(ids))
            for oid in ids:
                xs = np.array([e[2] for e in elite if e[1] == oid])
                if len(xs):
                    means[oid] = 0.2 * means[oid] + 0.8 * xs.mean(axis=0)
                    stds[oid] = np.maximum(0.2 * stds[oid] + 0.8 * xs.std(axis=0), [0.002, 0.002, 0.02])
        if debug:
            print(f'  CEM {iteration + 1}/{iters}: best={best[0]:.3f}, '
                  f'object_probs={dict(zip(ids, np.round(probs, 2)))}', flush=True)
    return best if best[1] is not None and best[0] > -1e5 else None


def run_episode(path, args, seed):
    meta = json.loads(path.read_text())
    client = None
    try:
        client, objects, bodies = create_scene(meta, args.gui, args.assets)
        rng = np.random.default_rng(seed)
        order = []
        print(f'{path.parent.name}: {len(objects)} objects visible at start', flush=True)
        if args.gui:
            time.sleep(args.start_delay)
        for step in range(len(objects)):
            result = cem_plan(order, objects, bodies, rng, args.samples, args.iters,
                              args.elites, args.settle_steps, args.debug)
            if result is None:
                print(f'  step {step}: no feasible placement', flush=True)
                break
            score, oid, x = result
            z = sum(objects[j]['dims'][2] for j in order) + objects[oid]['dims'][2] / 2 + 0.002
            target = [float(CENTER[0] + x[0]), float(CENTER[1] + x[1]), float(z)]
            p.resetBasePositionAndOrientation(bodies[oid], target,
                                              p.getQuaternionFromEuler([0, 0, float(x[2])]))
            p.resetBaseVelocity(bodies[oid], [0, 0, 0], [0, 0, 0])
            settle(args.settle_steps)
            ok = stable_stack(order + [oid], objects, bodies)
            print(f'  step {step}: object={oid} shape={objects[oid]["shape"]} '
                  f'place=({target[0]:.3f},{target[1]:.3f},{target[2]:.3f}) '
                  f'score={score:.3f} stable={ok}', flush=True)
            if not ok:
                break
            order.append(oid)
            if args.gui:
                time.sleep(args.step_delay)
        success = len(order) == len(objects)
        print(f'  result: {len(order)}/{len(objects)} stacked; success={success}; order={order}', flush=True)
        if args.gui:
            time.sleep(args.end_delay)
        return {'episode': path.parent.name, 'success': success, 'order': order,
                'stacked': len(order), 'total': len(objects)}
    finally:
        if client is not None:
            p.disconnect(client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('stacking_data_validated'))
    parser.add_argument('--assets', type=Path, default=Path('ravens/environments/assets'))
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--samples', type=int, default=48)
    parser.add_argument('--iters', type=int, default=4)
    parser.add_argument('--elites', type=int, default=8)
    parser.add_argument('--settle-steps', type=int, default=90)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--start-delay', type=float, default=3.0)
    parser.add_argument('--step-delay', type=float, default=1.5)
    parser.add_argument('--end-delay', type=float, default=5.0)
    parser.add_argument('--out', type=Path, default=Path('cem_stack_only_results.json'))
    args = parser.parse_args()
    if not 1 <= args.elites <= args.samples:
        parser.error('Require 1 <= elites <= samples')
    paths = sorted(args.data.glob('episode_*/episode.json'))[:args.episodes]
    if not paths:
        parser.error(f'No episode JSON files in {args.data}')
    results = [run_episode(path, args, args.seed + i) for i, path in enumerate(paths)]
    args.out.write_text(json.dumps(results, indent=2))
    print(f'Finished: {sum(r["success"] for r in results)}/{len(results)} successful. '
          f'Results: {args.out}')


if __name__ == '__main__':
    main()
