"""Mixed discrete/continuous CEM for heterogeneous Ravens stacking.

Object selection, grasp XY/yaw, and placement XY/yaw are all sampled by CEM.
Planning uses geometry-gated grasp candidates and PyBullet teleport placement
rollouts; the selected action is executed with the Ravens UR5 ParallelJaw.

Run from repository root:
 python cem_joint.py --data stacking_data --episodes 1 --gui

This is NOT friction/contact-based grasp planning; the provided ParallelJaw
attaches a fixed constraint after a geometry check.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pybullet as p

from generate_stacking_dataset import fits, support_halfwidths
from cem_grasp_beam import DatasetSceneTask, settle
from ravens.environments.environment import Environment

CENTER = np.array([0.5, 0.18], dtype=float)


def angle_delta(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def rotated_width(length, width, angle):
    return abs(length * math.cos(angle)) + abs(width * math.sin(angle))


def completion_possible(last_id, remaining, geometry):
    """Check if there exists SOME support-feasible order for remaining objects.

    This does not dictate the order. CEM still samples and ranks the next ID.
    Small N (3-7) keeps this exact recursive feasibility check inexpensive.
    """
    if not remaining:
        return True
    if last_id is not None and geometry[last_id]['shape'] == 'triangle':
        return False
    for oid in remaining:
        if last_id is not None and not fits(geometry[oid], geometry[last_id], margin=0.001):
            continue
        rest = tuple(i for i in remaining if i != oid)
        if completion_possible(oid, rest, geometry):
            return True
    return False


class JointCEM:
    """Categorical CEM for object ID + conditional Gaussians for 6 pose variables.

    a = (object_id, grasp_dx, grasp_dy, grasp_yaw,
         place_dx, place_dy, place_yaw)

    grasp XY offsets are relative to the object's center; place XY offsets
    are relative to the stack center. Gripper width is derived from object
    geometry and yaw, not independently motor-controlled by this gripper.
    """

    def __init__(self, env, task, rng, samples=48, iters=4, elites=8, settle_steps=90,
                 debug=False):
        self.env, self.task, self.rng = env, task, rng
        self.samples, self.iters, self.elites = samples, iters, elites
        self.settle_steps = settle_steps
        self.debug = debug
        self.geometry = {int(o['id']): o for o in task.meta['objects']}
        self.rollouts = 0

    def _grasp(self, oid, z, x):
        """Return grasp pose and candidate jaw span, or None.

        Kinematic grasp filter mirrors DatasetParallelJaw's local-X width
        check. Exact finger/mesh contact is NOT simulated here.
        """
        obj = self.geometry[oid]
        body = self.task.bodies[oid]
        pos, rot = p.getBasePositionAndOrientation(body)
        length, width, _ = map(float, obj['dims'])
        obj_yaw = p.getEulerFromQuaternion(rot)[2]
        relative = angle_delta(obj_yaw, x[2])
        jaw_width = length if obj['shape'] == 'cylinder' else rotated_width(length, width, relative)
        if jaw_width + 0.002 > self.env.ee.OPEN_INNER_WIDTH:
            return None
        # Match geometry-gated ParallelJaw local centering tolerance.
        if math.hypot(x[0], x[1]) > 0.022:
            return None
        pick_pos = (float(pos[0]+x[0]), float(pos[1]+x[1]), float(p.getAABB(body)[1][2]))
        pick_rot = p.getQuaternionFromEuler((0, 0, float(x[2])))
        return (pick_pos, pick_rot), jaw_width

    def _candidate_score(self, oid, x, placed, remaining):
        obj = self.geometry[oid]
        if placed and not fits(obj, self.geometry[placed[-1]], margin=0.001):
            return -1e5
        if not completion_possible(oid, tuple(i for i in remaining if i != oid), self.geometry):
            return -1e5
        grasp = self._grasp(oid, 0, x)
        if grasp is None:
            return -1e5
        pick, jaw_width = grasp
        px, py = CENTER + x[3:5]
        # Avoid unstable targets that are far from the stack center.
        if max(abs(x[3]), abs(x[4])) > 0.025:
            return -1e5
        body = self.task.bodies[oid]
        old_pose = p.getBasePositionAndOrientation(body)
        old_yaw = p.getEulerFromQuaternion(old_pose[1])[2]
        # Pick/Place action rotations are TCP angles. Calculate final object yaw
        # by conserving the relative yaw of object within the gripper.
        final_yaw = old_yaw + angle_delta(x[5], x[2])
        base_z = sum(self.geometry[i]['dims'][2] for i in placed)
        final_z = base_z + obj['dims'][2] / 2 + 0.002
        snapshot = p.saveState()
        self.rollouts += 1
        try:
            before = {j: np.asarray(p.getBasePositionAndOrientation(self.task.bodies[j])[0])
                      for j in placed}
            p.resetBasePositionAndOrientation(body, (float(px), float(py), float(final_z)),
                                              p.getQuaternionFromEuler((0, 0, final_yaw)))
            p.resetBaseVelocity(body, (0, 0, 0), (0, 0, 0))
            for _ in range(self.settle_steps):
                p.stepSimulation()
            # Strictly require previous objects to remain stable and the new one
            # to land at its intended height near the existing stack.
            expected_z = 0.0
            for j in placed + [oid]:
                b = self.task.bodies[j]
                pos, quat = p.getBasePositionAndOrientation(b)
                roll, pitch, _ = p.getEulerFromQuaternion(quat)
                expected_z += self.geometry[j]['dims'][2] / 2
                if (np.linalg.norm(np.asarray(pos[:2]) - CENTER) > 0.018
                        or abs(pos[2] - expected_z) > 0.018
                        or abs(roll) > 0.20 or abs(pitch) > 0.20):
                    return -1e5
                expected_z += self.geometry[j]['dims'][2] / 2
            disturbance = sum(np.linalg.norm(
                np.asarray(p.getBasePositionAndOrientation(self.task.bodies[j])[0]) - before[j])
                for j in placed)
            # Encourage stable, centered placements and reasonably accessible picks.
            # Shape and geometry determine support feasibility, not a prescribed order.
            travel = np.linalg.norm(np.asarray(pick[0][:2]) - CENTER)
            bx, by = support_halfwidths(self.geometry[placed[-1]]) if placed else (0.10, 0.10)
            margin = min(bx - obj['dims'][0]/2, by - obj['dims'][1]/2)
            score = (10.0 + 8.0 * min(margin, 0.03)
                     - 65.0 * np.linalg.norm(x[3:5])
                     - 35.0 * disturbance - 0.3 * travel
                     - 2.0 * np.linalg.norm(x[:2]))
            return float(score)
        finally:
            p.restoreState(snapshot)
            p.removeState(snapshot)

    def plan(self, placed):
        remaining = [oid for oid in self.geometry if oid not in placed]
        if not remaining:
            return None
        # If no support-compatible full sequence exists, fail explicitly.
        if not completion_possible(placed[-1] if placed else None,
                                   tuple(remaining), self.geometry):
            raise RuntimeError('No complete stacking order is geometry-feasible')

        ids = np.array(remaining, dtype=int)
        probs = np.full(len(ids), 1.0 / len(ids))
        means = {}
        stds = {}
        for oid in ids:
            obj = self.geometry[int(oid)]
            l, w, _ = obj['dims']
            # Two possible initial closing directions; let CEM choose and refine.
            seed_yaw = 0.0 if l <= w else math.pi / 2
            means[int(oid)] = np.array([0., 0., seed_yaw, 0., 0., seed_yaw])
            stds[int(oid)] = np.array([0.008, 0.008, 0.65, 0.010, 0.010, 0.65])
        best_score, best_oid, best_x = -float('inf'), None, None
        min_std = np.array([0.002, 0.002, 0.07, 0.002, 0.002, 0.07])
        for iteration in range(self.iters):
            batch = []
            for k in range(self.samples):
                oid = int(self.rng.choice(ids, p=probs))
                x = means[oid] + self.rng.normal(size=6) * stds[oid]
                if k == 0:
                    # Always evaluate one feasible-order candidate near center.
                    candidates = [int(i) for i in ids if completion_possible(
                        int(i), tuple(int(j) for j in ids if j != i), self.geometry)
                        and (not placed or fits(self.geometry[int(i)],
                                                self.geometry[placed[-1]], margin=0.001))]
                    if candidates:
                        oid = max(candidates, key=lambda i: float(probs[np.where(ids == i)[0][0]]))
                        x = means[oid].copy()
                score = self._candidate_score(oid, x, placed, remaining)
                batch.append((score, oid, x.copy()))
                if score > best_score:
                    best_score, best_oid, best_x = score, oid, x.copy()
            batch.sort(key=lambda item: item[0], reverse=True)
            elites = [b for b in batch[:self.elites] if b[0] > -1e4]
            if not elites:
                # Retain original distributions; retry a broader, fresh sample.
                for oid in ids:
                    stds[int(oid)] = np.maximum(stds[int(oid)],
                                                [0.008, 0.008, 0.65, 0.010, 0.010, 0.65])
                continue
            freq = np.array([sum(b[1] == i for b in elites) for i in ids], dtype=float)
            target_probs = (freq + 0.5) / (len(elites) + 0.5 * len(ids))
            probs = 0.2 * probs + 0.8 * target_probs
            probs /= probs.sum()
            for oid in ids:
                members = np.array([b[2] for b in elites if b[1] == oid])
                if not len(members):
                    continue
                # Angle wrapping before Gaussian updates avoids averaging 0 and 2pi.
                for angle_col in (2, 5):
                    members[:, angle_col] = means[int(oid)][angle_col] + np.array([
                        angle_delta(a, means[int(oid)][angle_col])
                        for a in members[:, angle_col]])
                means[int(oid)] = 0.2 * means[int(oid)] + 0.8 * members.mean(axis=0)
                stds[int(oid)] = np.maximum(
                    0.2 * stds[int(oid)] + 0.8 * members.std(axis=0), min_std)
            if self.debug:
                print(f'  CEM {iteration+1}/{self.iters}: best={best_score:.3f} '
                      f'object_probs={dict(zip(ids.tolist(), np.round(probs, 2).tolist()))}', flush=True)
        if best_oid is None or best_score <= -1e4:
            return None
        grasp = self._grasp(best_oid, 0, best_x)
        if grasp is None:
            return None
        pick_pose, jaw_width = grasp
        place_top = sum(self.geometry[i]['dims'][2] for i in placed) + self.geometry[best_oid]['dims'][2]
        place_pose = ((float(CENTER[0]+best_x[3]), float(CENTER[1]+best_x[4]), float(place_top)),
                      p.getQuaternionFromEuler((0., 0., float(best_x[5]))))
        return {'object_id': int(best_oid), 'pick': pick_pose, 'place': place_pose,
                'jaw_width': float(jaw_width), 'score': float(best_score),
                'grasp_yaw': float(best_x[2]), 'place_yaw': float(best_x[5])}


def run_episode(path, assets, gui, samples, iters, elites, settle_steps, seed, debug):
    meta = json.loads(Path(path).read_text())
    task = DatasetSceneTask(meta)
    env = Environment(str(assets), task=task, disp=gui, hz=240)
    placed, actions = [], []
    try:
        env.seed(seed)
        env.reset()
        planner = JointCEM(env, task, np.random.default_rng(seed), samples,
                           iters, elites, settle_steps, debug)
        print(f'{Path(path).parent.name}: {len(meta["objects"])} objects', flush=True)
        for step in range(len(meta['objects'])):
            a = planner.plan(placed)
            if a is None:
                print(f'  step {step}: CEM found no feasible full-stack action', flush=True)
                break
            oid = a['object_id']
            print(f'  step {step}: CEM chose object={oid} '
                  f'grasp_yaw={a["grasp_yaw"]:.2f} '
                  f'place_xy=({a["place"][0][0]:.3f},{a["place"][0][1]:.3f}) '
                  f'jaw_width={a["jaw_width"]:.3f}', flush=True)
            # Existing primitive implements geometry-gated, constraint-assisted grip.
            timeout = task.primitive(env.movej, env.movep, env.ee,
                                     a['pick'], a['place'])
            settle(180)
            # Recheck executed placement against stack pose and tilt constraints.
            z = 0.0
            stable = True
            for j in placed + [oid]:
                b = task.bodies[j]
                pos, quat = p.getBasePositionAndOrientation(b)
                roll, pitch, _ = p.getEulerFromQuaternion(quat)
                exp_z = z + planner.geometry[j]['dims'][2] / 2
                print(
                    f"    object={j} "
                    f"actual_xyz={np.round(pos, 4)} "
                    f"expected_xyz=({CENTER[0]:.3f}, {CENTER[1]:.3f}, {exp_z:.4f}) "
                    f"roll={roll:.3f} pitch={pitch:.3f}",
                    flush=True
)
                if (np.linalg.norm(np.asarray(pos[:2]) - CENTER) > 0.018
                        or abs(pos[2]-exp_z) > 0.018
                        or abs(roll) > 0.20 or abs(pitch) > 0.20):
                    stable = False
                z += planner.geometry[j]['dims'][2]
            ok = bool(not timeout and task.primitive.last_grasp and stable)
            actions.append({'object_id': oid, 'pick': [list(a['pick'][0]), list(a['pick'][1])],
                            'place': [list(a['place'][0]), list(a['place'][1])],
                            'jaw_width': a['jaw_width'], 'cem_score': a['score'],
                            'grasp_attached': bool(task.primitive.last_grasp),
                            'timeout': bool(timeout), 'stack_stable': bool(stable),
                            'success': ok})
            print(f'    executed: grasp={task.primitive.last_grasp}, '
                  f'stable={stable}, success={ok}', flush=True)
            if not ok:
                break
            placed.append(oid)
            if gui:
                time.sleep(0.3)
        return {'episode': Path(path).parent.name, 'total': len(meta['objects']),
                'stacked': len(placed), 'success': len(placed) == len(meta['objects']),
                'order_chosen_by_cem': placed, 'rollouts': planner.rollouts,
                'actions': actions}
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('stacking_data'))
    ap.add_argument('--assets', type=Path, default=Path('ravens/environments/assets'))
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--samples', type=int, default=48)
    ap.add_argument('--iters', type=int, default=4)
    ap.add_argument('--elites', type=int, default=8)
    ap.add_argument('--settle-steps', type=int, default=90)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--out', type=Path, default=Path('cem_joint_results.json'))
    args = ap.parse_args()
    if args.elites > args.samples or args.elites < 1 or args.samples < 1:
        ap.error('Require 1 <= --elites <= --samples')
    files = sorted(args.data.glob('episode_*/episode.json'))[:args.episodes]
    if not files:
        ap.error('No episode.json files found')
    results = []
    for i, file in enumerate(files):
        try:
            results.append(run_episode(file, args.assets, args.gui, args.samples,
                                       args.iters, args.elites, args.settle_steps,
                                       args.seed+i, args.debug))
        except (RuntimeError, ValueError) as e:
            print(f'{file.parent.name}: {type(e).__name__}: {e}', flush=True)
            results.append({'episode': file.parent.name, 'success': False, 'error': str(e)})
    args.out.write_text(json.dumps(results, indent=2))
    print(f'Finished: {sum(r["success"] for r in results)}/{len(results)} successful. '
          f'Results: {args.out}', flush=True)


if __name__ == '__main__':
    main()
