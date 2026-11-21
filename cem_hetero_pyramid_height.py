"""CEM target-free heterogeneous pyramid planner (3/6/10 objects).

Uses original episode_*/episode.json scenes, not Ravens target poses.
Infers 2/3/4 layers by object count; --layers optionally filters episodes.
Low CEM defaults: 24 samples, 3 iterations, 4 elites.
Requires hetero_pyramid_support.py next to this script.
"""
import argparse
import csv
import statistics
import json
import math
import time
from pathlib import Path

import numpy as np
import pybullet as p

from generate_stacking_ravens import fits, support_halfwidths
from cem_grasp_beam import DatasetSceneTask, settle
from ravens.environments.environment import Environment
from hetero_pyramid_support_height import resolve_layers, placement_target, pyramid_stable, row_and_col, height_completion_possible

CENTER = np.array([0.5, 0.18], dtype=float)

def stable_tower(task, geometry, ordered, debug=False):
    """Check upright bodies, vertical stacking, and COM projection on support.

    This is a conservative footprint/center-of-mass proxy (not a proof of
    dynamic stability). No global center XY is required.
    """
    for k, oid in enumerate(ordered):
        body = task.bodies[oid]
        pos, quat = p.getBasePositionAndOrientation(body)
        roll, pitch, yaw = p.getEulerFromQuaternion(quat)
        if not np.all(np.isfinite(pos)) or abs(roll) > 0.20 or abs(pitch) > 0.20:
            return False
        aabb = p.getAABB(body)
        if k == 0:
            if abs(aabb[0][2]) > .020:  # bottom rests near workspace surface
                return False
            continue
        lower_id = ordered[k - 1]
        support_body = task.bodies[lower_id]
        support_pos, support_quat = p.getBasePositionAndOrientation(support_body)
        support_aabb = p.getAABB(support_body)
        # Actual top of lower and actual bottom of upper should touch.
        gap = aabb[0][2] - support_aabb[1][2]
        # Recover support footprint in support's local XY coordinate frame.
        lower_yaw = p.getEulerFromQuaternion(support_quat)[2]
        dx, dy = pos[0]-support_pos[0], pos[1]-support_pos[1]
        local_x = math.cos(lower_yaw)*dx + math.sin(lower_yaw)*dy
        local_y = -math.sin(lower_yaw)*dx + math.cos(lower_yaw)*dy
        support = geometry[lower_id]
        length, width = map(float, support['dims'][:2])
        # Cylinder/box have different support footprints; the conservative
        # bounding rectangle is further reduced by margin.
        if support['shape'] == 'cylinder':
            supported = math.hypot(local_x, local_y) <= min(length, width)/2 - .002
        else:
            supported = (abs(local_x) <= length/2 - .002 and
                         abs(local_y) <= width/2 - .002)
        if debug:
            print(f'  support id={oid} on={lower_id} offset=({local_x:.3f},{local_y:.3f}) gap={gap:.3f} supported={supported}', flush=True)
        if not supported or gap < -.012 or gap > .014:
            return False
    return True


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
                 debug=False, structure="pyramid", layers=None):
        self.structure = structure
        self.layers = resolve_layers(len(task.meta["objects"]), layers)
        self.env, self.task, self.rng = env, task, rng
        self.samples, self.iters, self.elites = samples, iters, elites
        self.settle_steps = settle_steps
        self.debug = debug
        self.geometry = {int(o['id']): o for o in task.meta['objects']}
        self.rollouts = 0
        self._height_cache = {}
        self._height_compatible = set()

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
        if self.structure == 'pyramid' and oid not in self._height_compatible:
            return -1e5
        if self.structure == 'tower':
            if placed and not fits(obj, self.geometry[placed[-1]], margin=0.001):
                return -1e5
            if not completion_possible(oid, tuple(i for i in remaining if i != oid), self.geometry):
                return -1e5
        grasp = self._grasp(oid, 0, x)
        if grasp is None:
            return -1e5
        pick, jaw_width = grasp
        layout = placement_target(placed, oid, self.layers, self.geometry, self.task.bodies)
        if layout is None:
            return -1e5
        target_xy, final_z = layout
        px, py = target_xy + x[3:5]
        # Limit the sampled displacement around the immediate supporting object.
        if max(abs(x[3]), abs(x[4])) > 0.025:
            return -1e5
        body = self.task.bodies[oid]
        old_pose = p.getBasePositionAndOrientation(body)
        old_yaw = p.getEulerFromQuaternion(old_pose[1])[2]
        # Pick/Place action rotations are TCP angles. Calculate final object yaw
        # by conserving the relative yaw of object within the gripper.
        final_yaw = old_yaw + angle_delta(x[5], x[2])
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
            if not pyramid_stable(placed + [oid], self.layers, self.geometry, self.task.bodies):
                return -1e5
            disturbance = sum(np.linalg.norm(
                np.asarray(p.getBasePositionAndOrientation(self.task.bodies[j])[0]) - before[j])
                for j in placed)
            # Favor stable placements and reasonably accessible picks.
            # Geometry determines support feasibility, not a prescribed order.
            travel = np.linalg.norm(np.asarray(pick[0][:2]) - target_xy)
            margin = 0.02  # Physical overlap is validated by pyramid_stable().
            # Prefer larger, stable objects at the bottom of the pyramid.
            base_bonus = (0.8 * obj['dims'][0] / max(g['dims'][0] for g in self.geometry.values())
                          if row_and_col(len(placed), self.layers)[0] == 0 else 0.0)
            score = (10.0 + base_bonus + 8.0 * min(margin, 0.03)
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
        # Reject choices that make a complete height-compatible pyramid
        # impossible BEFORE evaluating candidate grasps and rollouts.
        self._height_compatible = set()
        for oid in remaining:
            key = tuple(placed + [oid])
            if key not in self._height_cache:
                self._height_cache[key] = height_completion_possible(
                    key, tuple(i for i in remaining if i != oid),
                    self.layers, self.geometry)
            if self._height_cache[key]:
                self._height_compatible.add(oid)
        if not self._height_compatible:
            print(f'  No height-compatible completion after {placed}; '
                  'check object heights or support tolerance.', flush=True)
            return None
        # If no support-compatible full sequence exists, fail explicitly.
        if self.structure == 'tower' and not completion_possible(placed[-1] if placed else None,
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
                    candidates = ([int(i) for i in ids if int(i) in self._height_compatible] if self.structure == 'pyramid' else
                        [int(i) for i in ids if completion_possible(
                            int(i), tuple(int(j) for j in ids if j != i), self.geometry)
                            and (not placed or fits(self.geometry[int(i)],
                                                    self.geometry[placed[-1]], margin=0.001))])
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
        layout = placement_target(placed, best_oid, self.layers, self.geometry, self.task.bodies)
        if layout is None:
            return None
        place_xy, center_z = layout
        place_top = center_z + self.geometry[best_oid]['dims'][2] / 2
        place_pose = ((float(place_xy[0]+best_x[3]), float(place_xy[1]+best_x[4]), float(place_top)),
                      p.getQuaternionFromEuler((0., 0., float(best_x[5]))))
        return {'object_id': int(best_oid), 'pick': pick_pose, 'place': place_pose,
                'jaw_width': float(jaw_width), 'score': float(best_score),
                'grasp_yaw': float(best_x[2]), 'place_yaw': float(best_x[5])}


def run_episode(path, assets, gui, samples, iters, elites, settle_steps, seed, debug,
                structure='pyramid', layers=None):
    meta = json.loads(Path(path).read_text())
    task = DatasetSceneTask(meta)
    env = Environment(str(assets), task=task, disp=gui, hz=240)
    placed, actions = [], []
    try:
        env.seed(seed)
        env.reset()
        if gui:
            p.resetDebugVisualizerCamera(cameraDistance=1.25, cameraYaw=45,
                cameraPitch=-65, cameraTargetPosition=[0.5, 0.0, 0.05])
        print('Spawned objects:', [(oid, tuple(round(v, 3) for v in
              p.getBasePositionAndOrientation(body)[0]))
              for oid, body in task.bodies.items()], flush=True)
        planner = JointCEM(env, task, np.random.default_rng(seed), samples,
                           iters, elites, settle_steps, debug, structure=structure, layers=layers)
        print(f'{structure}: {len(meta["objects"])} objects', flush=True)
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
            print("STARTING UR5 MOVEMENT", flush=True)
            settle(180)
            # Recheck executed placement and remaining support after settling.
            stable = pyramid_stable(placed + [oid], planner.layers, planner.geometry, task.bodies)
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
        return {'episode': (Path(path).parent.name if path is not None else f'pyramid_seed_{seed}'), 'structure': structure, 'layers': planner.layers, 'total': len(meta['objects']),
                'stacked': len(placed), 'success': len(placed) == len(meta['objects']),
                'order_chosen_by_cem': placed, 'rollouts': planner.rollouts,
                'actions': actions}
    finally:
        env.close()




def summarize(results, structure):
    """Count all requested episodes, including planning and execution failures.

    Reward is executed placement fraction, not native Ravens reward.
    95% CI is a normal-approximation interval, clipped to [0,1].
    """
    n = len(results)
    rewards = []
    trials = []
    full = partial = zero = errors = 0
    for r in results:
        total = r.get('total')
        stacked = max(0, int(r.get('stacked', 0)))
        valid_total = isinstance(total, int) and total > 0
        reward = float(r['native_reward']) if 'native_reward' in r else (min(stacked / total, 1.0) if valid_total else 0.0)
        rewards.append(reward)
        trials.append(len(r.get('actions') or []))
        if r.get('error'):
            errors += 1
        if r.get('success'):
            full += 1
        elif reward > 0:
            partial += 1
        else:
            zero += 1
    avg = statistics.mean(rewards) if n else 0.0
    se = (statistics.stdev(rewards) / math.sqrt(n)) if n > 1 else 0.0
    lo, hi = max(0.0, avg - 1.96 * se), min(1.0, avg + 1.96 * se)
    return {'Method': 'CEM - ' + structure.title(), 'Episodes': n,
            'Avg. Trials': round(statistics.mean(trials), 2) if n else 0,
            'Mean Reward': round(avg, 3), '95% CI': f'[{lo:.3f}, {hi:.3f}]',
            'Full Stacks': full, 'Partial Stacks': partial,
            'Zero Reward': zero, 'Errors': errors}


def print_summary(summary):
    headers = list(summary)
    values = [str(summary[k]) for k in headers]
    widths = [max(len(k), len(v)) for k, v in zip(headers, values)]
    print(' | '.join(k.ljust(w) for k, w in zip(headers, widths)))
    print('-+-'.join('-' * w for w in widths))
    print(' | '.join(v.ljust(w) for v, w in zip(values, widths)))



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--layers', type=int, choices=[2, 3, 4], default=None,
                    help='Required layers; auto-inferred for 3, 6, or 10 objects')
    ap.add_argument('--data', type=Path, default=Path('mixed-data-validated'))
    ap.add_argument('--assets', type=Path, default=Path('ravens/environments/assets'))
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--samples', type=int, default=24)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--elites', type=int, default=4)
    ap.add_argument('--settle-steps', type=int, default=90)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--out', type=Path, default=Path('results/cem_hetero_results.json'))
    ap.add_argument('--summary-out', type=Path, default=Path('results/cem_hetero_summary.csv'))
    args = ap.parse_args()
    if not 1 <= args.elites <= args.samples or args.episodes < 1:
        ap.error('Invalid samples/elites/episodes')
    files = [f for f in sorted(args.data.glob('episode_*/episode.json'))
             if (len(json.loads(f.read_text())['objects']) in (3, 6, 10)
                 and (args.layers is None or
                      len(json.loads(f.read_text())['objects']) == args.layers*(args.layers+1)//2))][:args.episodes]
    if not files:
        ap.error(f'No episode_*/episode.json found under {args.data}')
    results = []
    for i, file in enumerate(files):
        try:
            results.append(run_episode(file, args.assets, args.gui, args.samples,
                args.iters, args.elites, args.settle_steps, args.seed+i, args.debug, 'pyramid', args.layers))
        except Exception as exc:
            print(f'{file}: {type(exc).__name__}: {exc}', flush=True)
            results.append({'episode': file.parent.name, 'success':False,
                            'stacked':0, 'actions':[], 'error':str(exc)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    report = summarize(results, 'pyramid')
    with args.summary_out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(report))
        writer.writeheader(); writer.writerow(report)
    print_summary(report)
    print(f'Results: {args.out}; Summary: {args.summary_out}')

if __name__ == '__main__':
    main()
