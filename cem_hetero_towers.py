"""CEM heterogeneous tower planner. Input: original episode_*/episode.json.
Outputs results JSON and CSV summary; never reads a Ravens demo directory.

python cem_hetero_towers.py --data stacking_test --episodes 100 --samples 24 --iters 3 --elites 4 
--out results/cem_test.json --summary-out results/cem_test_summary.csv
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
from ravens.grasping.policies import POLICIES
from ravens.grasping.policies import create_grasp_planner

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
                 debug=False, structure="tower", grasp_scorer=None):
        self.structure = structure
        self.env, self.task, self.rng = env, task, rng
        self.samples, self.iters, self.elites = samples, iters, elites
        self.settle_steps = settle_steps
        self.debug = debug
        self.grasp_scorer = grasp_scorer
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
        if self.structure == 'tower':
            if placed and not fits(obj, self.geometry[placed[-1]], margin=0.001):
                return -1e5
            if not completion_possible(oid, tuple(i for i in remaining if i != oid), self.geometry):
                return -1e5
        grasp = self._grasp(oid, 0, x)
        if grasp is None:
            return -1e5
        pick, jaw_width = grasp
        target_xy = (PYRAMID_SLOTS[len(placed), :2] if self.structure == 'pyramid'
                     else np.asarray(p.getBasePositionAndOrientation(self.task.bodies[placed[-1]])[0][:2])
                     if placed else CENTER)
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
        if self.structure == 'pyramid':
            final_z = float(PYRAMID_SLOTS[len(placed), 2]) + 0.002
        else:
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
            if self.structure == 'pyramid':
                # During scoring, accept small XY offsets from the slot center.
                previous_ok = pyramid_stable(self.task, placed)
                here, orientation = p.getBasePositionAndOrientation(body)
                roll, pitch, _ = p.getEulerFromQuaternion(orientation)
                expected = PYRAMID_SLOTS[len(placed)].copy()
                placement_ok = (np.linalg.norm(np.asarray(here[:2]) - expected[:2]) <= 0.018
                                and abs(here[2] - expected[2]) <= 0.018
                                and abs(roll) <= 0.20 and abs(pitch) <= 0.20)
                if not (previous_ok and placement_ok):
                    return -1e5
            else:
                if not stable_tower(self.task, self.geometry, placed + [oid]):
                    return -1e5
            disturbance = sum(np.linalg.norm(
                np.asarray(p.getBasePositionAndOrientation(self.task.bodies[j])[0]) - before[j])
                for j in placed)
            # Favor stable placements and reasonably accessible picks.
            # Geometry determines support feasibility, not a prescribed order.
            travel = np.linalg.norm(np.asarray(pick[0][:2]) - target_xy)
            if self.structure == 'pyramid':
                margin = 0.02  # Layer support evaluated by PyBullet settlement.
            else:
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
        best_by_oid = {}
        min_std = np.array([0.002, 0.002, 0.07, 0.002, 0.002, 0.07])
        for iteration in range(self.iters):
            batch = []
            for k in range(self.samples):
                oid = int(self.rng.choice(ids, p=probs))
                x = means[oid] + self.rng.normal(size=6) * stds[oid]
                if k == 0:
                    # Always evaluate one feasible-order candidate near center.
                    candidates = ([int(i) for i in ids] if self.structure == 'pyramid' else
                        [int(i) for i in ids if completion_possible(
                            int(i), tuple(int(j) for j in ids if j != i), self.geometry)
                            and (not placed or fits(self.geometry[int(i)],
                                                    self.geometry[placed[-1]], margin=0.001))])
                    if candidates:
                        oid = max(candidates, key=lambda i: float(probs[np.where(ids == i)[0][0]]))
                        x = means[oid].copy()
                score = self._candidate_score(oid, x, placed, remaining)
                batch.append((score, oid, x.copy()))
                if score > -1e4 and (oid not in best_by_oid or
                                     score > best_by_oid[oid][0]):
                    best_by_oid[oid] = (score, x.copy())
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
        if self.grasp_scorer is not None and best_by_oid:
            body_to_oid = {self.task.bodies[oid]: oid for oid in best_by_oid}
            fallback_body = self.task.bodies[best_oid]
            selected_body = self.grasp_scorer.select_object(
                body_to_oid, fallback=fallback_body)
            selected_oid = body_to_oid.get(selected_body, best_oid)
            best_oid = selected_oid
            best_score, best_x = best_by_oid[selected_oid]
        grasp = self._grasp(best_oid, 0, best_x)
        if grasp is None:
            return None
        pick_pose, jaw_width = grasp
        if self.grasp_scorer is not None:
            body = self.task.bodies[best_oid]
            refined_pose = self.grasp_scorer.pick_pose(body, pick_pose)
            refined_position, refined_rotation = refined_pose
            body_position = p.getBasePositionAndOrientation(body)[0]
            refined_x = best_x.copy()
            refined_x[0] = float(refined_position[0] - body_position[0])
            refined_x[1] = float(refined_position[1] - body_position[1])
            refined_x[2] = float(p.getEulerFromQuaternion(refined_rotation)[2])
            refined_grasp = self._grasp(best_oid, 0, refined_x)
            if refined_grasp is not None:
                pick_pose, jaw_width = refined_grasp
                best_x = refined_x
        if self.structure == 'pyramid':
            goal = PYRAMID_SLOTS[len(placed)]
            place_xy = goal[:2]
            # Pick/place primitive expects gripper pose at the object's top.
            place_top = float(goal[2] + self.geometry[best_oid]['dims'][2] / 2)
        else:
            place_xy = (np.asarray(p.getBasePositionAndOrientation(self.task.bodies[placed[-1]])[0][:2])
                        if placed else CENTER)
            place_top = sum(self.geometry[i]['dims'][2] for i in placed) + self.geometry[best_oid]['dims'][2]
        place_pose = ((float(place_xy[0]+best_x[3]), float(place_xy[1]+best_x[4]), float(place_top)),
                      p.getQuaternionFromEuler((0., 0., float(best_x[5]))))
        return {'object_id': int(best_oid), 'pick': pick_pose, 'place': place_pose,
                'jaw_width': float(jaw_width), 'score': float(best_score),
                'grasp_yaw': float(best_x[2]), 'place_yaw': float(best_x[5])}


def run_episode(path, assets, gui, samples, iters, elites, settle_steps, seed, debug,
                structure='tower', grasp_scorer=None, grasp_robustness=None):
    meta = (pyramid_meta(seed) if structure == 'pyramid'
            else json.loads(Path(path).read_text()))
    task = DatasetSceneTask(meta, grasp_robustness=grasp_robustness,
                            grasp_seed=seed)
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
                           iters, elites, settle_steps, debug, structure=structure,
                           grasp_scorer=grasp_scorer)
        gate_before = ((grasp_scorer.selection_calls,
                        grasp_scorer.selection_changes,
                        grasp_scorer.selection_abstentions)
                       if grasp_scorer is not None else (0, 0, 0))
        print(f'{structure}: {len(meta["objects"])} objects', flush=True)
        for step in range(len(meta['objects'])):
            if grasp_scorer is not None:
                grasp_scorer.prepare(env._get_obs(), task.bodies.values())
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
            if structure == 'pyramid':
                stable = pyramid_stable(task, placed + [oid])
            else:
                stable = stable_tower(task, planner.geometry, placed + [oid], debug=debug)
            ok = bool(not timeout and task.primitive.last_grasp and stable)
            actions.append({'object_id': oid, 'pick': [list(a['pick'][0]), list(a['pick'][1])],
                            'place': [list(a['place'][0]), list(a['place'][1])],
                            'jaw_width': a['jaw_width'], 'cem_score': a['score'],
                            'grasp_attached': bool(task.primitive.last_grasp),
                            'grasp_robustness': getattr(env.ee, 'last_robustness', None),
                            'timeout': bool(timeout), 'stack_stable': bool(stable),
                            'success': ok})
            print(f'    executed: grasp={task.primitive.last_grasp}, '
                  f'stable={stable}, success={ok}', flush=True)
            if not ok:
                break
            placed.append(oid)
            if gui:
                time.sleep(0.3)
        gate_after = ((grasp_scorer.selection_calls,
                       grasp_scorer.selection_changes,
                       grasp_scorer.selection_abstentions)
                      if grasp_scorer is not None else (0, 0, 0))
        return {'episode': (Path(path).parent.name if path is not None else f'pyramid_seed_{seed}'), 'structure': structure, 'total': len(meta['objects']),
                'stacked': len(placed), 'success': len(placed) == len(meta['objects']),
                'order_chosen_by_cem': placed, 'rollouts': planner.rollouts,
                'grasp_gate_calls': gate_after[0] - gate_before[0],
                'grasp_gate_changes': gate_after[1] - gate_before[1],
                'grasp_gate_abstentions': gate_after[2] - gate_before[2],
                'actions': actions}
    finally:
        env.close()




def summarize(results, structure, method='CEM'):
    """Count all requested episodes, including planning and execution failures.

    Tower reward is the completion fraction; pyramid uses native Ravens reward.
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
    return {'Method': method + ' - ' + structure.title(), 'Episodes': n,
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
    ap.add_argument('--data', type=Path, default=Path('mixed-data-validated'))
    ap.add_argument('--assets', type=Path, default=Path('ravens/environments/assets'))
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--samples', type=int, default=48)
    ap.add_argument('--iters', type=int, default=4)
    ap.add_argument('--elites', type=int, default=8)
    ap.add_argument('--settle-steps', type=int, default=90)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--grasp-policy', choices=POLICIES, default='geometry')
    ap.add_argument('--grasp-checkpoint', default='')
    ap.add_argument('--grasp-mc-samples', type=int, default=8)
    ap.add_argument('--grasp-min-probability', type=float, default=0.65)
    ap.add_argument('--grasp-max-uncertainty', type=float, default=0.08)
    ap.add_argument('--grasp-min-margin', type=float, default=0.15)
    ap.add_argument('--grasp-refine-pose', action='store_true')
    ap.add_argument('--grasp-backend-arg', action='append', default=[])
    ap.add_argument('--grasp-backend-timeout', type=float, default=120)
    ap.add_argument('--grasp-max-tilt-degrees', type=float, default=20)
    ap.add_argument('--grasp-position-noise-mm', type=float, default=0.0)
    ap.add_argument('--grasp-yaw-noise-degrees', type=float, default=0.0)
    ap.add_argument('--grasp-slip-scale', type=float, default=0.0)
    ap.add_argument('--out', type=Path, default=Path('results/cem_hetero_results.json'))
    ap.add_argument('--summary-out', type=Path, default=Path('results/cem_hetero_summary.csv'))
    args = ap.parse_args()
    if (not 1 <= args.elites <= args.samples or args.episodes < 1 or
            args.start_index < 0 or args.grasp_position_noise_mm < 0 or
            args.grasp_yaw_noise_degrees < 0 or
            not 0 <= args.grasp_slip_scale <= 1):
        ap.error('Invalid samples/elites/episodes or grasp robustness setting')
    all_files = sorted(args.data.glob('episode_*/episode.json'))
    files = all_files[args.start_index:args.start_index + args.episodes]
    if not files:
        ap.error(f'No episode_*/episode.json found under {args.data}')
    grasp_scorer = create_grasp_planner(
        args.grasp_policy, checkpoint=args.grasp_checkpoint,
        mc_samples=args.grasp_mc_samples,
        refine_pose=args.grasp_refine_pose,
        min_probability=args.grasp_min_probability,
        max_uncertainty=args.grasp_max_uncertainty,
        min_margin=args.grasp_min_margin,
        backend_command=args.grasp_backend_arg,
        backend_timeout=args.grasp_backend_timeout,
        max_tilt_degrees=args.grasp_max_tilt_degrees)
    grasp_robustness = {
        'position_noise_mm': args.grasp_position_noise_mm,
        'yaw_noise_degrees': args.grasp_yaw_noise_degrees,
        'slip_scale': args.grasp_slip_scale,
    }
    results = []
    for i, file in enumerate(files):
        try:
            results.append(run_episode(file, args.assets, args.gui, args.samples,
                args.iters, args.elites, args.settle_steps,
                args.seed + args.start_index + i, args.debug, 'tower', grasp_scorer,
                grasp_robustness))
        except Exception as exc:
            print(f'{file}: {type(exc).__name__}: {exc}', flush=True)
            results.append({'episode': file.parent.name, 'success':False,
                            'stacked':0, 'actions':[], 'error':str(exc)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    method = ('CEM' if grasp_scorer is None else
              'CEM + ' + args.grasp_policy.replace('_', '-'))
    report = summarize(results, 'tower', method)
    with args.summary_out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(report))
        writer.writeheader(); writer.writerow(report)
    print_summary(report)
    print(f'Results: {args.out}; Summary: {args.summary_out}')

if __name__ == '__main__':
    main()
