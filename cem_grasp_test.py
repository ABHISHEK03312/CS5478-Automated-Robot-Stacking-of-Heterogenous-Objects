"""Bridge heterogeneous CEM stacking_data scenes to the Ravens UR5 parallel jaw.

Run from repository root:
 python cem_grasp_test.py --data stacking_data --episodes 1 --gui

This uses a kinematic, constraint-based grasp, NOT physical fingertip friction.
Continuous placements are optimized by teleport simulation; executed by UR5 IK.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pybullet as p

from cem_hetero import find_stack_order
from generate_stacking_dataset import make_body, COLORS
from ravens.environments.environment import Environment
from ravens.tasks.grippers import ParallelJaw
from ravens.tasks.primitives import PickPlaceParallelJaw

TARGET = (0.5, 0.18)  # Dataset x + 0.5, to fit the UR5 workspace.


class DatasetParallelJaw(ParallelJaw):
    """Reject grasps wider than jaw opening, using dataset geometry.

    Keeps the existing geometric, fixed-constraint grasp model. Not a force model.
    Gripper local X is assumed to be the closing direction; calibrate if needed.
    """

    def activate(self):
        pose = p.getBasePositionAndOrientation(self.body)
        inv = p.invertTransform(*pose)
        candidates = []
        for body in self.obj_ids['rigid']:
            if body not in getattr(self, 'geometry', {}):
                continue
            obj = self.geometry[body]
            opos = p.getBasePositionAndOrientation(body)
            local_pos, local_rot = p.multiplyTransforms(inv[0], inv[1], *opos)
            lx, ly, lz = local_pos
            if abs(lx) > 0.028 or abs(ly) > 0.028 or abs(lz - self.TOOL_CENTER_Z) > 0.05:
                continue
            length, width, _ = obj['dims']
            if obj['shape'] == 'cylinder':
                jaw_width = length
            else:
                # Project rectangular bounding footprint onto jaw closing axis.
                yaw = p.getEulerFromQuaternion(local_rot)[2]
                jaw_width = abs(length * math.cos(yaw)) + abs(width * math.sin(yaw))
            if jaw_width + 0.002 <= self.OPEN_INNER_WIDTH:
                error = lx * lx + ly * ly + (lz - self.TOOL_CENTER_Z) ** 2
                candidates.append((error, body, (local_pos, local_rot), jaw_width))
        if not candidates:
            self.activated = False
            return
        _, body, local_pose, width = min(candidates, key=lambda item: item[0])
        closure = np.clip((self.OPEN_INNER_WIDTH - width) / 2 - 0.001, 0, 0.03)
        self._command(closure, steps=60)
        self.contact_constraint = p.createConstraint(
            parentBodyUniqueId=self.body, parentLinkIndex=-1,
            childBodyUniqueId=body, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=(0, 0, 0),
            parentFramePosition=local_pose[0],
            parentFrameOrientation=local_pose[1],
            childFramePosition=(0, 0, 0),
            childFrameOrientation=(0, 0, 0, 1))
        self.activated = True


class CheckedPickPlace(PickPlaceParallelJaw):
    """Record whether a grasp was actually attached during the UR5 motion."""

    def __call__(self, movej, movep, ee, pose0, pose1):
        self.last_grasp = False
        identity = (0, 0, 0, 1)
        # Use WORLD vertical offsets: poses use world-aligned yaw only.
        def shifted(pose, dz):
            return ((pose[0][0], pose[0][1], pose[0][2] + dz), pose[1])
        pick = shifted(pose0, -self.grasp_offset)
        place = shifted(pose1, -self.grasp_offset)
        pre_pick = shifted(pick, self.height)
        pre_place = shifted(place, self.height)
        ee.release()
        if movep(pre_pick) or movep(pick, self.speed):
            return True
        ee.activate()
        if not ee.check_grasp():
            ee.release()
            movep(pre_pick)
            return True
        self.last_grasp = True
        if movep(pre_pick, self.speed) or movep(pre_place) or movep(place, self.speed):
            ee.release()
            return True
        ee.release()
        return bool(movep(pre_place, self.speed))


class DatasetSceneTask:
    def __init__(self, meta):
        self.meta = meta
        self.ee = DatasetParallelJaw
        self.primitive = CheckedPickPlace(height=0.14, speed=0.01)
        self.bodies = {}
        self.max_steps = len(meta['objects'])

    def set_assets_root(self, assets_root):
        self.assets_root = assets_root

    def reset(self, env):
        self.bodies = {}
        geom = {}
        for obj in self.meta['objects']:
            oid = int(obj['id'])
            body = make_body(obj['shape'], obj['dims'], COLORS[oid % len(COLORS)])
            start = [float(obj['start_xy'][0]) + 0.5,
                     float(obj['start_xy'][1]), float(obj['dims'][2]) / 2 + 0.001]
            p.resetBasePositionAndOrientation(body, start, [0, 0, 0, 1])
            env.obj_ids['rigid'].append(body)
            self.bodies[oid] = body
            geom[body] = obj
        env.ee.geometry = geom
        for _ in range(80):
            p.stepSimulation()

    def reward(self):
        return 0.0, {}

    def done(self):
        return False


def stable(meta, bodies, placed, center=TARGET, xy_tol=0.018, z_tol=0.018):
    by_id = {int(o['id']): o for o in meta['objects']}
    z = 0.0
    for oid in placed:
        body = bodies[oid]
        obj = by_id[oid]
        pos, rot = p.getBasePositionAndOrientation(body)
        roll, pitch, _ = p.getEulerFromQuaternion(rot)
        expected_z = z + obj['dims'][2] / 2
        if (math.hypot(pos[0]-center[0], pos[1]-center[1]) > xy_tol
                or abs(pos[2]-expected_z) > z_tol
                or abs(roll) > 0.2 or abs(pitch) > 0.2):
            return False
        z += obj['dims'][2]
    return True


def settle(steps=180):
    for _ in range(steps):
        p.stepSimulation()


def cem_place(meta, bodies, oid, placed, rng, samples=20, iters=3):
    """Optimizes XY and yaw with reversible teleport rollouts (fast surrogate)."""
    by_id = {int(o['id']): o for o in meta['objects']}
    height = sum(by_id[i]['dims'][2] for i in placed)
    candidate_z = height + by_id[oid]['dims'][2] / 2 + 0.002
    mean = np.array([*TARGET, 0.0])
    std = np.array([0.012, 0.012, 0.25])
    best = (float('-inf'), None)
    for _ in range(iters):
        acts = mean + rng.normal(size=(samples, 3)) * std
        acts[0] = mean
        scored = []
        for a in acts:
            state = p.saveState()
            try:
                prev = {i: np.asarray(p.getBasePositionAndOrientation(bodies[i])[0])
                        for i in placed}
                p.resetBasePositionAndOrientation(
                    bodies[oid], [float(a[0]), float(a[1]), candidate_z],
                    p.getQuaternionFromEuler([0, 0, float(a[2])]))
                p.resetBaseVelocity(bodies[oid], [0, 0, 0], [0, 0, 0])
                settle(120)
                if stable(meta, bodies, placed + [oid]):
                    disturbance = sum(np.linalg.norm(
                        np.asarray(p.getBasePositionAndOrientation(bodies[i])[0]) - prev[i])
                        for i in placed)
                    score = (10 - 70 * math.hypot(a[0]-TARGET[0], a[1]-TARGET[1])
                             - 25 * disturbance)
                else:
                    score = -1000.0
            finally:
                p.restoreState(state)
                p.removeState(state)
            scored.append((score, a.copy()))
        scored.sort(reverse=True, key=lambda item: item[0])
        if scored[0][0] > best[0]:
            best = scored[0]
        elite = np.array([a for _, a in scored[:min(5, samples)]])
        mean = 0.2 * mean + 0.8 * elite.mean(axis=0)
        std = np.maximum(0.2 * std + 0.8 * elite.std(axis=0), [0.001, 0.001, 0.03])
    return best[1] if best[0] > 0 else None


def pick_pose_for(obj, body, gripper):
    """Try to orient the jaws across the shorter horizontal dimension."""
    x, y, _ = p.getBasePositionAndOrientation(body)[0]
    length, width, height = obj['dims']
    if min(length, width) + 0.002 > gripper.OPEN_INNER_WIDTH:
        return None
    # Assume closing direction is local X, with pick action yaw around Z.
    yaw = 0.0 if length <= width else math.pi / 2
    quat = p.getQuaternionFromEuler([0, 0, yaw])
    top_z = p.getAABB(body)[1][2]
    return ((x, y, top_z), quat)


def run_episode(path, assets, gui=False, samples=20, iters=3, seed=0):
    meta = json.loads(Path(path).read_text())
    task = DatasetSceneTask(meta)
    env = Environment(str(assets), task=task, disp=gui, hz=240)
    results = []
    try:
        env.seed(seed)
        env.reset()
        order = find_stack_order(meta['objects'])
        by_id = {int(o['id']): o for o in meta['objects']}
        placed = []
        rng = np.random.default_rng(seed)
        print(f'{Path(path).parent.name}: {len(order)} objects, order={order}')
        for oid in order:
            obj = by_id[oid]
            pick = pick_pose_for(obj, task.bodies[oid], env.ee)
            if pick is None:
                print(f'  object {oid} ({obj["shape"]}): too wide for 78mm jaws')
                break
            target = cem_place(meta, task.bodies, oid, placed, rng, samples, iters)
            if target is None:
                print(f'  object {oid}: CEM found no stable placement')
                break
            z = sum(by_id[i]['dims'][2] for i in placed) + obj['dims'][2]
            # Placement pose is defined at the top of the object, like Ravens.
            place = ((float(target[0]), float(target[1]), z),
                     p.getQuaternionFromEuler([0, 0, float(target[2]) + p.getEulerFromQuaternion(pick[1])[2]]))
            # Avoid env.step() here: we need explicit grasp outcome and bounded settling.
            timed_out = task.primitive(env.movej, env.movep, env.ee, pick, place)
            settle(180)
            ok = (not timed_out and task.primitive.last_grasp and
                  stable(meta, task.bodies, placed + [oid]))
            results.append({'object_id': oid, 'shape': obj['shape'],
                            'grasp_attached': bool(task.primitive.last_grasp),
                            'timeout': bool(timed_out), 'stack_stable': bool(ok)})
            print(f'  {oid}: grasp={task.primitive.last_grasp}, stack_ok={ok}')
            if not ok:
                break
            placed.append(oid)
            if gui:
                time.sleep(0.5)
        return {'episode': Path(path).parent.name, 'stacked': len(placed),
                'total': len(order), 'success': len(placed) == len(order),
                'actions': results}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('stacking_data'))
    parser.add_argument('--assets', type=Path, default=Path('ravens/environments/assets'))
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('cem_grasp_results.json'))
    args = parser.parse_args()
    files = sorted(args.data.glob('episode_*/episode.json'))[:args.episodes]
    if not files:
        parser.error('No episode.json files found in --data')
    results = [run_episode(f, args.assets, args.gui, args.samples, args.iters, i + 1)
               for i, f in enumerate(files)]
    args.out.write_text(json.dumps(results, indent=2))
    print(f'Success: {sum(r["success"] for r in results)}/{len(results)}; saved {args.out}')

if __name__ == '__main__':
    main()
