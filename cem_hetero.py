"""CEM placement planner for heterogeneous stacking_data episodes.

Standalone PyBullet planner, no Ravens imports and no learning. Uses geometry
for object ordering and PyBullet DIRECT snapshots for CEM candidate scoring.
Object transfers are teleports: two-jaw grasps are NOT tested.
"""
import math
import numpy as np
import pybullet as p

from generate_stacking_dataset import fits

TARGET_XY = (0.0, 0.18)  # reserved stacking area in dataset generator


def find_stack_order(objects):
    """Backtracking search for a complete bottom-to-top support-compatible order."""
    by_id = {int(o['id']): o for o in objects}
    remaining = tuple(by_id)

    def search(path, unused):
        if not unused:
            return path
        options = sorted(unused, key=lambda i: by_id[i]['dims'][0] * by_id[i]['dims'][1], reverse=True)
        for oid in options:
            if path and not fits(by_id[oid], by_id[path[-1]], margin=0.001):
                continue
            ans = search(path + [oid], tuple(i for i in unused if i != oid))
            if ans is not None:
                return ans
        return None

    order = search([], remaining)
    if order is None:
        raise ValueError('No full support-compatible stack order found in episode')
    return order


def stable_stack(objects, body_ids, order, xy=TARGET_XY, xy_tol=0.018,
                 height_tol=0.018, angle_tol=0.22):
    """Validate centered upright stack and expected heights after settling."""
    by_id = {int(o['id']): o for o in objects}
    expected_floor = 0.0
    for oid in order:
        obj = by_id[oid]
        pos, quat = p.getBasePositionAndOrientation(body_ids[oid])
        roll, pitch, _ = p.getEulerFromQuaternion(quat)
        expected_center_z = expected_floor + obj['dims'][2] / 2
        if (math.hypot(pos[0] - xy[0], pos[1] - xy[1]) > xy_tol or
                abs(pos[2] - expected_center_z) > height_tol or
                abs(roll) > angle_tol or abs(pitch) > angle_tol):
            return False
        expected_floor += obj['dims'][2]
    return True


class HeteroCEM:
    def __init__(self, objects, body_ids, samples=24, elites=5, iterations=3,
                 seed=0, settle_steps=180, spread=0.012, verbose=False):
        self.objects = {int(o['id']): o for o in objects}
        self.bodies = body_ids
        self.order = find_stack_order(objects)
        self.samples = int(samples)
        self.elites = int(elites)
        self.iterations = int(iterations)
        self.rng = np.random.default_rng(seed)
        self.settle_steps = int(settle_steps)
        self.spread = float(spread)
        self.verbose = verbose
        if self.samples < 2 or self.elites < 1 or self.elites > self.samples or self.iterations < 1:
            raise ValueError('Require samples >= 2, 1 <= elites <= samples, iterations >= 1')
        self.rollouts = 0

    def settle(self):
        for _ in range(self.settle_steps):
            p.stepSimulation()

    def evaluate(self, oid, xyz_yaw, placed):
        """Simulator rollout; reward is meaningful only if stack remains stable."""
        sid = p.saveState()
        self.rollouts += 1
        try:
            x, y, yaw = map(float, xyz_yaw)
            z = sum(self.objects[i]['dims'][2] for i in placed) + self.objects[oid]['dims'][2] / 2 + 0.001
            original_other = {
                i: p.getBasePositionAndOrientation(self.bodies[i])[0]
                for i in placed
            }
            p.resetBasePositionAndOrientation(
                self.bodies[oid], [x, y, z], p.getQuaternionFromEuler([0, 0, yaw]))
            p.resetBaseVelocity(self.bodies[oid], [0, 0, 0], [0, 0, 0])
            self.settle()
            candidate_order = placed + [oid]
            if not stable_stack(list(self.objects.values()), self.bodies, candidate_order):
                return -1000.0
            disturbance = sum(np.linalg.norm(
                np.array(p.getBasePositionAndOrientation(self.bodies[i])[0]) - np.array(pos))
                for i, pos in original_other.items())
            # Favor centered placements, minimal shaking of previous stack, no unnecessary yaw.
            return (10.0 - 80.0 * math.hypot(x-TARGET_XY[0], y-TARGET_XY[1])
                    - 30.0 * disturbance - 0.2 * abs(math.sin(yaw)))
        finally:
            p.restoreState(sid)
            p.removeState(sid)

    def optimize(self, oid, placed):
        mean = np.array([*TARGET_XY, 0.0], dtype=float)
        std = np.array([self.spread, self.spread, math.pi / 8], dtype=float)
        best_score, best = -float('inf'), None
        for k in range(self.iterations):
            proposals = mean + self.rng.normal(size=(self.samples, 3)) * std
            proposals[0] = mean  # evaluate exact center each round
            scored = [(self.evaluate(oid, v, placed), v.copy()) for v in proposals]
            scored.sort(key=lambda item: item[0], reverse=True)
            if scored[0][0] > best_score:
                best_score, best = scored[0]
            elite = np.array([v for _, v in scored[:self.elites]])
            # Yaw circular averaging around the current mean, avoiding pi wrap issues.
            elite[:, 2] = mean[2] + (elite[:, 2] - mean[2] + math.pi) % (2*math.pi) - math.pi
            mean = 0.2 * mean + 0.8 * elite.mean(axis=0)
            std = np.maximum(0.2 * std + 0.8 * elite.std(axis=0), [0.001, 0.001, 0.025])
            if self.verbose:
                print(f'   CEM iter {k+1}: best={scored[0][0]:.3f} std={std.round(3)}')
        if best_score < 0:
            return None, best_score
        return best, best_score

    def place(self, oid, action, placed):
        x, y, yaw = map(float, action)
        z = sum(self.objects[i]['dims'][2] for i in placed) + self.objects[oid]['dims'][2] / 2 + 0.001
        p.resetBasePositionAndOrientation(self.bodies[oid], [x, y, z], p.getQuaternionFromEuler([0, 0, yaw]))
        p.resetBaseVelocity(self.bodies[oid], [0, 0, 0], [0, 0, 0])
        self.settle()
        return stable_stack(list(self.objects.values()), self.bodies, placed + [oid])
