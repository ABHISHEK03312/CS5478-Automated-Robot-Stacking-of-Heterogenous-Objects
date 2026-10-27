"""Replay geometry-oracle tower episodes into Ravens Dataset format.

Run from the Ravens repository root:
 python replay_stacking_to_ravens.py --data mixed-stacking-validated --out data/mixed-stacking-validated-train --assets_root ./ravens/environments/assets

IMPORTANT: Uses teleportation and a CUSTOM incremental pose-goal reward, not
Environment.step(), task.reward(), or verified parallel-jaw grasps.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pybullet as p
from ravens.environments.environment import Environment
from ravens.dataset import Dataset

BOUNDS = ((0.25, 0.75), (-0.50, 0.50), (0.0, 0.28))
COLORS = ((0.94, 0.34, 0.27, 1), (0.26, 0.58, 0.91, 1),
          (0.27, 0.75, 0.51, 1), (0.94, 0.70, 0.26, 1),
          (0.67, 0.47, 0.86, 1))


def make_body(obj, rgba):
    x, y, z = map(float, obj['dims'])
    shape = obj['shape']
    if shape in ('cube', 'rectangle'):
        kw = dict(shapeType=p.GEOM_BOX, halfExtents=[x/2, y/2, z/2])
        collision = p.createCollisionShape(**kw)
        visual = p.createVisualShape(**kw, rgbaColor=rgba)
    elif shape == 'cylinder':
        collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=x/2, height=z)
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=x/2, length=z,
                                     rgbaColor=rgba)
    elif shape == 'triangle':
        vertices = [[-x/2,-y/2,-z/2],[-x/2,y/2,-z/2],[-x/2,0,z/2],
                    [x/2,-y/2,-z/2],[x/2,y/2,-z/2],[x/2,0,z/2]]
        indices = [0,2,1, 3,4,5, 0,1,4, 0,4,3,
                   1,2,5, 1,5,4, 2,0,3, 2,3,5]
        collision = p.createCollisionShape(p.GEOM_MESH, vertices=vertices,
                                           indices=indices)
        visual = p.createVisualShape(p.GEOM_MESH, vertices=vertices,
                                     indices=indices, rgbaColor=rgba)
    else:
        raise ValueError('Unsupported shape: ' + str(shape))
    return p.createMultiBody(baseMass=0.12, baseCollisionShapeIndex=collision,
                             baseVisualShapeIndex=visual)


def valid_pose(xyz):
    return len(xyz) == 3 and all(lo <= float(v) <= hi
                                  for v, (lo, hi) in zip(xyz, BOUNDS))


def observe(env):
    # Exact camera list and image structure from the user's Environment class.
    return env._get_obs()


def replay(env, folder, settle_steps, pos_tolerance):
    meta = json.loads((folder / 'episode.json').read_text(encoding='utf-8'))
    objects = meta['objects']
    actions = meta['actions']
    if not actions or len(actions) != len(objects):
        raise ValueError('Expected one action per object')
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1/240)
    # Ground at z=0; no robot is simulated during teleport replay.
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE))
    env.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
    bodies = {}
    for obj in objects:
        object_id = int(obj['id'])
        dims = obj['dims']
        xy = obj['start_xy']
        pose = [float(xy[0]), float(xy[1]), float(dims[2])/2 + 0.001]
        if not valid_pose(pose):
            raise ValueError(f'Object {object_id} starts outside Ravens bounds')
        body = make_body(obj, COLORS[object_id % len(COLORS)])
        p.resetBasePositionAndOrientation(body, pose, [0, 0, 0, 1])
        bodies[object_id] = body
        env.obj_ids['rigid'].append(body)
    for _ in range(settle_steps):
        p.stepSimulation()

    episode = []
    awarded = set()
    n = len(actions)
    for step, source_action in enumerate(actions):
        obj_id = int(source_action['object_id'])
        if obj_id not in bodies or obj_id in awarded:
            raise ValueError(f'Duplicate or unknown object ID {obj_id}')
        pick = source_action['pick_xyz']
        place = source_action['place_xyz']
        if not valid_pose(pick) or not valid_pose(place):
            raise ValueError(f'Action {step} outside Ravens bounds: pick={pick}, place={place}, bounds={BOUNDS}')
        opening = float(source_action.get('gripper_opening_m', 0))
        if not 0 < opening <= 0.15:
            raise ValueError(f'Action {step} exceeds 15 cm gripper opening')

        obs = observe(env)
        q0 = p.getQuaternionFromEuler([0, 0, float(source_action.get('pick_yaw', 0))])
        q1 = p.getQuaternionFromEuler([0, 0, float(source_action.get('place_yaw', 0))])
        action = {'pose0': (tuple(map(float, pick)), tuple(q0)),
                  'pose1': (tuple(map(float, place)), tuple(q1))}
        # Object identity/opening are retained in info; vanilla Transporter
        # action heads do NOT predict opening width.
        info = {'source_episode': folder.name, 'step': step,
                'object_id': obj_id, 'shape': source_action['shape'],
                'gripper_opening_m': opening,
                'label_type': 'teleported_geometry_oracle',
                'objects': meta['objects']}

        body = bodies[obj_id]
        p.resetBasePositionAndOrientation(body, place, q1)
        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
        for _ in range(settle_steps):
            p.stepSimulation()
        actual, rotation = p.getBasePositionAndOrientation(body)
        roll, pitch, _ = p.getEulerFromQuaternion(rotation)
        goal_met = (np.linalg.norm(np.asarray(actual)-np.asarray(place)) <= pos_tolerance
                    and abs(roll) < 0.20 and abs(pitch) < 0.20)
        if goal_met:
            awarded.add(obj_id)
        reward = (1.0 / n) if goal_met else 0.0
        info['goal_met'] = bool(goal_met)
        info['reward_source'] = 'custom_pose_goal_not_ravens_task_reward'
        episode.append((obs, action, reward, info))
    terminal_info = {'source_episode': folder.name, 'terminal': True,
                     'reward_source': 'custom_pose_goal_not_ravens_task_reward',
                     'success': len(awarded) == n}
    episode.append((observe(env), None, 0.0, terminal_info))
    return episode, len(awarded) == n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('stacking_data_ravens'))
    parser.add_argument('--out', type=Path, default=Path('data/mixed-stacking-train'))
    parser.add_argument('--assets_root', default='./ravens/environments/assets/')
    parser.add_argument('--limit', type=int, default=0, help='0 means all episodes')
    parser.add_argument('--settle_steps', type=int, default=120)
    parser.add_argument('--pos_tolerance', type=float, default=0.012)
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()
    folders = sorted(x.parent for x in args.data.glob('episode_*/episode.json'))
    if args.limit:
        folders = folders[:args.limit]
    if not folders:
        raise SystemExit(f'No episode.json files found in {args.data}')
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(str(args.out))
    env = Environment(assets_root=args.assets_root, disp=args.gui)
    saved = failed = 0
    try:
        for folder in folders:
            try:
                episode, success = replay(env, folder, args.settle_steps,
                                          args.pos_tolerance)
                if not success:
                    print(f'SKIP {folder.name}: pose-goal check failed')
                    failed += 1
                    continue
                dataset.add(int(folder.name.split('_')[-1]), episode)
                saved += 1
                print(f'Saved {folder.name}: {len(episode)-1} actions')
            except Exception as exc:
                failed += 1
                print(f'SKIP {folder.name}: {type(exc).__name__}: {exc}')
    finally:
        env.close()
    print(f'Done: saved={saved}, skipped={failed}, output={args.out}')


if __name__ == '__main__':
    main()
