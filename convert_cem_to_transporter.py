"""Re-render existing stacking_data episodes as original Ravens Transporter PKLs.

Run from the Ravens repository root:
    python convert_cem_to_transporter.py --data stacking_data --out data/mixed-shape-stacking-train

Requires pybullet and numpy. Does NOT run an oracle or invent CEM actions:
uses the stored geometry-oracle demonstration sequence from episode.json.
Three Ravens RealSense D415 camera views are newly rendered from the same
object geometry; this is not a byte-for-byte reuse of original 256px images.

Source table coordinates are centered at x=0, while stock Ravens workspace
is centered at x=0.5. To keep all source object CENTERS within the stock
Ravens workspace, source table x is mapped to x=0.5 + 0.90*x.
This slightly changes starting x positions, but preserves shapes, IDs, order,
relative placements, and object dimensions. No real grasp success is verified.
"""
import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pybullet as p

# Same color list as generate_stacking_dataset.py.
COLORS = [[0.94, 0.34, 0.27, 1], [0.26, 0.58, 0.91, 1],
          [0.27, 0.75, 0.51, 1], [0.94, 0.70, 0.26, 1],
          [0.67, 0.47, 0.86, 1]]


def transformed_xyz(xyz):
    return (0.5 + 0.90 * float(xyz[0]), float(xyz[1]), float(xyz[2]))


def quat_yaw(yaw):
    return (0., 0., math.sin(yaw / 2.), math.cos(yaw / 2.))


def make_body(obj):
    kind = obj['shape']
    x, y, z = map(float, obj['dims'])
    if kind in ('cube', 'rectangle'):
        coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=[x/2, y/2, z/2])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[x/2, y/2, z/2],
                                  rgbaColor=COLORS[int(obj['id']) % len(COLORS)])
    elif kind == 'cylinder':
        coll = p.createCollisionShape(p.GEOM_CYLINDER, radius=x/2, height=z)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=x/2, length=z,
                                  rgbaColor=COLORS[int(obj['id']) % len(COLORS)])
    elif kind == 'triangle':
        vertices = [[-x/2,-y/2,-z/2], [-x/2,y/2,-z/2], [-x/2,0,z/2],
                    [x/2,-y/2,-z/2], [x/2,y/2,-z/2], [x/2,0,z/2]]
        faces = [0,2,1, 3,4,5, 0,1,4, 0,4,3,
                 1,2,5, 1,5,4, 2,0,3, 2,3,5]
        coll = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, indices=faces)
        vis = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=faces,
                                  rgbaColor=COLORS[int(obj['id']) % len(COLORS)])
    else:
        raise ValueError('Unknown shape: ' + kind)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=coll,
                             baseVisualShapeIndex=vis)


def render_view(config):
    """Matches Ravens Environment.render_camera geometry and depth units."""
    h, w = config['image_size']
    rotation = np.asarray(p.getMatrixFromQuaternion(config['rotation'])).reshape(3, 3)
    camera_pos = np.asarray(config['position'], dtype=float)
    lookat = camera_pos + rotation @ np.array([0., 0., 1.])
    up = rotation @ np.array([0., -1., 0.])
    view = p.computeViewMatrix(camera_pos, lookat, up)
    focal = config['intrinsics'][0]
    fov = 2 * np.degrees(np.arctan((h / 2) / focal))
    near, far = config['zrange']
    proj = p.computeProjectionMatrixFOV(fov, float(w) / h, near, far)
    _, _, rgba, zbuf, _ = p.getCameraImage(
        w, h, viewMatrix=view, projectionMatrix=proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    zbuf = np.asarray(zbuf, dtype=np.float32).reshape(h, w)
    depth = (2 * near * far) / (far + near - (2 * zbuf - 1) * (far - near))
    return rgb, depth.astype(np.float32)


def make_info(meta, body_by_id, step, final=False):
    info = {}
    for obj in meta['objects']:
        oid = int(obj['id'])
        pos, quat = p.getBasePositionAndOrientation(body_by_id[oid])
        info[body_by_id[oid]] = (tuple(pos), tuple(quat), tuple(obj['dims']))
    # Also store source metadata with noninteger keys (Transporter ignores info).
    info['source_episode'] = int(meta['episode'])
    info['oracle_label_only'] = True
    info['step'] = int(step)
    info['final'] = bool(final)
    return info


def convert_one(meta_path, dest, index, cameras):
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    objects = meta['objects']
    labels = meta['actions']
    if not labels or len(labels) != len(objects):
        raise ValueError('Incomplete episode: ' + str(meta_path))
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE))
    body_by_id = {}
    for obj in objects:
        oid = int(obj['id'])
        body_by_id[oid] = make_body(obj)
        pos = transformed_xyz((*obj['start_xy'], float(obj['dims'][2])/2))
        p.resetBasePositionAndOrientation(body_by_id[oid], pos, [0, 0, 0, 1])

    all_color, all_depth, all_actions, all_rewards, all_info = [], [], [], [], []
    # An episode is N action observations plus one terminal observation.
    for step in range(len(labels) + 1):
        views = [render_view(camera) for camera in cameras]
        all_color.append(np.stack([view[0] for view in views], axis=0))
        all_depth.append(np.stack([view[1] for view in views], axis=0))
        is_final = step == len(labels)
        all_info.append(make_info(meta, body_by_id, step, is_final))
        if is_final:
            all_actions.append(None)
            all_rewards.append(1.0 / len(labels))
            break
        label = labels[step]
        pick = (transformed_xyz(label['pick_xyz']), quat_yaw(float(label['pick_yaw'])))
        place = (transformed_xyz(label['place_xyz']), quat_yaw(float(label['place_yaw'])))
        all_actions.append({'pose0': pick, 'pose1': place})
        # Ravens stores PREVIOUS transition reward at each action observation.
        all_rewards.append(0.0 if step == 0 else 1.0 / len(labels))
        oid = int(label['object_id'])
        p.resetBasePositionAndOrientation(body_by_id[oid], place[0], [0, 0, 0, 1])

    filename = f'{index:06d}-{2*index}.pkl'
    values = {
        'color': np.asarray(all_color, dtype=np.uint8),
        'depth': np.asarray(all_depth, dtype=np.float32),
        'action': all_actions,
        'reward': all_rewards,
        'info': all_info,
    }
    for name, contents in values.items():
        folder = dest / name
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / filename).open('wb') as f:
            pickle.dump(contents, f, protocol=pickle.HIGHEST_PROTOCOL)
    return len(labels), sum(float(obj['dims'][2]) for obj in objects)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path('stacking_data'))
    ap.add_argument('--out', type=Path, default=Path('data/mixed-shape-stacking-train'))
    ap.add_argument('--limit', type=int, default=None, help='Convert first N episodes for testing')
    args = ap.parse_args()
    episodes = sorted(args.data.glob('episode_*/episode.json'))
    if args.limit is not None:
        episodes = episodes[:args.limit]
    if not episodes:
        ap.error('No source episode_*/episode.json files under ' + str(args.data))
    if list((args.out / 'action').glob('*.pkl')):
        ap.error('Output already has episodes; choose an empty --out folder')
    try:
        from ravens.tasks import cameras
    except ImportError as e:
        ap.error('Run from your Ravens repository root (and activate CS5478): ' + str(e))
    client = p.connect(p.DIRECT)
    oversized = []
    n_steps = 0
    try:
        for i, path in enumerate(episodes):
            n, height = convert_one(path, args.out, i, cameras.RealSenseD415.CONFIG)
            n_steps += n
            if height > 0.28:
                oversized.append((path.parent.name, round(height, 3)))
            if (i + 1) % 10 == 0 or i + 1 == len(episodes):
                print(f'{i+1}/{len(episodes)} episodes, {n_steps} actions', flush=True)
    finally:
        p.disconnect(client)
    print(f'Wrote {len(episodes)} episodes to {args.out}')
    print('Ravens data shape: color [steps,3,480,640,3], depth [steps,3,480,640]')
    print('Includes final observation and top-level pose0/pose1 actions.')
    print('Synthetic rewards are labels, not measured robot rewards.')
    if oversized:
        print(f'WARNING: {len(oversized)} stacks exceed stock Ravens Z bound 0.28 m.')
        print('Stock Transporter will crop their upper portions. Examples:', oversized[:5])
        print('For those scenes, Transporter bounds/cameras must be adapted or episodes filtered.')
    print('Note: scenes are re-rendered, and table x positions are slightly rescaled.')
    print('This only converts the dataset; it does not validate real robot grasping.')


if __name__ == '__main__':
    main()
