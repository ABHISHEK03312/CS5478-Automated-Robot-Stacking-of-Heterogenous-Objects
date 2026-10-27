"""Generate synthetic stacking demonstrations for a Transporter-style policy.

Install: pip install pybullet numpy pillow
Run: python generate_stacking_dataset.py --episodes 1000 --out stacking_data

These are *geometry-oracle* demonstrations. The object is teleported from its
source pose to its target pose; two-finger grasp feasibility is NOT tested.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image

TABLE_HALF = 0.36
CAM_RES = 256
CAM_SPAN = 0.80
COLORS = [[0.94, 0.34, 0.27, 1], [0.26, 0.58, 0.91, 1],
          [0.27, 0.75, 0.51, 1], [0.94, 0.70, 0.26, 1],
          [0.67, 0.47, 0.86, 1]]

#Generates 3D object in PyBullet - cube/rectangle/triangular prism/cylinder
def make_body(shape, dims, rgba):
    """dims: (length,width,height), including the cylinder bounding diameter."""
    x, y, z = dims
    if shape in ('cube', 'rectangle'):
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[x/2, y/2, z/2])
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[x/2, y/2, z/2], rgbaColor=rgba)
    elif shape == 'cylinder':
        collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=x/2, height=z)
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=x/2, length=z, rgbaColor=rgba)
    else:  # triangular prism: length x, triangular cross-section in yz plane
        vertices = [[-x/2,-y/2,-z/2],[-x/2,y/2,-z/2],[-x/2,0,z/2],
                    [ x/2,-y/2,-z/2],[ x/2,y/2,-z/2],[ x/2,0,z/2]]
        faces = [0,2,1, 3,4,5, 0,1,4, 0,4,3,
                 1,2,5, 1,5,4, 2,0,3, 2,3,5]
        collision = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, indices=faces)
        visual = p.createVisualShape(p.GEOM_MESH, vertices=vertices,
                                     indices=faces, rgbaColor=rgba)
    return p.createMultiBody(baseMass=0.12, baseCollisionShapeIndex=collision,
                             baseVisualShapeIndex=visual, basePosition=[0,0,z/2])

#Generates a random shape (cube/rectangle/cylinder) with random dimensions
def random_shape(rng):
    kind = str(rng.choice(['cube','rectangle','triangle','cylinder']))
    if kind == 'cube':
        side = float(rng.uniform(0.035, 0.075))
        dims = [side, side, side]
    elif kind == 'rectangle':
        dims = [float(rng.uniform(0.045, 0.090)),
                float(rng.uniform(0.032, 0.065)),
                float(rng.uniform(0.025, 0.060))]
    elif kind == 'cylinder':
        diameter = float(rng.uniform(0.040, 0.080))
        dims = [diameter, diameter, float(rng.uniform(0.030, 0.065))]
    else:
        dims = [float(rng.uniform(0.045, 0.085)),
                float(rng.uniform(0.040, 0.075)),
                float(rng.uniform(0.030, 0.060))]
    return kind, dims

def sample_table_positions(rng, objects):
    """Choose non-overlapping initial XY locations; avoid the stacking target."""
    for obj in objects:
        radius = 0.5 * math.hypot(*obj['dims'][:2])
        for _ in range(2000):
            xy = rng.uniform(-0.27, 0.27, 2)
            if np.linalg.norm(xy - np.array([0, 0.18])) < radius + 0.10:
                continue
            if all(np.linalg.norm(xy - np.array(old['start_xy'])) >
                   radius + 0.5*math.hypot(*old['dims'][:2]) + 0.020
                   for old in objects if 'start_xy' in old):
                obj['start_xy'] = xy.tolist()
                break
        else:
            return False
    return True

def support_halfwidths(obj):
    if obj['shape'] == 'triangle':
        return (0, 0)  #sloping top: don't use as a support
    x, y, _ = obj['dims']
    if obj['shape'] == 'cylinder':
        return (x / (2*math.sqrt(2)), y / (2*math.sqrt(2)))
    return (x/2, y/2)

#checks if one object can fit on top of another
def fits(top, below, margin=0.004):
    bx, by = support_halfwidths(below)
    tx, ty, _ = top['dims']
    if top['shape'] == 'cylinder':
        #ensure disk fits in the support rectangle
        return tx/2 + margin <= bx and ty/2 + margin <= by
    if top['shape'] == 'triangle':
        #base is a rectangle even though upper surface is sloped
        return tx/2 + margin <= bx and ty/2 + margin <= by
    return tx/2 + margin <= bx and ty/2 + margin <= by

def generate_full_stack_objects(rng, n):
    """Construct a heterogeneous, support-compatible stack containing ALL n objects.

    Work from the top downward, making each support larger than the item above.
    A triangle can only be the top item because its upper face is sloping.
    """
    for _ in range(2000):
        top_shape = str(rng.choice(['cube', 'rectangle', 'triangle', 'cylinder']))
        width = float(rng.uniform(0.019, 0.026))
        top = {'shape': top_shape,
               'dims': [width, width, float(rng.uniform(0.018, 0.035))]}
        from_top = [top]
        for level in range(1, n):
            above = from_top[-1]
            parent_shape = str(rng.choice(['cube', 'rectangle', 'cylinder']))
            aw, ah = above['dims'][:2]
            #cylinder's circular upper surface needs a larger diameter
            #for the bounding footprint of the next object to fit.
            multiplier = float(rng.uniform(1.13, 1.25))
            if parent_shape == 'cylinder':
                side = math.sqrt(2) * max(aw, ah) * multiplier
                dims = [side, side, float(rng.uniform(0.022, 0.045))]
            elif parent_shape == 'cube':
                side = max(aw, ah) * multiplier
                dims = [side, side, side]
            else:
                dims = [aw * multiplier, ah * multiplier,
                        float(rng.uniform(0.022, 0.05))]
            from_top.append({'shape': parent_shape, 'dims': dims})
        stack = list(reversed(from_top))  # base -> top
        if (max(stack[0]['dims'][:2]) <= 0.15 and
                all(fits(stack[i+1], stack[i], margin=0.001)
                    for i in range(n-1))):
            #Permute identities so pick order isn't just object ID order.
            ids = rng.permutation(n)
            for obj, object_id in zip(stack, ids):
                obj['id'] = int(object_id)
            return sorted(stack, key=lambda o: o['id']), [o['id'] for o in stack]
    return None, None


def stack_settled(stack, bodies):
    """Check that all items remain centered, upright and at their stack heights."""
    z = 0.0
    for obj in stack:
        pos, quat = p.getBasePositionAndOrientation(bodies[obj['id']])
        roll, pitch, _ = p.getEulerFromQuaternion(quat)
        expected_z = z + obj['dims'][2]/2
        if (math.hypot(pos[0], pos[1]-0.18) > 0.012 or
                abs(pos[2] - expected_z) > 0.012 or
                abs(roll) > 0.20 or abs(pitch) > 0.20):
            return False
        z += obj['dims'][2]
    return True


#captures top-down RGB-D images
def camera():
    view = p.computeViewMatrix(cameraEyePosition=[0, 0, 1.0],
                               cameraTargetPosition=[0, 0, 0],
                               cameraUpVector=[0, 1, 0])
    half = CAM_SPAN/2
    proj = p.computeProjectionMatrix(-half, half, -half, half, 0.01, 2.0)
    _, _, rgba, depth, _ = p.getCameraImage(CAM_RES, CAM_RES, view, proj,
                                            renderer=p.ER_TINY_RENDERER)
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(CAM_RES, CAM_RES, 4)[:, :, :3]
    buf = np.asarray(depth, dtype=np.float32).reshape(CAM_RES, CAM_RES)
    near, far = 0.01, 2.0
    depth_m = far*near/(far-(far-near)*buf)
    return rgb, depth_m, view, proj

#converts 3D world coordinates to 2D pixel coordinates in the camera image
def pixel_from_world(xyz, view, proj):
    V = np.array(view).reshape(4, 4, order='F')
    P = np.array(proj).reshape(4, 4, order='F')
    q = P @ V @ np.array([*xyz, 1.0])
    q /= q[3]
    u = (q[0]+1)*0.5*CAM_RES
    v = (1-q[1])*0.5*CAM_RES
    return [float(u), float(v)]

#executes a single episode of stacking n objects, saving RGB-D images
def generate_episode(rng, episode, out):
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1/240)
    p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE))
    n = int(rng.integers(3, 8))
    objects, stack_ids = generate_full_stack_objects(rng, n)
    if objects is None:
        return 0
    if not sample_table_positions(rng, objects):
        return 0
    for obj in objects:
        obj['body'] = make_body(obj['shape'], obj['dims'], COLORS[obj['id'] % len(COLORS)])
        yaw = 0.0  # align objects with their labeled grasps in v1
        pos = [*obj['start_xy'], obj['dims'][2]/2 + 0.001]
        p.resetBasePositionAndOrientation(obj['body'], pos,
                                           p.getQuaternionFromEuler([0, 0, yaw]))
    for _ in range(120):
        p.stepSimulation()
    by_id = {o['id']: o for o in objects}
    stack = [by_id[i] for i in stack_ids]
    bodies = {o['id']: o['body'] for o in objects}
    folder = out / f'episode_{episode:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    z_level = 0.0
    actions = []
    #Geometry-based teleports, rather than robot-arm executions.
    #A physical stability check is made after each placement.
    for step, obj in enumerate(stack):
        rgb, depth, view, proj = camera()
        Image.fromarray(rgb).save(folder / f'rgb_{step:02d}.png')
        np.save(folder / f'depth_{step:02d}.npy', depth)
        src, quat = p.getBasePositionAndOrientation(obj['body'])
        yaw = p.getEulerFromQuaternion(quat)[2]
        target = [0.0, 0.18, z_level + obj['dims'][2]/2]
        action = {
            'object_id': obj['id'], 'shape': obj['shape'],
            'rgb': f'rgb_{step:02d}.png', 'depth': f'depth_{step:02d}.npy',
            'pick_xyz': list(src), 'pick_yaw': yaw,
            'place_xyz': target, 'place_yaw': 0.0,
            'pick_pixel_uv': pixel_from_world(src, view, proj),
            'place_pixel_uv': pixel_from_world(target, view, proj),
            'gripper_opening_m': min(obj['dims'][:2]) + 0.008,
            'label_type': 'geometry_oracle_unverified_grasp',
        }
        actions.append(action)
        p.resetBasePositionAndOrientation(obj['body'], target, [0, 0, 0, 1])
        p.resetBaseVelocity(obj['body'], [0, 0, 0], [0, 0, 0])
        for _ in range(120):
            p.stepSimulation()
        z_level += obj['dims'][2]
        if not stack_settled(stack[:step+1], bodies):
            #This scene is not a validated stack; discard and retry.
            import shutil
            shutil.rmtree(folder)
            return 0
    #Save final positions for downstream analysis; not a grasp-success label.
    final = {}
    for obj in objects:
        pos, quat = p.getBasePositionAndOrientation(obj['body'])
        final[str(obj['id'])] = {'position': pos, 'orientation_xyzw': quat}
    meta = {'episode': episode, 'num_objects': n,
            'stack_object_ids': [o['id'] for o in stack],
            'all_objects_stacked': len(stack) == n,
            'objects': [{k:v for k,v in obj.items() if k != 'body'} for obj in objects],
            'actions': actions, 'final_states': final,
            'simulation_note': 'All objects stacked; physics settling checked. Objects teleported; physical two-jaw grasps NOT verified.'}
    (folder / 'episode.json').write_text(json.dumps(meta, indent=2))
    return len(actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--out', type=Path, default=Path('stacking_data'))
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    client = p.connect(p.DIRECT)
    try:
        count, actions = 0, 0
        # Retry when the full stack fails generation or settling checks.
        for attempt in range(args.episodes * 50):
            if count >= args.episodes:
                break
            steps = generate_episode(rng, count, args.out)
            if steps:
                count += 1
                actions += steps
            if count and count % 100 == 0 and steps:
                print(f'{count} episodes, {actions} labeled actions')
        print(f'Done: {count} episodes, {actions} actions in {args.out}')
    finally:
        p.disconnect(client)


if __name__ == '__main__':
    main()
