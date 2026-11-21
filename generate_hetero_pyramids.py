"""Generate diverse heterogeneous 2–4 layer pyramid episodes for Ravens replay.

pip install numpy pybullet pillow
python generate_hetero_pyramids.py --episodes 100 --out hetero-pyramid-episodes
python generate_hetero_pyramids.py --episodes 20 --layers 3 --gui

Pyramid layers contain L, L-1, ..., 1 objects. Cylinders are deliberately
common (including in base and intermediate layers). Other objects are cubes,
rectangular prisms, vertical cylinders, and apex triangular prisms. Layer heights
vary, as do object dimensions and lengths. Each completed episode is checked
against PyBullet settling (with retries), but grasp actions are teleported
geometry-oracle demonstrations, NOT robot-executed parallel-jaw grasps.
"""
import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image

BOUNDS = ((0.25, 0.75), (-0.50, 0.50), (0, 0.28))
COLORS = ((.94,.34,.27,1),(.26,.58,.91,1),(.27,.75,.51,1),
          (.94,.70,.26,1),(.67,.47,.86,1),(.18,.76,.77,1),(.95,.45,.65,1))
CENTER = (.50, 0.)
RES = 256
GRIPPER_LIMIT = .15
BOUNDS_MARGIN = 0.005  # keep all pick/place centers away from workspace edges


def valid_pose(pose, margin=BOUNDS_MARGIN):
    return (len(pose) == 3 and all(
        math.isfinite(float(v)) and (lo + margin) <= float(v) <= (hi - margin)
        for v, (lo, hi) in zip(pose, BOUNDS)))


def within_ravens_bounds(objects, actions):
    # Require valid source poses, goal poses, and actual recorded pick poses.
    return (all(valid_pose([*o['start_xy'], o['dims'][2]/2 + .001])
                and valid_pose(o['goal_xyz']) for o in objects)
            and all(valid_pose(a['pick_xyz']) and valid_pose(a['place_xyz'])
                    and 0 < a['gripper_opening_m'] <= GRIPPER_LIMIT
                    for a in actions))



def create_body(obj, color):
    x, y, z = obj['dims']
    kind = obj['shape']
    if kind in ('cube', 'rectangle'):
        kw = dict(shapeType=p.GEOM_BOX, halfExtents=[x/2, y/2, z/2])
        c, v = p.createCollisionShape(**kw), p.createVisualShape(**kw, rgbaColor=color)
    elif kind == 'cylinder':
        c = p.createCollisionShape(p.GEOM_CYLINDER, radius=x/2, height=z)
        v = p.createVisualShape(p.GEOM_CYLINDER, radius=x/2, length=z, rgbaColor=color)
    elif kind == 'triangle':
        verts = [[-x/2,-y/2,-z/2],[-x/2,y/2,-z/2],[-x/2,0,z/2],
                 [ x/2,-y/2,-z/2],[ x/2,y/2,-z/2],[ x/2,0,z/2]]
        faces = [0,2,1,3,4,5,0,1,4,0,4,3,1,2,5,1,5,4,2,0,3,2,3,5]
        c = p.createCollisionShape(p.GEOM_MESH, vertices=verts, indices=faces)
        v = p.createVisualShape(p.GEOM_MESH, vertices=verts, indices=faces, rgbaColor=color)
    else:
        raise ValueError(kind)
    return p.createMultiBody(baseMass=.09, baseCollisionShapeIndex=c, baseVisualShapeIndex=v)


def camera():
    view = p.computeViewMatrix([.5,0,1.0],[.5,0,0],[0,1,0])
    proj = p.computeProjectionMatrix(-.5,.5,-.5,.5,.01,2.0)
    _,_,rgba,depth,_ = p.getCameraImage(RES,RES,view,proj,renderer=p.ER_TINY_RENDERER)
    rgb = np.asarray(rgba,dtype=np.uint8).reshape(RES,RES,4)[:,:,:3]
    buf = np.asarray(depth,dtype=np.float32).reshape(RES,RES)
    return rgb, (2*.01/(2-(2-.01)*buf)).astype(np.float32)


def make_objects(rng, layers):
    """Generate nested, bridged 2D pyramid rows with usable flat contact areas.

    Each row has a common height (otherwise a single upper piece cannot rest
    on both supports). A cylinder is only used where its circular TOP provides
    a useful contact patch. Supporting cubes are avoided when their width
    would be constrained by their row height.
    """
    count = layers * (layers + 1) // 2
    pitch = float(rng.uniform(.077, .082))
    # Keep each row's stacked components low and squat. Different rows
    # intentionally have different heights to avoid identical-looking pieces.
    heights = [float(rng.uniform(.031, .041)) for _ in range(layers)]
    heights[-1] = float(rng.uniform(.033, .046))
    z_bases = np.cumsum([0.0] + heights[:-1]).tolist()

    all_slots = [(layer, slot) for layer in range(layers)
                 for slot in range(layers-layer)]
    apex = (layers-1, 0)

    # Structural diversity rather than a rigid 50% cylinder quota.
    # Every pyramid has at least one cylinder in the outer base. Larger
    # pyramids MAY have a second cylindrical middle-row support, and the
    # apex varies among cylinder, triangular prism and cube.
    assigned = {}
    outer_base = [(0, 0), (0, layers-1)]
    rng.shuffle(outer_base)
    assigned[outer_base[0]] = 'cylinder'
    if layers >= 4 and rng.random() < .60:
        assigned[(1, int(rng.choice([0, layers-2])))] = 'cylinder'
    if layers == 2:
        assigned[apex] = str(rng.choice(['cube','cylinder','triangle']))
    else:
        assigned[apex] = str(rng.choice(['cylinder','triangle','cube']))

    # Put long rectangular prisms throughout the load-bearing rows.
    # Occasionally introduce an additional cylinder in the outer base.
    if layers >= 3 and rng.random() < .45:
        assigned[outer_base[1]] = 'cylinder'

    objects = []
    for layer in range(layers):
        n = layers-layer
        for slot in range(n):
            shape = assigned.get((layer, slot), 'rectangle')
            h = heights[layer]
            if shape == 'cylinder':
                # Wide flat disks make two-support bridging plausible.
                diameter = float(rng.uniform(.076, .080))
                dims = [diameter, diameter, h]
            elif shape == 'rectangle':
                # Long x-direction slabs provide extended contact patches.
                length = float(rng.uniform(.095, .125))
                width = float(rng.uniform(.071, .076))
                dims = [length, width, h]
            elif shape == 'cube':
                # A real cube at the top. Its height/footprint can differ
                # from the load-bearing rows without losing row alignment.
                side = float(rng.uniform(.038, .048))
                h = side
                dims = [side,side,side]
            else:  # triangle: flat bottom, sloping top, apex only
                dims = [float(rng.uniform(.050, .072)),
                        float(rng.uniform(.052, .068)),h]
            y = (slot - (n-1)/2.0)*pitch
            objects.append({'shape':shape,'dims':[float(v) for v in dims],
                            'layer':layer,'slot':slot,
                            'goal_xyz':[CENTER[0],float(y),
                                        float(z_bases[layer]+h/2)]})

    # Verify large *intersections*, not just barely touching bounding boxes.
    # Avoid cylindrical supports if the upper object only clips their rim.
    for layer in range(1,layers):
        lower=sorted((o for o in objects if o['layer']==layer-1),
                     key=lambda x:x['slot'])
        upper=sorted((o for o in objects if o['layer']==layer),
                     key=lambda x:x['slot'])
        for top in upper:
            for support in (lower[top['slot']],lower[top['slot']+1]):
                offset=abs(top['goal_xyz'][1]-support['goal_xyz'][1])
                overlap=(top['dims'][1]+support['dims'][1])/2-offset
                if overlap < .016:
                    return None,None
                # A cylinder's contact top is circular; require a
                # meaningful interior patch, not a rim-only overlap.
                if support['shape']=='cylinder':
                    radius=support['dims'][1]/2
                    inner_offset=max(0.,offset-top['dims'][1]/2)
                    if inner_offset > radius-.013:
                        return None,None

    identities=rng.permutation(count)
    for i,obj in enumerate(objects):
        obj['id']=int(identities[i])
    return objects,[o['id'] for o in objects]

def sample_starts(rng, objects):
    """Randomized staging strips leave room for all 10 initial objects.

    Two side regions are clear of the central pyramid. Jittered cells give
    variation without solving an unnecessarily difficult 10-object packing
    problem during every candidate attempt.
    """
    # 6 cells per side; 4-layer pyramids need at most 10 starts.
    cells = [(x, y) for x in (.325, .675)
             for y in np.linspace(-.37, .37, 6)]
    rng.shuffle(cells)
    placed = []
    for obj, (base_x, base_y) in zip(objects, cells):
        half_x = obj['dims'][0] / 2
        half_y = obj['dims'][1] / 2
        # Axis-aligned, conservative non-overlap checks.
        for _ in range(100):
            x = float(base_x + rng.uniform(-.008,.008))
            y = float(base_y + rng.uniform(-.016,.016))
            if not (.25+.005+half_x <= x <= .75-.005-half_x
                    and -.5+.005+half_y <= y <= .5-.005-half_y):
                continue
            if all((abs(x-px) > half_x+phx+.012 or
                    abs(y-py) > half_y+phy+.012)
                   for px,py,phx,phy in placed):
                obj['start_xy'] = [x,y]
                placed.append((x,y,half_x,half_y))
                break
        else:
            return False
    return len(placed)==len(objects)

def settled(objects, bodies, upto, xy_tol=.009, z_tol=.009, tilt_tol=.17):
    """Check all already placed objects, not only the last placement."""
    for obj in objects[:upto]:
        pos,rot = p.getBasePositionAndOrientation(bodies[obj['id']])
        roll,pitch,_=p.getEulerFromQuaternion(rot)
        target=obj['goal_xyz']
        if (math.hypot(pos[0]-target[0],pos[1]-target[1])>xy_tol or
            abs(pos[2]-target[2])>z_tol or
            abs(roll)>tilt_tol or abs(pitch)>tilt_tol):
            return False
    return True


def generate_episode(rng, layers, out, settle_steps, validate, gui=False):
    objects, order = make_objects(rng,layers)
    if objects is None or not sample_starts(rng,objects):
        return False
    if not all(valid_pose([*o['start_xy'],o['dims'][2]/2+.001])
               and valid_pose(o['goal_xyz']) for o in objects):
        return False
    cid=p.connect(p.GUI if gui else p.DIRECT)
    try:
        p.setGravity(0,0,-9.81)
        p.setTimeStep(1/240)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        floor=p.createMultiBody(0,p.createCollisionShape(p.GEOM_PLANE))
        p.changeDynamics(floor,-1,lateralFriction=1.5)
        bodies={}
        for obj in objects:
            b=create_body(obj,COLORS[obj['id']%len(COLORS)])
            p.changeDynamics(b,-1,lateralFriction=1.5,linearDamping=.05,angularDamping=.12)
            x,y=obj['start_xy']; z=obj['dims'][2]/2+.001
            p.resetBasePositionAndOrientation(b,[x,y,z],[0,0,0,1])
            bodies[obj['id']]=b
        for _ in range(settle_steps): p.stepSimulation()

        # Write images only into a temporary directory; no failed partial episodes.
        with tempfile.TemporaryDirectory(prefix='pyramid_tmp_') as temp:
            temp=Path(temp)
            actions=[]
            for step,obj in enumerate(objects):
                rgb,depth=camera()
                Image.fromarray(rgb).save(temp/f'rgb_{step:02d}.png')
                np.save(temp/f'depth_{step:02d}.npy',depth)
                b=bodies[obj['id']]
                pick,_=p.getBasePositionAndOrientation(b)
                place=obj['goal_xyz']
                actions.append({'object_id':obj['id'],'shape':obj['shape'],
                    'layer':obj['layer'],'slot':obj['slot'],
                    'pick_xyz':list(map(float,pick)),
                    'place_xyz':place,'pick_yaw':0.0,'place_yaw':0.0,
                    'gripper_opening_m':float(max(obj['dims'][:2])+.008),
                    'rgb':f'rgb_{step:02d}.png','depth':f'depth_{step:02d}.npy',
                    'label_type':'teleported_geometry_oracle'})
                # Reject any drifted pick immediately; never write a Ravens-
                # incompatible episode that would be skipped by the converter.
                if not valid_pose(pick) or not valid_pose(place):
                    return False
                p.resetBasePositionAndOrientation(b,place,[0,0,0,1])
                p.resetBaseVelocity(b,[0,0,0],[0,0,0])
                for _ in range(settle_steps): p.stepSimulation()
                if validate and not settled(objects,bodies,step+1):
                    return False
                # Also ensure no object was ejected outside Ravens bounds,
                # including not-yet-picked objects displaced by collisions.
                if any(not valid_pose(p.getBasePositionAndOrientation(body)[0])
                       for body in bodies.values()):
                    return False

            if not within_ravens_bounds(objects, actions):
                return False
            metadata={'stack_type':'pyramid','num_layers':layers,
                'layer_counts':list(range(layers,0,-1)),
                'workspace_bounds':BOUNDS,'stack_xy':CENTER,
                'max_gripper_opening_m':GRIPPER_LIMIT,
                'objects':objects,'stack_object_ids':order,'actions':actions,
                'physics_validated':bool(validate),'synthetic_oracle':True,
                'note':'PyBullet checked teleport placements, NOT executed robot grasps'}
            (temp/'episode.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
            out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists(): shutil.rmtree(out)
            shutil.copytree(temp,out)
        return True
    finally:
        p.disconnect(cid)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--episodes',type=int,default=100)
    ap.add_argument('--out',type=Path,default=Path('hetero-pyramid-episodes'))
    ap.add_argument('--layers',type=int,choices=[2,3,4],default=None)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--settle_steps',type=int,default=120)
    ap.add_argument('--max_attempts',type=int,default=80,
                    help='Attempts per episode, including physics rejections')
    ap.add_argument('--no_validate',action='store_true',
                    help='Disable post-placement physics checking (not recommended)')
    ap.add_argument('--gui',action='store_true')
    ap.add_argument('--strict',action='store_true',
                    help='Exit with error if fewer than requested episodes were produced')
    args=ap.parse_args()
    rng=np.random.default_rng(args.seed)
    success=0
    for i in range(args.episodes):
        layers=args.layers if args.layers is not None else int(rng.integers(2,5))
        out=args.out/f'episode_{i:06d}'
        for attempt in range(1,args.max_attempts+1):
            attempt_rng=np.random.default_rng(int(rng.integers(0,2**31-1)))
            if generate_episode(attempt_rng,layers,out,args.settle_steps,
                                not args.no_validate,args.gui):
                success+=1
                print(f'Generated {out} ({layers} layers, attempt {attempt})')
                break
        else:
            print(f'FAILED episode_{i:06d}: no valid {layers}-layer scene in {args.max_attempts} attempts')
    print(f'Done: {success}/{args.episodes} episodes in {args.out}', flush=True)
    if args.strict and success != args.episodes:
        raise SystemExit(f'Only {success}/{args.episodes} generated; increase --max_attempts')

if __name__ == '__main__':
    main()
