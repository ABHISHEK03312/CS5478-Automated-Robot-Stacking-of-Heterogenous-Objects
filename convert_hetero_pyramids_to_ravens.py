"""Convert heterogeneous pyramid geometry-oracle episodes to Ravens Dataset.

Run in Ravens repository root:
python convert_hetero_pyramids_to_ravens.py --data hetero-pyramid-episodes --out data/hetero-pyramid-train --assets_root ./ravens/environments/assets

WARNING: Uses teleportation and custom incremental pose reward, NOT robot
Environment.step(), task.reward(), or validated parallel-jaw grasps.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pybullet as p
from ravens.environments.environment import Environment
from ravens.dataset import Dataset

BOUNDS=((.25,.75),(-.50,.50),(0.,.28))
COLORS=((.94,.34,.27,1),(.26,.58,.91,1),(.27,.75,.51,1),(.94,.70,.26,1),(.67,.47,.86,1))


def make_body(obj,rgba):
    x,y,z=map(float,obj['dims']);shape=obj['shape']
    if shape in ('cube','rectangle'):
        kw=dict(shapeType=p.GEOM_BOX,halfExtents=[x/2,y/2,z/2])
        c=p.createCollisionShape(**kw);v=p.createVisualShape(**kw,rgbaColor=rgba)
    elif shape=='cylinder':
        c=p.createCollisionShape(p.GEOM_CYLINDER,radius=x/2,height=z)
        v=p.createVisualShape(p.GEOM_CYLINDER,radius=x/2,length=z,rgbaColor=rgba)
    elif shape=='triangle':
        vertices=[[-x/2,-y/2,-z/2],[-x/2,y/2,-z/2],[-x/2,0,z/2],
                  [x/2,-y/2,-z/2],[x/2,y/2,-z/2],[x/2,0,z/2]]
        indices=[0,2,1,3,4,5,0,1,4,0,4,3,1,2,5,1,5,4,2,0,3,2,3,5]
        c=p.createCollisionShape(p.GEOM_MESH,vertices=vertices,indices=indices)
        v=p.createVisualShape(p.GEOM_MESH,vertices=vertices,indices=indices,rgbaColor=rgba)
    else:raise ValueError(shape)
    return p.createMultiBody(baseMass=.09,baseCollisionShapeIndex=c,baseVisualShapeIndex=v)


def valid_pose(xyz):
    return len(xyz)==3 and all(lo<=float(v)<=hi for v,(lo,hi) in zip(xyz,BOUNDS))


def preflight(folder):
    meta = json.loads((folder / 'episode.json').read_text(encoding='utf-8'))
    objects = meta.get('objects', [])
    actions = meta.get('actions', [])
    layers = int(meta.get('num_layers', -1))
    issues = []
    if meta.get('stack_type') != 'pyramid':
        issues.append('stack_type is not pyramid')
    if layers not in (2, 3, 4) or len(objects) != layers * (layers + 1) // 2:
        issues.append('incorrect object/layer count')
    if len(actions) != len(objects):
        issues.append('incorrect number of actions')
    for obj in objects:
        start = [*obj['start_xy'], float(obj['dims'][2]) / 2 + .001]
        if not valid_pose(start):
            issues.append(f"object {obj['id']} start={start} out of bounds")
        if not valid_pose(obj['goal_xyz']):
            issues.append(f"object {obj['id']} goal={obj['goal_xyz']} out of bounds")
    for i, action in enumerate(actions):
        for name in ('pick_xyz', 'place_xyz'):
            if not valid_pose(action[name]):
                issues.append(f"action {i} {name}={action[name]} out of bounds (x=.25..75, y=-.5..5, z=0..28)")
        if not 0 < float(action.get('gripper_opening_m', 0)) <= .15:
            issues.append(f"action {i} invalid gripper opening")
    return issues


def replay(env,folder,settle_steps,pos_tolerance):
    meta=json.loads((folder/'episode.json').read_text(encoding='utf-8'))
    if meta.get('stack_type')!='pyramid':raise ValueError('Not a pyramid episode')
    layers=int(meta['num_layers']);objects=meta['objects'];actions=meta['actions']
    if len(objects)!=layers*(layers+1)//2:raise ValueError('Object count mismatch')
    if len(actions)!=len(objects):raise ValueError('Expected one action per object')
    p.resetSimulation();p.setGravity(0,0,-9.81);p.setTimeStep(1/240)
    p.setPhysicsEngineParameter(numSolverIterations=150)
    ground=p.createMultiBody(0,p.createCollisionShape(p.GEOM_PLANE))
    p.changeDynamics(ground,-1,lateralFriction=1.5)
    env.obj_ids={'fixed':[],'rigid':[],'deformable':[]};bodies={}
    for o in objects:
        oid=int(o['id']);xy=o['start_xy'];z=float(o['dims'][2])/2+.001
        pose=[float(xy[0]),float(xy[1]),z]
        if not valid_pose(pose):raise ValueError(f'Object {oid} outside workspace')
        b=make_body(o,COLORS[oid%len(COLORS)])
        p.changeDynamics(b,-1,lateralFriction=1.5,linearDamping=.05,angularDamping=.12)
        p.resetBasePositionAndOrientation(b,pose,[0,0,0,1]);bodies[oid]=b
        env.obj_ids['rigid'].append(b)
    for _ in range(settle_steps):p.stepSimulation()
    episode=[];awarded=set();n=len(actions)
    for step,source in enumerate(actions):
        oid=int(source['object_id']);pick=source['pick_xyz'];place=source['place_xyz']
        if oid not in bodies or oid in awarded:raise ValueError(f'Duplicate or unknown id {oid}')
        if not valid_pose(pick) or not valid_pose(place):raise ValueError(f'Out of bounds action {step}: pick={pick}, place={place}')
        opening=float(source.get('gripper_opening_m',0))
        if not 0<opening<=.15:raise ValueError(f'Invalid gripper opening {opening}')
        # The environment's multi-view observations, not the generator's PNGs.
        obs=env._get_obs()
        q0=p.getQuaternionFromEuler([0,0,float(source.get('pick_yaw',0))])
        q1=p.getQuaternionFromEuler([0,0,float(source.get('place_yaw',0))])
        action={'pose0':(tuple(map(float,pick)),tuple(q0)),
                'pose1':(tuple(map(float,place)),tuple(q1))}
        info={'source_episode':folder.name,'step':step,'object_id':oid,
              'shape':source['shape'],'stack_type':'pyramid','num_layers':layers,
              'layer':int(source['layer']),'slot':int(source['slot']),
              'gripper_opening_m':opening,'label_type':'teleported_geometry_oracle',
              'objects':objects}
        b=bodies[oid]
        p.resetBasePositionAndOrientation(b,place,q1)
        p.resetBaseVelocity(b,[0,0,0],[0,0,0])
        for _ in range(settle_steps):p.stepSimulation()
        actual,quat=p.getBasePositionAndOrientation(b)
        roll,pitch,_=p.getEulerFromQuaternion(quat)
        ok=(np.linalg.norm(np.asarray(actual)-np.asarray(place))<=pos_tolerance
            and abs(roll)<.20 and abs(pitch)<.20)
        if ok:
            for previous in actions[:step]:
                pid=int(previous['object_id'])
                prev_pos,prev_q=p.getBasePositionAndOrientation(bodies[pid])
                r,t,_=p.getEulerFromQuaternion(prev_q)
                if (np.linalg.norm(np.asarray(prev_pos)-np.asarray(previous['place_xyz']))>pos_tolerance
                    or abs(r)>=.20 or abs(t)>=.20):ok=False;break
        if ok:awarded.add(oid)
        info['goal_met']=bool(ok)
        info['reward_source']='custom_pose_goal_not_ravens_task_reward'
        episode.append((obs,action,1.0/n if ok else 0.0,info))
    success=len(awarded)==n
    episode.append((env._get_obs(),None,0.0,
                    {'source_episode':folder.name,'terminal':True,
                     'stack_type':'pyramid','num_layers':layers,
                     'reward_source':'custom_pose_goal_not_ravens_task_reward',
                     'success':success}))
    return episode,success


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path('hetero-pyramid-episodes'))
    ap.add_argument('--out',type=Path,default=Path('data/hetero-pyramid-train'))
    ap.add_argument('--assets_root',default='./ravens/environments/assets/')
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--settle_steps',type=int,default=120)
    ap.add_argument('--pos_tolerance',type=float,default=.012)
    ap.add_argument('--gui',action='store_true')
    ap.add_argument('--preflight',action='store_true',
                    help='Check every JSON pose without camera rendering')
    ap.add_argument('--allow_partial',action='store_true',
                    help='Save valid episodes even if some fail conversion')
    args=ap.parse_args()
    folders=sorted(x.parent for x in args.data.glob('episode_*/episode.json'))
    if args.limit:folders=folders[:args.limit]
    if not folders:raise SystemExit(f'No episode.json in {args.data}')
    valid_folders, rejected = [], []
    for folder in folders:
        issues = preflight(folder)
        if issues:
            rejected.append(folder.name)
            print(f'INVALID {folder.name}: ' + '; '.join(issues[:4]), flush=True)
        else:
            valid_folders.append(folder)
    print(f'Preflight: {len(valid_folders)}/{len(folders)} in bounds', flush=True)
    if args.preflight:
        if rejected:
            raise SystemExit(f'Preflight failed: {len(rejected)} invalid episodes')
        return
    if rejected and not args.allow_partial:
        raise SystemExit(
            'Refusing partial conversion: fix/regenerate the above episodes, '
            'or use --allow_partial to skip them.')
    if not valid_folders:
        raise SystemExit('No valid episodes')
    args.out.mkdir(parents=True,exist_ok=True)
    dataset = Dataset(str(args.out))
    env = Environment(assets_root=args.assets_root,disp=args.gui)
    saved, failures = 0, list(rejected)
    try:
        for folder in valid_folders:
            try:
                episode, success = replay(env, folder, args.settle_steps,
                                          args.pos_tolerance)
                if not success:
                    print(f'FAILED {folder.name}: replay stability failed', flush=True)
                    failures.append(folder.name)
                    if not args.allow_partial:
                        break
                    continue
                dataset.add(int(folder.name.split('_')[-1]), episode)
                saved += 1
                print(f'Saved {folder.name}: {len(episode)-1} actions', flush=True)
            except Exception as exc:
                failures.append(folder.name)
                print(f'FAILED {folder.name}: {type(exc).__name__}: {exc}', flush=True)
                if not args.allow_partial:
                    break
    except KeyboardInterrupt:
        print(f'Interrupted; {saved} episodes saved so far.', flush=True)
        raise
    finally:
        env.close()
    print(f'Done: saved={saved}, failed={len(failures)}, '
          f'output={args.out}', flush=True)
    if failures and not args.allow_partial:
        raise SystemExit('Conversion was incomplete; review FAILED episode and rerun.')

if __name__=='__main__':main()
