"""Replay validated stacking episodes in the PyBullet GUI.

Run from your project root:
    python generated_images_viewer.py
    python generated_images_viewer.py --data mixed-stacking-validated --limit 100

Reads episode_XXXXXX/episode.json, NOT the converted Ravens .pkl dataset.
Moves objects directly to the recorded placements (no UR5 or real grasps).
"""
import argparse
import json
import time
from pathlib import Path

import pybullet as p

COLORS = [
    (0.94, 0.34, 0.27, 1),
    (0.26, 0.58, 0.91, 1),
    (0.27, 0.75, 0.51, 1),
    (0.94, 0.70, 0.26, 1),
    (0.67, 0.47, 0.86, 1),
]


def make_body(obj):
    x, y, z = map(float, obj["dims"])
    shape = obj["shape"]
    color = COLORS[int(obj["id"]) % len(COLORS)]
    if shape in ("cube", "rectangle"):
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=(x / 2, y / 2, z / 2))
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=(x / 2, y / 2, z / 2), rgbaColor=color)
    elif shape == "cylinder":
        collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=x / 2, height=z)
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=x / 2, length=z, rgbaColor=color)
    elif shape == "triangle":
        vertices = [
            [-x/2, -y/2, -z/2], [-x/2, y/2, -z/2], [-x/2, 0, z/2],
            [ x/2, -y/2, -z/2], [ x/2, y/2, -z/2], [ x/2, 0, z/2],
        ]
        indices = [0, 2, 1, 3, 4, 5, 0, 1, 4, 0, 4, 3,
                   1, 2, 5, 1, 5, 4, 2, 0, 3, 2, 3, 5]
        collision = p.createCollisionShape(p.GEOM_MESH, vertices=vertices, indices=indices)
        visual = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=indices, rgbaColor=color)
    else:
        raise ValueError(f"Unsupported object shape: {shape}")
    return p.createMultiBody(baseMass=0.12, baseCollisionShapeIndex=collision,
                             baseVisualShapeIndex=visual)


def step_physics(n=120):
    for _ in range(n):
        p.stepSimulation()


def replay_episode(meta_path, index, total, args):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1 / 240)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_PLANE))
    stack_xy = meta.get("stack_xy", (0.5, 0.18))
    p.resetDebugVisualizerCamera(cameraDistance=0.65, cameraYaw=45, cameraPitch=-28,
                                 cameraTargetPosition=(stack_xy[0], stack_xy[1], 0.08))
    bodies = {}
    for obj in meta["objects"]:
        oid = int(obj["id"])
        body = make_body(obj)
        pos = (float(obj["start_xy"][0]), float(obj["start_xy"][1]),
               float(obj["dims"][2]) / 2 + 0.001)
        p.resetBasePositionAndOrientation(body, pos, (0, 0, 0, 1))
        bodies[oid] = body
    step_physics()
    print(f"Episode {index}/{total}: {meta_path.parent.name} ({len(meta['objects'])} objects)", flush=True)
    time.sleep(args.start_delay)
    for i, action in enumerate(meta["actions"], 1):
        oid = int(action["object_id"])
        yaw = float(action.get("place_yaw", 0))
        p.resetBasePositionAndOrientation(bodies[oid], action["place_xyz"],
                                          p.getQuaternionFromEuler((0, 0, yaw)))
        p.resetBaseVelocity(bodies[oid], (0, 0, 0), (0, 0, 0))
        step_physics()
        print(f"  Placed {i}/{len(meta['actions'])}: {action['shape']} (ID {oid})", flush=True)
        time.sleep(args.step_delay)
    time.sleep(args.end_delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("mixed-stacking-validated"), help="Folder containing episode_XXXXXX subfolders")
    parser.add_argument("--limit", type=int, default=0, help="Maximum episodes; 0 = all")
    parser.add_argument("--start-delay", type=float, default=1.0)
    parser.add_argument("--step-delay", type=float, default=0.8)
    parser.add_argument("--end-delay", type=float, default=2.0)
    args = parser.parse_args()
    episodes = sorted(args.data.glob("episode_*/episode.json"))
    if args.limit > 0:
        episodes = episodes[:args.limit]
    if not episodes:
        parser.error(f"No episode_*/episode.json files found in {args.data.resolve()}")
    client = p.connect(p.GUI)
    if client < 0:
        raise RuntimeError("Could not open PyBullet GUI")
    try:
        for i, path in enumerate(episodes, 1):
            if not p.isConnected():
                break
            replay_episode(path, i, len(episodes), args)
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
