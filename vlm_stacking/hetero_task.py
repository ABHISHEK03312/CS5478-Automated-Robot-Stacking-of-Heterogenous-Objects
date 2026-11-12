"""Task factory that recreates a hetero-pyramid-data episode in simulation.

The factory reads the saved info pickle for a specific episode, then returns a
StackBlockPyramidParallelJaw subclass whose reset() spawns the correct objects
(cylinders, rectangles, cubes) at the recorded starting positions and registers
the recorded goal positions as the Ravens task goals.
"""

import glob
import os
import pickle
import tempfile

import numpy as np
import pybullet as p

from ravens.tasks.stack_block_pyramid import StackBlockPyramidParallelJaw
from ravens.tasks.task import Task
from ravens.utils import utils


# Rotational symmetry for each shape used in is_match().
# 0 = full rotational symmetry (cylinder); np.pi = 2-fold; np.pi/2 = 4-fold.
_SYMMETRY = {
    "cylinder":  0,
    "rectangle": np.pi,
    "cube":      np.pi / 2,
}

# Cycle through these for per-object colours (distinctive across shapes).
_COLOR_NAMES = ["blue", "red", "green", "orange", "yellow",
                "purple", "cyan", "pink", "brown", "gray"]


def _cylinder_urdf(radius, height):
    """Write a minimal URDF for an upright cylinder; return its temp path."""
    mass = 0.06
    i_xy = mass * (3 * radius ** 2 + height ** 2) / 12.0
    i_z = mass * radius ** 2 / 2.0
    xml = (
        '<?xml version="1.0"?>\n'
        '<robot name="hetero_cylinder">\n'
        '  <link name="body">\n'
        '    <inertial>\n'
        '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        '      <mass value="{mass}"/>\n'
        '      <inertia ixx="{ixy}" ixy="0" ixz="0"'
        ' iyy="{ixy}" iyz="0" izz="{iz}"/>\n'
        '    </inertial>\n'
        '    <visual>\n'
        '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        '      <geometry>'
        '<cylinder radius="{r}" length="{h}"/>'
        '</geometry>\n'
        '    </visual>\n'
        '    <collision>\n'
        '      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        '      <geometry>'
        '<cylinder radius="{r}" length="{h}"/>'
        '</geometry>\n'
        '    </collision>\n'
        '  </link>\n'
        '</robot>\n'
    ).format(mass=mass, ixy=i_xy, iz=i_z, r=radius, h=height)
    fd, path = tempfile.mkstemp(suffix=".urdf", prefix="hetero_cyl_")
    with os.fdopen(fd, "w") as fh:
        fh.write(xml)
    return path


def load_hetero_episode_info(dataset_path, episode_index):
    """Return (info_list, seed_int) for the episode at episode_index."""
    info_dir = os.path.join(dataset_path, "info")
    if not os.path.isdir(info_dir):
        raise FileNotFoundError("Dataset info folder not found: %s" % info_dir)
    files = sorted(glob.glob(os.path.join(info_dir, "*.pkl")))
    if not files:
        raise ValueError("No .pkl files in %s" % info_dir)
    if episode_index >= len(files):
        raise IndexError(
            "episode_index %d out of range (dataset has %d episodes)" %
            (episode_index, len(files)))
    filepath = files[episode_index]
    fname = os.path.basename(filepath)
    seed = int(fname[fname.find("-") + 1:-4])
    with open(filepath, "rb") as fh:
        info = pickle.load(fh)
    return info, seed


def make_hetero_episode_task(episode_info):
    """Return a Task class that spawns objects from the given episode info.

    The returned class inherits from StackBlockPyramidParallelJaw so it reuses
    the parallel-jaw gripper and pick primitive without any extra wiring.
    """
    step0 = next(
        s for s in episode_info
        if isinstance(s, dict) and "objects" in s)
    objects_spec = list(step0["objects"])
    n_objects = len(objects_spec)

    class HeteroPyramidEpisodeTask(StackBlockPyramidParallelJaw):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_steps = n_objects * 3

        def reset(self, env):
            # Bypass StackBlockPyramid.reset() (which spawns 6 standard cubes).
            Task.reset(self, env)

            temp_urdfs = []
            # Map from dataset object id -> (pybullet_body_id, symmetry)
            spawned = {}

            for idx, obj in enumerate(objects_spec):
                shape = obj["shape"]
                dims = obj["dims"]        # [x, y, z] full extents; for cylinder x==y==diameter
                start_xy = obj["start_xy"]
                z_spawn = float(dims[2]) / 2.0
                pos = (float(start_xy[0]), float(start_xy[1]), z_spawn)
                rot = (0.0, 0.0, 0.0, 1.0)

                if shape == "cylinder":
                    radius = float(dims[0]) / 2.0
                    height = float(dims[2])
                    urdf_path = _cylinder_urdf(radius, height)
                    temp_urdfs.append(urdf_path)
                    body = p.loadURDF(urdf_path, pos, rot)
                else:
                    # rectangle or cube — use the standard box template
                    urdf_path = self.fill_template(
                        "box/box-template.urdf",
                        {"DIM": [float(d) for d in dims]})
                    temp_urdfs.append(urdf_path)
                    body = p.loadURDF(urdf_path, pos, rot)

                env.obj_ids["rigid"].append(body)
                p.changeDynamics(body, -1, lateralFriction=1.0)
                color_name = _COLOR_NAMES[idx % len(_COLOR_NAMES)]
                p.changeVisualShape(
                    body, -1,
                    rgbaColor=list(utils.COLORS[color_name]) + [1.0])
                spawned[obj["id"]] = (body, _SYMMETRY.get(shape, np.pi))

            for path in temp_urdfs:
                try:
                    os.remove(path)
                except OSError:
                    pass

            # Build Ravens goals: one entry per layer, targets from goal_xyz.
            from collections import defaultdict
            by_layer = defaultdict(list)
            for obj in objects_spec:
                by_layer[obj["layer"]].append(obj)

            for layer_idx in sorted(by_layer.keys()):
                layer_objs = by_layer[layer_idx]
                pybullet_objs = [
                    (spawned[o["id"]][0], (spawned[o["id"]][1], None))
                    for o in layer_objs]
                target_poses = [
                    (tuple(float(v) for v in o["goal_xyz"]),
                     (0.0, 0.0, 0.0, 1.0))
                    for o in layer_objs]
                n = len(layer_objs)
                reward_fraction = n / n_objects
                self.goals.append((
                    pybullet_objs,
                    np.ones((n, n), dtype=np.int32),
                    target_poses,
                    False,   # replace
                    True,    # rotations
                    "pose",  # metric
                    None,    # params
                    reward_fraction,
                ))

    return HeteroPyramidEpisodeTask
