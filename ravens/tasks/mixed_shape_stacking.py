"""Mixed-shape 3-2-1 pyramid for a Ravens parallel-jaw task.

Five rectangular blocks of different sizes plus one upright cylinder.
Uses the existing parallel-jaw task's gripper and task/oracle conventions.
This is a target-based demonstration task, not a free-form stacking task.

Install as ravens/tasks/mixed_shape_stacking.py and register using the
MixedShapeStackingMixin pattern described in the instructions.
"""

import os
import tempfile

import numpy as np
import pybullet as p

from ravens.tasks.task import Task
from ravens.utils import utils


class MixedShapeStackingMixin:
    """Override scene generation while reusing the existing gripper/task logic."""

    # 5 different rectangular prisms + 1 cylinder.
    # Keep all heights at 4 cm so the standard 3-2-1 pyramid target z-levels work.
    # Order matters: first 3 = base layer, next 2 = middle layer, last = top layer.
    OBJECTS = (
        ('box', (0.055, 0.035, 0.040), 'purple'),
        ('box', (0.050, 0.030, 0.040), 'blue'),
        ('box', (0.046, 0.038, 0.040), 'green'),
        ('box', (0.042, 0.028, 0.040), 'yellow'),
        ('box', (0.036, 0.024, 0.040), 'orange'),
        ('cylinder', (0.018, 0.040), 'red'),  # (radius, height)
    )

    @staticmethod
    def _cylinder_urdf(radius, height):
        """Create a temporary URDF for a centred upright cylinder."""
        mass = 0.06
        i_xy = mass * (3 * radius**2 + height**2) / 12.0
        i_z = mass * radius**2 / 2.0
        xml = f'''<?xml version="1.0"?>
<robot name="mixed_stack_cylinder">
  <link name="body">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{i_xy}" ixy="0" ixz="0" iyy="{i_xy}" iyz="0" izz="{i_z}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="{radius}" length="{height}"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="{radius}" length="{height}"/></geometry>
    </collision>
  </link>
</robot>
'''
        fd, filename = tempfile.mkstemp(suffix='.urdf', prefix='ravens_stack_')
        with os.fdopen(fd, 'w') as file:
            file.write(xml)
        return filename

    def reset(self, env):
        # Do NOT call parent reset() because that would create the original cubes.
        Task.reset(self, env)

        stand_pose = self.get_random_pose(env, (0.05, 0.15, 0.005))
        if stand_pose[0] is None:
            raise RuntimeError('Could not place pyramid stand')
        env.add_object('stacking/stand.urdf', stand_pose, 'fixed')

        objects = []
        for index, (shape, dims, colour) in enumerate(self.OBJECTS):
            if shape == 'box':
                footprint = dims
                urdf_path = self.fill_template('box/box-template.urdf', {'DIM': dims})
                # Non-square rectangles only match after 180-degree yaw.
                symmetry = np.pi
            elif shape == 'cylinder':
                radius, height = dims
                footprint = (2 * radius, 2 * radius, height)
                urdf_path = self._cylinder_urdf(radius, height)
                symmetry = 0  # yaw is irrelevant for an upright cylinder
            else:
                raise ValueError(f'Unknown shape: {shape}')

            try:
                pose = self.get_random_pose(env, footprint)
                if pose[0] is None:
                    raise RuntimeError(f'No free position for object {index}')
                body = env.add_object(urdf_path, pose)
            finally:
                if os.path.exists(urdf_path):
                    os.remove(urdf_path)

            p.changeVisualShape(body, -1, rgbaColor=list(utils.COLORS[colour]) + [1.0])
            p.changeDynamics(body, -1, lateralFriction=1.0)
            objects.append((body, (symmetry, None)))

        # Standard 3-2-1 pyramid target offsets relative to the stand.
        offsets = (
            (0, -0.050, 0.030),
            (0,  0.000, 0.030),
            (0,  0.050, 0.030),
            (0, -0.025, 0.080),
            (0,  0.025, 0.080),
            (0,  0.000, 0.130),
        )
        targets = [(utils.apply(stand_pose, offset), stand_pose[1])
                   for offset in offsets]

        # Ravens goal tuple format:
        # (objects, match_matrix, target_poses, replace, rotations, metric, params, reward_fraction)
        for start, end, reward in ((0, 3, 0.5),
                                   (3, 5, 1.0 / 3.0),
                                   (5, 6, 1.0 / 6.0)):
            count = end - start
            self.goals.append((
                objects[start:end],
                np.ones((count, count), dtype=np.int32),
                targets[start:end],
                False,
                True,
                'pose',
                None,
                reward,
            ))
