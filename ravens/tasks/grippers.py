# coding=utf-8
# Copyright 2024 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classes to handle gripper dynamics."""

import os

import numpy as np
from ravens.utils import pybullet_utils

import pybullet as p

SPATULA_BASE_URDF = 'ur5/spatula/spatula-base.urdf'
SUCTION_BASE_URDF = 'ur5/suction/suction-base.urdf'
SUCTION_HEAD_URDF = 'ur5/suction/suction-head.urdf'
PARALLEL_JAW_URDF = 'ur5/gripper/simple_parallel_jaw.urdf'


class Gripper:
  """Base gripper class."""

  def __init__(self, assets_root):
    self.assets_root = assets_root
    self.activated = False

  def step(self):
    """This function can be used to create gripper-specific behaviors."""
    return

  def activate(self, objects):
    del objects
    return

  def release(self):
    return


class Spatula(Gripper):
  """Simulate simple spatula for pushing."""

  def __init__(self, assets_root, robot, ee, obj_ids):  # pylint: disable=unused-argument
    """Creates spatula and 'attaches' it to the robot."""
    super().__init__(assets_root)

    # Load spatula model.
    pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
    base = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, SPATULA_BASE_URDF), pose[0], pose[1])
    p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=base,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, 0.01))


class ParallelJaw(Gripper):
  """Kinematic two-finger gripper with geometry-gated rigid grasps."""

  TOOL_CENTER_Z = 0.12
  OPEN_INNER_WIDTH = 0.15 #CHANGE IS HERE

  def __init__(self, assets_root, robot, ee, obj_ids):
    super().__init__(assets_root)
    self.robot = robot
    self.mount_link = ee
    self.obj_ids = obj_ids
    self.contact_constraint = None

    parent_state = p.getLinkState(robot, ee)
    pose = (parent_state[4], parent_state[5])
    self.body = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, PARALLEL_JAW_URDF), pose[0], pose[1],
        useFixedBase=True)

    joints = [p.getJointInfo(self.body, i)
              for i in range(p.getNumJoints(self.body))]
    joint_ids = {joint[1].decode('utf-8'): joint[0] for joint in joints}
    self.left_finger = joint_ids['left_finger_joint']
    self.right_finger = joint_ids['right_finger_joint']
    self.finger_joints = (self.left_finger, self.right_finger)
    self.finger_positions = {joint: 0.0 for joint in self.finger_joints}

    # Ravens drives the arm kinematically. Collision response against a second
    # kinematic body creates large artificial impulses, so grasp eligibility is
    # checked geometrically and transport uses a temporary rigid constraint.
    for gripper_link in range(-1, p.getNumJoints(self.body)):
      p.setCollisionFilterGroupMask(self.body, gripper_link, 0, 0)

    for finger in self.finger_joints:
      p.setJointMotorControl2(
          self.body, finger, p.VELOCITY_CONTROL, force=0)
    self._command(0.0, steps=1)

  def _command(self, position, steps=120):
    """Move both prismatic fingers to the requested closure distance."""
    target = float(np.clip(position, 0.0, 0.03))
    starts = dict(self.finger_positions)
    for fraction in np.linspace(0, 1, max(1, steps)):
      for joint in self.finger_joints:
        start = starts[joint]
        self.finger_positions[joint] = float(
            start + fraction * (target - start))
      self.step()
      p.stepSimulation()

  def activate(self):
    """Close around the centered object and preserve the resulting grasp."""
    base_pose = p.getBasePositionAndOrientation(self.body)
    inverse = p.invertTransform(base_pose[0], base_pose[1])
    candidates = []
    for object_id in self.obj_ids['rigid']:
      object_pose = p.getBasePositionAndOrientation(object_id)
      local_pose = p.multiplyTransforms(
          inverse[0], inverse[1], object_pose[0], object_pose[1])
      x, y, z = local_pose[0]
      if (abs(x) < 0.035 and abs(y) < 0.045 and
          abs(z - self.TOOL_CENTER_Z) < 0.05):
        distance = x * x + y * y + (z - self.TOOL_CENTER_Z)**2
        candidates.append((distance, object_id, local_pose))

    if not candidates:
      self.activated = False
      return

    _, object_id, local_pose = min(candidates)
    dimensions = p.getVisualShapeData(object_id)[0][3]
    object_width = max(float(dimensions[0]), float(dimensions[1]))
    closure = np.clip(
        (self.OPEN_INNER_WIDTH - object_width) / 2 - 0.001, 0, 0.03)
    self._command(closure, steps=120)

    self.contact_constraint = p.createConstraint(
        parentBodyUniqueId=self.body,
        parentLinkIndex=-1,
        childBodyUniqueId=object_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=local_pose[0],
        childFramePosition=(0, 0, 0),
        parentFrameOrientation=local_pose[1],
        childFrameOrientation=(0, 0, 0, 1))
    self.activated = True

  def release(self):
    """Remove the temporary grasp constraint and open both fingers."""
    if self.contact_constraint is not None:
      p.removeConstraint(self.contact_constraint)
      self.contact_constraint = None
    self.activated = False
    self._command(0.0, steps=180)

  def step(self):
    """Keep the kinematic tool and its fingers aligned with the UR5 wrist."""
    state = p.getLinkState(self.robot, self.mount_link)
    p.resetBasePositionAndOrientation(self.body, state[4], state[5])
    for joint in self.finger_joints:
      p.resetJointState(
          self.body, joint, self.finger_positions[joint], targetVelocity=0)

  def detect_contact(self):
    """Return whether the held object touches the placement surface."""
    if not self.activated or self.contact_constraint is None:
      return False

    try:
      object_id = p.getConstraintInfo(self.contact_constraint)[2]
    except:  # pylint: disable=bare-except
      self.contact_constraint = None
      return False

    points = p.getContactPoints(bodyA=object_id)
    points = [point for point in points
              if point[2] not in (self.body, self.robot, object_id)]
    return bool(points)

  def check_grasp(self):
    """Return whether an object is currently secured between the jaws."""
    return self.contact_constraint is not None


class VLMParallelJaw(ParallelJaw):
  """Parallel-jaw gripper that grasps only the object selected by a VLM."""

  def __init__(self, assets_root, robot, ee, obj_ids):
    super().__init__(assets_root, robot, ee, obj_ids)
    self.grasp_command = None
    self.last_grasp_command = None
    self.last_grasp_succeeded = None

  def set_grasp_command(self, command):
    """Queue a validated VLM object selection for the next activation."""
    if not isinstance(command, dict) or "object_id" not in command:
      raise ValueError("A VLM grasp command must include an object_id.")
    self.grasp_command = dict(command)

  def activate(self):
    """Close around the commanded object using the normal jaw mechanics."""
    command = self.grasp_command
    self.grasp_command = None
    self.last_grasp_command = command
    if command is None:
      self.activated = False
      self.last_grasp_succeeded = False
      return

    object_id = int(command["object_id"])
    if object_id not in self.obj_ids["rigid"]:
      self.activated = False
      self.last_grasp_succeeded = False
      return

    all_objects = self.obj_ids
    self.obj_ids = {"rigid": [object_id]}
    try:
      super().activate()
      self.last_grasp_succeeded = bool(self.activated)
    finally:
      self.obj_ids = all_objects


class Suction(Gripper):
  """Simulate simple suction dynamics."""

  def __init__(self, assets_root, robot, ee, obj_ids):
    """Creates suction and 'attaches' it to the robot.

    Has special cases when dealing with rigid vs deformables. For rigid,
    only need to check contact_constraint for any constraint. For soft
    bodies (i.e., cloth or bags), use cloth_threshold to check distances
    from gripper body (self.body) to any vertex in the cloth mesh. We
    need correct code logic to handle gripping potentially a rigid or a
    deformable (and similarly for releasing).

    To be clear on terminology: 'deformable' here should be interpreted
    as a PyBullet 'softBody', which includes cloths and bags. There's
    also cables, but those are formed by connecting rigid body beads, so
    they can use standard 'rigid body' grasping code.

    To get the suction gripper pose, use p.getLinkState(self.body, 0),
    and not p.getBasePositionAndOrientation(self.body) as the latter is
    about z=0.03m higher and empirically seems worse.

    Args:
      assets_root: str for root directory with assets.
      robot: int representing PyBullet ID of robot.
      ee: int representing PyBullet ID of end effector link.
      obj_ids: list of PyBullet IDs of all suctionable objects in the env.
    """
    super().__init__(assets_root)

    # Load suction gripper base model (visual only).
    pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
    base = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, SUCTION_BASE_URDF), pose[0], pose[1])
    p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=base,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, 0.01))

    # Load suction tip model (visual and collision) with compliance.
    # urdf = 'assets/ur5/suction/suction-head.urdf'
    pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))
    self.body = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, SUCTION_HEAD_URDF), pose[0], pose[1])
    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=self.body,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, -0.08))
    p.changeConstraint(constraint_id, maxForce=50)

    # Reference to object IDs in environment for simulating suction.
    self.obj_ids = obj_ids

    # Indicates whether gripper is gripping anything (rigid or def).
    self.activated = False

    # For gripping and releasing rigid objects.
    self.contact_constraint = None

    # Defaults for deformable parameters, and can override in tasks.
    self.def_ignore = 0.035  # TODO(daniel) check if this is needed
    self.def_threshold = 0.030
    self.def_nb_anchors = 1

    # Track which deformable is being gripped (if any), and anchors.
    self.def_grip_item = None
    self.def_grip_anchors = []

    # Determines release when gripped deformable touches a rigid/def.
    # TODO(daniel) should check if the code uses this -- not sure?
    self.def_min_vetex = None
    self.def_min_distance = None

    # Determines release when a gripped rigid touches defs (e.g. cloth-cover).
    self.init_grip_distance = None
    self.init_grip_item = None

  def activate(self):
    """Simulate suction using a rigid fixed constraint to contacted object."""
    # TODO(andyzeng): check deformables logic.
    # del def_ids

    if not self.activated:
      points = p.getContactPoints(bodyA=self.body, linkIndexA=0)
      # print(points)
      if points:

        # Handle contact between suction with a rigid object.
        for point in points:
          obj_id, contact_link = point[2], point[4]
        if obj_id in self.obj_ids['rigid']:
          body_pose = p.getLinkState(self.body, 0)
          obj_pose = p.getBasePositionAndOrientation(obj_id)
          world_to_body = p.invertTransform(body_pose[0], body_pose[1])
          obj_to_body = p.multiplyTransforms(world_to_body[0],
                                             world_to_body[1],
                                             obj_pose[0], obj_pose[1])
          self.contact_constraint = p.createConstraint(
              parentBodyUniqueId=self.body,
              parentLinkIndex=0,
              childBodyUniqueId=obj_id,
              childLinkIndex=contact_link,
              jointType=p.JOINT_FIXED,
              jointAxis=(0, 0, 0),
              parentFramePosition=obj_to_body[0],
              parentFrameOrientation=obj_to_body[1],
              childFramePosition=(0, 0, 0),
              childFrameOrientation=(0, 0, 0))

        self.activated = True

  def release(self):
    """Release gripper object, only applied if gripper is 'activated'.

    If suction off, detect contact between gripper and objects.
    If suction on, detect contact between picked object and other objects.

    To handle deformables, simply remove constraints (i.e., anchors).
    Also reset any relevant variables, e.g., if releasing a rigid, we
    should reset init_grip values back to None, which will be re-assigned
    in any subsequent grasps.
    """
    if self.activated:
      self.activated = False

      # Release gripped rigid object (if any).
      if self.contact_constraint is not None:
        try:
          p.removeConstraint(self.contact_constraint)
          self.contact_constraint = None
        except:  # pylint: disable=bare-except
          pass
        self.init_grip_distance = None
        self.init_grip_item = None

      # Release gripped deformable object (if any).
      if self.def_grip_anchors:
        for anchor_id in self.def_grip_anchors:
          p.removeConstraint(anchor_id)
        self.def_grip_anchors = []
        self.def_grip_item = None
        self.def_min_vetex = None
        self.def_min_distance = None

  def detect_contact(self):
    """Detects a contact with a rigid object."""
    body, link = self.body, 0
    if self.activated and self.contact_constraint is not None:
      try:
        info = p.getConstraintInfo(self.contact_constraint)
        body, link = info[2], info[3]
      except:  # pylint: disable=bare-except
        self.contact_constraint = None
        pass

    # Get all contact points between the suction and a rigid body.
    points = p.getContactPoints(bodyA=body, linkIndexA=link)
    # print(points)
    # exit()
    if self.activated:
      points = [point for point in points if point[2] != self.body]

    # # We know if len(points) > 0, contact is made with SOME rigid item.
    if points:
      return True

    return False

  def check_grasp(self):
    """Check a grasp (object in contact?) for picking success."""

    suctioned_object = None
    if self.contact_constraint is not None:
      suctioned_object = p.getConstraintInfo(self.contact_constraint)[2]
    return suctioned_object is not None
