"""Adapter from a selected scene object to the Ravens Transporter policy."""

import numpy as np

from ravens.utils import utils
from vlm_stacking.planner import OllamaGraspPlanner
from vlm_stacking.ravens_sim import RavensSimulationTargetProvider


class RavensTransporterPolicy:
  """Constrain Transporter pick/place predictions using an object-level plan.

  `mask_provider(object_id, state)` must return a boolean mask aligned with
  Ravens' fused heightmap. `target_provider(slot_id, state)` returns the desired
  object pose for the named slot. The policy selects a masked grasp and
  compensates for the measured grasp offset so the object reaches that pose.
  """

  def __init__(self, transporter, mask_provider, target_provider,
               target_radius_m=0.012):
    self.transporter = transporter
    self.mask_provider = mask_provider
    self.target_provider = target_provider
    self.target_radius_m = target_radius_m

  def make_action(self, object_id, support_id, state):
    img = self.transporter.get_image(state["obs"])
    mask = np.asarray(self.mask_provider(object_id, state), dtype=bool)
    if mask.shape != img.shape[:2]:
      raise ValueError("Object mask must match fused heightmap size %r, got %r" %
                       (img.shape[:2], mask.shape))
    if not np.any(mask):
      raise RuntimeError("Selected object %s is not visible/pickable." %
                         object_id)

    pick_conf = self.transporter.attention.forward(img)
    if pick_conf.ndim == 3:
      allowed = np.broadcast_to(mask[:, :, None], pick_conf.shape)
    else:
      allowed = mask
    masked_pick = np.where(allowed, pick_conf, -np.inf)
    p0_index = np.unravel_index(np.argmax(masked_pick), masked_pick.shape)
    p0_pix = p0_index[:2]
    p0_theta = 0.0
    if len(p0_index) > 2:
      p0_theta = p0_index[2] * (2 * np.pi / pick_conf.shape[2])

    hmap = img[:, :, 3]
    p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.transporter.bounds,
                              self.transporter.pix_size)
    p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
    pick_pose = (np.asarray(p0_xyz), np.asarray(p0_xyzw))

    target_object_pose = self.target_provider(support_id, state)
    object_state = next((item for item in state["objects"]
                         if str(item["object_id"]) == str(object_id)), None)
    if object_state is None:
      raise RuntimeError("No measured pose is available for object %s." % object_id)
    object_pose = (np.asarray(object_state["position_m"], dtype=np.float64),
                   np.asarray(object_state["quaternion_xyzw"], dtype=np.float64))
    object_to_pick = utils.multiply(utils.invert(pick_pose), object_pose)
    pick_to_object = utils.invert(object_to_pick)
    place_pose = utils.multiply(target_object_pose, pick_to_object)
    return {
        "pose0": pick_pose,
        "pose1": (np.asarray(place_pose[0]), np.asarray(place_pose[1])),
    }


class OllamaVLMGraspPolicy:
  """Use Ollama to choose a grasp, then execute the same parallel-jaw motion.

  The scene provider supplies a labeled image and measured object states. The
  VLM only returns a bounded yaw and local grasp offset; this adapter converts
  that command into simulator poses and preserves the requested object target.
  """

  def __init__(self, grasp_planner: OllamaGraspPlanner, target_provider: RavensSimulationTargetProvider):
    self.grasp_planner = grasp_planner
    self.target_provider = target_provider
    self.grasp_commands = {}
    self.last_command = None

  def prepare_grasps(self, instruction, state):
    """Ask once for commands for every object visible in the initial scene."""
    self.grasp_commands = self.grasp_planner.plan_all(
        instruction, state.get("image_bytes"),
        state.get("objects_for_prompt", state["objects"]))

  def refresh_grasp(self, object_id, instruction, state):
    """Replan one object's grasp after its cached grasp failed to acquire it."""
    object_id = str(object_id)
    object_state = next((item for item in state["objects"]
                         if str(item["object_id"]) == object_id), None)
    command = self.grasp_planner.plan(
        instruction, state.get("image_bytes"), object_state,
        state.get("objects_for_prompt", state["objects"]))
    command["object_id"] = object_id
    self.grasp_commands[object_id] = command
    return command

  def make_action(self, object_id, support_id, state, base_action=None,
                  instruction="Build a stable pyramid using all the blocks."):
    object_id = str(object_id)
    object_state = next((item for item in state["objects"]
                         if str(item["object_id"]) == object_id), None)
    command = self.grasp_commands.get(object_id)
    if command is None:
      command = self.refresh_grasp(object_id, instruction, state)
    self.last_command = dict(command)
    gripper = getattr(state.get("env"), "ee", None)
    if gripper is not None and hasattr(gripper, "set_grasp_command"):
      gripper.set_grasp_command(self.last_command)

    dimensions = np.asarray(object_state["dimensions"], dtype=np.float64)
    object_pose = (np.asarray(object_state["position_m"], dtype=np.float64),
                   np.asarray(object_state["quaternion_xyzw"], dtype=np.float64))
    local_offset = (np.asarray((
        command["local_offset_x_fraction"] * dimensions[0],
        command["local_offset_y_fraction"] * dimensions[1], 0.0)),
        (0.0, 0.0, 0.0, 1.0))
    offset_pose = utils.multiply(object_pose, local_offset)
    pick_position = np.asarray(offset_pose[0], dtype=np.float64)
    pick_position[2] = object_pose[0][2] + dimensions[2] / 2.0
    pick_rotation = utils.eulerXYZ_to_quatXYZW((
        0.0, 0.0, np.deg2rad(command["grasp_yaw_degrees"])))
    pick_pose = (pick_position, np.asarray(pick_rotation, dtype=np.float64))

    if base_action is not None:
      original_pick = base_action["pose0"]
      original_place = base_action["pose1"]
      object_to_original_pick = utils.multiply(
          utils.invert(original_pick), object_pose)
      target_object_pose = utils.multiply(original_place,
                                          object_to_original_pick)
    else:
      target_object_pose = self.target_provider(support_id, state)

    object_to_pick = utils.multiply(utils.invert(pick_pose), object_pose)
    pick_to_object = utils.invert(object_to_pick)
    place_pose = utils.multiply(target_object_pose, pick_to_object)
    return {
        "pose0": pick_pose,
        "pose1": (np.asarray(place_pose[0]), np.asarray(place_pose[1])),
    }
