"""Convenience adapters for the PyBullet Ravens simulator.

These adapters deliberately use Ravens' oracle segmentation and simulator
object state. They make a useful integration prototype, but they are not a
real-camera perception system.
"""

import cv2
import numpy as np
import pybullet as p
from ravens.utils import utils

def _color_name_from_rgba(rgba):
  """Map a simulator RGB(A) value to the nearest Ravens palette color."""
  if rgba is None or len(rgba) < 3:
    return "unknown"
  rgb = np.asarray(rgba[:3], dtype=np.float32)
  palette = {name: np.asarray(value, dtype=np.float32)
             for name, value in utils.COLORS.items()}
  color_name = min(palette, key=lambda name: np.linalg.norm(rgb - palette[name]))
  distance = np.linalg.norm(rgb - palette[color_name])
  return color_name if distance <= 0.25 else "unknown"


class RavensSimulationSceneProvider:
  """Build a labeled top-down scene and inventory from a Ravens environment."""

  def __init__(self, env, task, rigid_only=True, debug_image_path="labeled_scene.png"):
    self.env = env
    self.task = task
    self.rigid_only = rigid_only
    self.debug_image_path = debug_image_path
    self._target_slots = target_slot_poses(task, env)

  @staticmethod
  def slot_catalog(state):
    colors = {str(obj["object_id"]): obj.get("color", "unknown")
              for obj in state["objects_for_prompt"]}
    catalog = []
    for slot in state["target_slots"].values():
      row = {key: value for key, value in slot.items()
             if key not in ("pose", "eligible_object_ids", "support_object_ids")}
      row["allowed_object_ids"] = list(slot["eligible_object_ids"])
      row["allowed_colors"] = sorted({
          colors[object_id] for object_id in slot["eligible_object_ids"]
          if object_id in colors})
      catalog.append(row)
    return catalog

  def __call__(self):
    obs = self.env._get_obs()  # pylint: disable=protected-access
    cmap, hmap, object_mask = self.task.get_true_image(self.env)
    info = self.env.info
    candidate_ids = (self.env.obj_ids["rigid"] if self.rigid_only else
                     self.env.obj_ids["rigid"] + self.env.obj_ids["fixed"])

    objects = []
    objects_for_prompt = []
    labeled = cmap.copy()
    for object_id in candidate_ids:
      mask = object_mask == int(object_id)
      if not np.any(mask):
        continue
      state = info.get(object_id)
      if state is None:
        continue
      position, rotation, dimensions = state
      shape = p.getVisualShapeData(object_id)
      ys, xs = np.where(mask)
      center = (int(np.mean(xs)), int(np.mean(ys)))
      label = str(object_id)
      # Explicit labels let the VLM map its symbolic plan back to simulator
      # IDs. The catalog independently provides metric geometry.
      cv2.circle(labeled, center, 3, (255, 255, 255), -1)
      cv2.putText(labeled, label, (center[0] + 4, center[1] - 4),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2,
                  cv2.LINE_AA)
      cv2.putText(labeled, label, (center[0] + 4, center[1] - 4),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                  cv2.LINE_AA)
      visual_rgba = ([float(v) for v in shape[0][7]] if shape else None)
      objects.append({
          "object_id": str(object_id),
          "color": _color_name_from_rgba(visual_rgba),
          "dimensions": [float(v) for v in dimensions],
          "position_m": [float(v) for v in position],
          "quaternion_xyzw": [float(v) for v in rotation],
          "visual_rgba": visual_rgba,
          "visible_pixel_count": int(np.count_nonzero(mask)),
      })
      objects_for_prompt.append({
          "object_id": str(object_id),
          "color": _color_name_from_rgba(visual_rgba),
          "dimensions": [float(v) for v in dimensions],
      })

    # Put symbolic target labels in the same image coordinate frame as the
    # objects so the VLM can resolve left/right instructions consistently.
    for slot_id, slot in self._target_slots.items():
      row, col = utils.xyz_to_pix(
          slot["pose"][0], self.task.bounds, self.task.pix_size)
      if 0 <= row < labeled.shape[0] and 0 <= col < labeled.shape[1]:
        cv2.putText(labeled, slot_id, (int(col), int(row)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(labeled, slot_id, (int(col), int(row)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1,
                    cv2.LINE_AA)

    ok, encoded = cv2.imencode(".png", labeled)
    if not ok:
      raise RuntimeError("Could not encode the simulator scene image.")

    if self.debug_image_path:
      cv2.imwrite(self.debug_image_path, labeled)
    state = {
        "obs": obs,
        "info": info,
        "object_mask": object_mask,
        "heightmap": hmap,
        "objects": objects,
        "objects_for_prompt": objects_for_prompt,
        "target_slots": self._target_slots,
        "env": self.env,
        "task": self.task
    }
    return encoded.tobytes(), objects, objects_for_prompt, state


class RavensSimulationMaskProvider:
  """Return the oracle top-down mask for a selected simulator object."""

  def __call__(self, object_id, state):
    return state["object_mask"] == int(object_id)


def target_slot_poses(task, env=None):
  """Return stable named slots from the task's goal poses."""
  slots = {}
  names = (("base_left", "base_center", "base_right"),
           ("middle_left", "middle_right"), ("apex",))
  descriptions = (
      ("left base position on the stand", "center base position on the stand",
       "right base position on the stand"),
      ("left middle position, supported by the base row",
       "right middle position, supported by the base row"),
      ("apex position, supported by both middle blocks",))
  for layer_index, goal in enumerate(task.goals):
    targets = goal[2]
    layer_names = names[layer_index] if layer_index < len(names) else tuple(
        "layer_%d_slot_%d" % (layer_index, i) for i in range(len(targets)))
    layer_descriptions = (descriptions[layer_index]
                          if layer_index < len(descriptions) else tuple(
                              "slot %d in layer %d" % (i, layer_index)
                              for i in range(len(targets))))
    for index, target in enumerate(targets):
      slot_id = layer_names[index] if index < len(layer_names) else (
          "layer_%d_slot_%d" % (layer_index, index))
      slots[slot_id] = {
          "slot_id": slot_id,
          "layer_index": layer_index,
          "description": (layer_descriptions[index]
                          if index < len(layer_descriptions) else slot_id),
          "pose": target,
          "eligible_object_ids": [str(object_id)
                                  for object_id, _ in goal[0]],
          "support_object_ids": (
              [int(object_id) for object_id, _ in task.goals[layer_index - 1][0]]
              if layer_index > 0 else
              list(env.obj_ids["fixed"]) if env is not None else []),
      }
  return slots


def target_slot_catalog(task, objects=None):
  """Prompt-safe slot descriptions without simulator coordinates."""
  colors = {str(obj["object_id"]): obj.get("color", "unknown")
            for obj in (objects or [])}
  catalog = []
  for slot in target_slot_poses(task).values():
    row = {key: value for key, value in slot.items()
           if key not in ("pose", "eligible_object_ids", "support_object_ids")}
    row["allowed_object_ids"] = list(slot["eligible_object_ids"])
    if objects is not None:
      row["allowed_colors"] = sorted({
          colors[object_id] for object_id in slot["eligible_object_ids"]
          if object_id in colors})
    catalog.append(row)
  return catalog


def is_object_at_target(task, object_id, target_pose):
  """Match the same XY/yaw criterion used by Ravens and the search agents."""
  object_id = int(object_id)
  symmetry = np.pi / 2
  for goal in task.goals:
    for candidate, (candidate_symmetry, _) in goal[0]:
      if candidate == object_id:
        symmetry = candidate_symmetry
        break
  pose = p.getBasePositionAndOrientation(object_id)
  return task.is_match(pose, target_pose, symmetry)


class RavensSimulationTargetProvider:
  """Map a named target slot to the full desired object pose."""

  def __init__(self, anchor_xy=(0.5, 0.0)):
    self.anchor_xy = tuple(anchor_xy)

  def __call__(self, slot_id, state):
    slots = state.get("target_slots", {})
    if slot_id not in slots:
      raise RuntimeError("Unknown planned target slot %s." % slot_id)
    return slots[slot_id]["pose"]


class RavensSimulationExecutor:
  """Execute one Ravens discrete pick/place action."""

  def __init__(self, env, task=None):
    self.env = env
    self.task = task

  def __call__(self, action, object_id=None, support_id=None,
               target_pose=None, support_ids=None):
    del support_id
    object_id = int(object_id) if object_id is not None else None
    before = None
    symmetry = np.pi / 2
    if object_id is not None and self.task is not None:
      for goal in self.task.goals:
        for candidate, (candidate_symmetry, _) in goal[0]:
          if candidate == object_id:
            symmetry = candidate_symmetry
            break
    if object_id is not None:
      before = np.asarray(p.getBasePositionAndOrientation(object_id)[0])
    obs, reward, done, info = self.env.step(action)
    moved = None
    placement_verified = None
    position_error = None
    height_error = None
    if object_id is not None:
      after_pose = p.getBasePositionAndOrientation(object_id)
      after = np.asarray(after_pose[0])
      moved = bool(np.linalg.norm(after - before) > 0.005)
      if target_pose is not None:
        position_error = float(np.linalg.norm(
            after[:2] - np.asarray(target_pose[0][:2])))
        height_error = float(abs(after[2] - target_pose[0][2]))
        # Ravens' pyramid reward and the search agents match XY/yaw only.
        # The task's Z targets include a known offset from physical support
        # height, so retain Z as a diagnostic without using it to reject a
        # placement.
        pose_match = (self.task.is_match(after_pose, target_pose, symmetry)
                      if self.task is not None else position_error < 0.01)
        contact = True
        if support_ids is not None:
          contact = any(p.getContactPoints(
              bodyA=object_id, bodyB=int(support_id))
                        for support_id in support_ids)
        placement_verified = bool(pose_match and contact)
    timed_out = bool(done and self.task is not None and
                     not self.task.done())
    return {
        # "obs": obs,
        "reward": float(reward),
        "done": bool(done),
        "timed_out": timed_out,
        "success": (placement_verified if placement_verified is not None else moved),
        "moved": moved,
        "placement_verified": placement_verified,
        "position_error_xy_m": position_error,
        "position_error_z_m": height_error,
        # "info": info,
    }
