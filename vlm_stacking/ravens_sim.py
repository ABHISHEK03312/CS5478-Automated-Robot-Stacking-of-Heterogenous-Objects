"""Convenience adapters for the PyBullet Ravens simulator.

These adapters deliberately use Ravens' oracle segmentation and simulator
object state. They make a useful integration prototype, but they are not a
real-camera perception system.
"""

import cv2
import numpy as np
import pybullet as p


class RavensSimulationSceneProvider:
  """Build a labeled top-down scene and inventory from a Ravens environment."""

  def __init__(self, env, task, rigid_only=True):
    self.env = env
    self.task = task
    self.rigid_only = rigid_only

  def __call__(self):
    obs = self.env._get_obs()  # pylint: disable=protected-access
    cmap, hmap, object_mask = self.task.get_true_image(self.env)
    info = self.env.info
    candidate_ids = (self.env.obj_ids["rigid"] if self.rigid_only else
                     self.env.obj_ids["rigid"] + self.env.obj_ids["fixed"])

    objects = []
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
      objects.append({
          "object_id": str(object_id),
          "dimensions": [float(v) for v in dimensions],
          "position_m": [float(v) for v in position],
          "quaternion_xyzw": [float(v) for v in rotation],
          "visual_rgba": ([float(v) for v in shape[0][7]]
                          if shape else None),
          "visible_pixel_count": int(np.count_nonzero(mask)),
      })

    ok, encoded = cv2.imencode(".png", labeled)
    if not ok:
      raise RuntimeError("Could not encode the simulator scene image.")

    cv2.imwrite("labeled_scene.png", labeled)
    state = {
        "obs": obs,
        "info": info,
        "object_mask": object_mask,
        "heightmap": hmap,
        "objects": objects,
        "env": self.env,
        "task": self.task
    }
    return encoded.tobytes(), objects, state


class RavensSimulationMaskProvider:
  """Return the oracle top-down mask for a selected simulator object."""

  def __call__(self, object_id, state):
    return state["object_mask"] == int(object_id)


class RavensSimulationTargetProvider:
  """Place the first object at an anchor, then center on each support object."""

  def __init__(self, anchor_xy=(0.5, 0.0)):
    self.anchor_xy = tuple(anchor_xy)

  def __call__(self, support_id, state):
    if support_id is None:
      return self.anchor_xy
    support = state["info"].get(int(support_id))
    if support is None:
      raise RuntimeError("Planned support object %s is no longer in the scene." %
                         support_id)
    return float(support[0][0]), float(support[0][1])


class RavensSimulationExecutor:
  """Execute one Ravens discrete pick/place action."""

  def __init__(self, env):
    self.env = env

  def __call__(self, action, object_id=None, support_id=None):
    del support_id
    object_id = int(object_id) if object_id is not None else None
    before = None
    if object_id is not None:
      before = np.asarray(p.getBasePositionAndOrientation(object_id)[0])
    obs, reward, done, info = self.env.step(action)
    moved = None
    if object_id is not None:
      after = np.asarray(p.getBasePositionAndOrientation(object_id)[0])
      moved = bool(np.linalg.norm(after - before) > 0.005)
    return {
        # "obs": obs,
        "reward": float(reward),
        "done": bool(done),
        "success": moved,
        # "info": info,
    }
