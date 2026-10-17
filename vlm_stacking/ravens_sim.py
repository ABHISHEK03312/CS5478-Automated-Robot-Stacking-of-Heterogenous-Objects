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
        "obs": obs,
        "reward": float(reward),
        "done": bool(done),
        "success": moved,
        "info": info,
    }


class RavensSimulationStackEvaluator:
  """Check candidate towers with a save/restore PyBullet stability rollout.

  Objects are placed at the fixed base center in the candidate order, allowed
  to settle under gravity, and checked for displacement and tilt. This checks
  stacking stability for this simulator scene; it does not simulate robot arm
  reachability or the Transporter pick/place trajectory.
  """

  def __init__(self, env, settle_steps=240, position_tolerance=0.008,
               tilt_tolerance=0.20):
    self.env = env
    self.settle_steps = max(1, int(settle_steps))
    self.position_tolerance = float(position_tolerance)
    self.tilt_tolerance = float(tilt_tolerance)

  def __call__(self, plan, objects):
    import math
    import pybullet as p
    by_id = {str(item["object_id"]): item for item in objects}
    base_id = self.env.obj_ids["fixed"][0]
    base_pos, base_quat = p.getBasePositionAndOrientation(base_id)
    base_dims = self.env.info[base_id][2]
    anchor_xy = (float(base_pos[0]), float(base_pos[1]))
    lower_z = float(base_pos[2]) + float(base_dims[2]) / 2.0
    targets = {}
    for object_id in plan["stack_order"]:
      item = by_id[str(object_id)]
      dims = item.get("dimensions")
      position = item.get("position_m")
      if dims is None or len(dims) < 3 or position is None:
        return {"feasible": False, "stability_score": 0.0,
                "reason": "missing dimensions or source position"}
      bounds = self.env.position_bounds
      if not (bounds.low[0] <= position[0] <= bounds.high[0] and
              bounds.low[1] <= position[1] <= bounds.high[1]):
        return {"feasible": False, "stability_score": 0.0,
                "reason": "source object is outside robot workspace"}
      height = float(dims[2])
      lower_z += height / 2.0
      targets[str(object_id)] = (anchor_xy[0], anchor_xy[1], lower_z)
      lower_z += height / 2.0

    state_id = p.saveState()
    try:
      for object_id in plan["stack_order"]:
        body = int(object_id)
        p.resetBasePositionAndOrientation(body, targets[str(object_id)],
                                          base_quat)
        p.resetBaseVelocity(body, (0, 0, 0), (0, 0, 0))
      for _ in range(self.settle_steps):
        p.stepSimulation()

      max_displacement = 0.0
      max_tilt = 0.0
      for object_id in plan["stack_order"]:
        position, quaternion = p.getBasePositionAndOrientation(int(object_id))
        target = targets[str(object_id)]
        displacement = math.sqrt(sum((position[i] - target[i]) ** 2
                                     for i in range(3)))
        roll, pitch, _ = p.getEulerFromQuaternion(quaternion)
        max_displacement = max(max_displacement, displacement)
        max_tilt = max(max_tilt, abs(roll), abs(pitch))
      feasible = (max_displacement <= self.position_tolerance and
                  max_tilt <= self.tilt_tolerance)
      score = max(0.0, 1.0 - max_displacement /
                  max(self.position_tolerance, 1e-6) -
                  max_tilt / max(self.tilt_tolerance, 1e-6))
      return {"feasible": feasible, "stability_score": score,
              "max_displacement_m": max_displacement,
              "max_tilt_rad": max_tilt,
              "reason": "settled within tolerances" if feasible else
                        "candidate moved or tilted beyond tolerance"}
    finally:
      p.restoreState(stateId=state_id)
      p.removeState(state_id)
