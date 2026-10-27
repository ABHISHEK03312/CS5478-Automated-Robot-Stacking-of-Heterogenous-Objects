"""Adapter from a selected scene object to the Ravens Transporter policy."""

import numpy as np

from ravens.utils import utils


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
