"""Adapter from a selected scene object to the Ravens Transporter policy."""

import numpy as np

from ravens.utils import utils


class RavensTransporterPolicy:
  """Constrain Transporter pick/place predictions using an object-level plan.

  `mask_provider(object_id, state)` must return an object mask aligned with
  Ravens' 320x160 heightmap. `target_provider(support_id, state)` returns a
  target `(x, y)` in world coordinates. The standard policy is allowed to
  choose the grasp pixel and local orientation only inside the selected
  object's mask; its place distribution is restricted to a small neighborhood
  around the geometric stacking target.
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

    target_xy = self.target_provider(support_id, state)
    target_xyz = np.asarray([target_xy[0], target_xy[1], 0.0], dtype=np.float32)
    target_pix = utils.xyz_to_pix(
        target_xyz, self.transporter.bounds, self.transporter.pix_size)
    if not (0 <= target_pix[0] < img.shape[0] and
            0 <= target_pix[1] < img.shape[1]):
      raise ValueError("Planned stack target is outside the heightmap bounds.")
    place_conf = self.transporter.transport.forward(img, p0_pix)

    yy, xx = np.ogrid[:place_conf.shape[0], :place_conf.shape[1]]
    radius_px = max(1, int(round(
        self.target_radius_m / self.transporter.pix_size)))
    target_region = ((yy - target_pix[0]) ** 2 +
                     (xx - target_pix[1]) ** 2 <= radius_px ** 2)
    target_region = np.broadcast_to(target_region[:, :, None], place_conf.shape)
    if not np.any(target_region):
      raise ValueError("No valid place pixels near the planned support.")
    restricted_place = np.where(target_region, place_conf, -np.inf)
    if not np.any(np.isfinite(restricted_place)):
      raise RuntimeError("Transporter returned no valid place score.")
    p1_index = np.unravel_index(np.argmax(restricted_place),
                                restricted_place.shape)
    p1_pix = p1_index[:2]
    p1_theta = p1_index[2] * (2 * np.pi / place_conf.shape[2])

    hmap = img[:, :, 3]
    p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.transporter.bounds,
                              self.transporter.pix_size)
    p1_xyz = utils.pix_to_xyz(p1_pix, hmap, self.transporter.bounds,
                              self.transporter.pix_size)
    p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
    p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))
    return {
        "pose0": (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
        "pose1": (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
    }
