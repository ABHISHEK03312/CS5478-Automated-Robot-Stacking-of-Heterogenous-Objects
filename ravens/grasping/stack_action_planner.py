"""Adapter from planner pick/place poses to a joint action critic."""

import numpy as np

from ravens.dataset import BOUNDS
from ravens.dataset import PIXEL_SIZE
from ravens.grasping.observations import fused_heightmap
from ravens.grasping.stack_replay import StackActionTransition
from ravens.utils import utils


class StackActionQScorer:
  """Score complete planner actions against the current RGB-D observation."""

  def __init__(self, model, width=0.048):
    self.model = model
    self.width = float(width)
    self.image = None

  def prepare(self, obs):
    self.image = fused_heightmap(obs)

  def quality(self, pick_pose, place_pose):
    if self.image is None:
      raise ValueError('prepare(obs) must be called before quality')
    pixel = np.asarray(utils.xyz_to_pix(
        pick_pose[0], BOUNDS, PIXEL_SIZE), dtype=np.int32)
    if not (0 <= pixel[0] < self.image.shape[0] and
            0 <= pixel[1] < self.image.shape[1]):
      return 0.0
    pick_yaw = utils.quatXYZW_to_eulerXYZ(pick_pose[1])[2]
    place_yaw = utils.quatXYZW_to_eulerXYZ(place_pose[1])[2]
    transition = StackActionTransition(
        image=self.image, pick_pixel=pixel, pick_yaw=float(pick_yaw),
        width=self.width, place_position=np.asarray(place_pose[0]),
        place_yaw=float(place_yaw), reward=0.0,
        next_image=self.image, done=False, metadata={})
    return float(self.model.predict([transition])[0])
