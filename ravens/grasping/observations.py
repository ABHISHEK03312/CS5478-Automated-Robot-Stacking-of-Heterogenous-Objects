"""RGB-D observation conversion shared by grasp tools."""

import numpy as np
from ravens.dataset import BOUNDS
from ravens.dataset import CAMERA_CONFIG
from ravens.dataset import PIXEL_SIZE
from ravens.utils import utils


def fused_heightmap(obs):
  """Return the six-channel representation used by Transporter agents."""
  color, height = utils.get_fused_heightmap(
      obs, CAMERA_CONFIG, BOUNDS, PIXEL_SIZE)
  return np.concatenate((color,
                         height[..., None],
                         height[..., None],
                         height[..., None]), axis=2).astype(np.float32)
