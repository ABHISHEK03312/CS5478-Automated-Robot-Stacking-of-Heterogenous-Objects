"""Candidate-aligned RGB-D crops for grasp success prediction."""

import numpy as np
from scipy import ndimage


def candidate_crop(image, pixel, yaw, size=48):
  """Sample a square crop whose horizontal axis follows the gripper yaw."""
  image = np.asarray(image, dtype=np.float32)
  if image.ndim != 3:
    raise ValueError('image must have shape (height, width, channels)')
  if size < 2:
    raise ValueError('crop size must be at least 2')
  row, col = map(float, pixel)
  offset = np.arange(size, dtype=np.float32) - (size - 1) / 2
  out_row, out_col = np.meshgrid(offset, offset, indexing='ij')
  cosine, sine = np.cos(yaw), np.sin(yaw)
  source_row = row + cosine * out_row + sine * out_col
  source_col = col - sine * out_row + cosine * out_col
  crop = np.empty((size, size, image.shape[2]), dtype=np.float32)
  for channel in range(image.shape[2]):
    crop[..., channel] = ndimage.map_coordinates(
        image[..., channel], (source_row, source_col), order=1,
        mode='constant', cval=0.0)
  return crop
