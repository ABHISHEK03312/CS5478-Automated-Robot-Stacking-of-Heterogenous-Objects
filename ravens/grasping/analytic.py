"""Dex-Net-inspired analytic robustness scoring for planar grasps.

This is not the Dex-Net GQ-CNN. It provides a lightweight, dependency-free
baseline with the same candidate-scoring interface as the learned grasp model.
Candidates are scored repeatedly under deterministic pixel and yaw
perturbations, then ranked by mean geometric quality minus uncertainty.
"""

import numpy as np
from scipy import ndimage

from ravens.dataset import PIXEL_SIZE


class RobustAnalyticGraspModel:
  """Score centered, flat, isolated, width-feasible planar grasps."""

  _PERTURBATIONS = (
      (0, 0, 0.0),
      (-2, 0, 0.0),
      (2, 0, 0.0),
      (0, -2, 0.0),
      (0, 2, 0.0),
      (0, 0, -np.deg2rad(5)),
      (0, 0, np.deg2rad(5)),
      (-1, 1, np.deg2rad(3)),
  )

  def __init__(self, min_surface_height=0.015, min_width=0.015,
               max_width=0.15, width_margin=0.008,
               flatness_scale=0.004, clearance_scale=0.025):
    self.min_surface_height = float(min_surface_height)
    self.min_width = float(min_width)
    self.max_width = float(max_width)
    self.width_margin = float(width_margin)
    self.flatness_scale = float(flatness_scale)
    self.clearance_scale = float(clearance_scale)

  def _score(self, depth, labels, label, candidate, perturbation,
             isolation):
    row_offset, col_offset, yaw_offset = perturbation
    row = int(candidate['pixel'][0] + row_offset)
    col = int(candidate['pixel'][1] + col_offset)
    if not (0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]):
      return 0.0
    if labels[row, col] != label:
      return 0.0

    pixels = np.argwhere(labels == label).astype(np.float32)
    centered = pixels - np.asarray((row, col), dtype=np.float32)
    yaw = float(candidate['yaw']) + yaw_offset
    major = np.asarray((np.sin(yaw), np.cos(yaw)), dtype=np.float32)
    minor = np.asarray((np.cos(yaw), -np.sin(yaw)), dtype=np.float32)
    major_projection = centered @ major
    minor_projection = centered @ minor
    half_major = max(float(np.max(np.abs(major_projection))), 1.0)
    half_minor = max(float(np.max(np.abs(minor_projection))), 1.0)

    centroid_offset = pixels.mean(axis=0) - (row, col)
    normalized_offset = np.sqrt(
        (float(centroid_offset @ major) / half_major) ** 2 +
        (float(centroid_offset @ minor) / half_minor) ** 2)
    centering = float(np.exp(-2.5 * normalized_offset))

    component_depth = depth[labels == label]
    flatness = float(np.exp(
        -np.std(component_depth) / max(self.flatness_scale, 1e-6)))

    observed_width = ((np.max(minor_projection) -
                       np.min(minor_projection) + 1) * PIXEL_SIZE +
                      self.width_margin)
    requested_width = float(candidate['width'])
    if not self.min_width <= requested_width <= self.max_width:
      width_fit = 0.0
    else:
      width_fit = float(np.exp(
          -abs(requested_width - observed_width) / 0.02))

    return float(np.clip(
        0.40 * centering + 0.25 * flatness + 0.20 * isolation +
        0.15 * width_fit, 0.0, 1.0))

  def score_candidates(self, image, candidates, mc_samples=8,
                       uncertainty_weight=1.0):
    """Return candidates ranked by perturbation-robust analytic quality."""
    if not candidates:
      return []
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 4:
      raise ValueError('image must have shape (height, width, at least 4)')
    depth = image[..., 3]
    surface = depth >= self.min_surface_height
    labels, _ = ndimage.label(surface)
    passes = max(1, min(int(mc_samples), len(self._PERTURBATIONS)))
    ranked = []
    for candidate in candidates:
      row, col = map(int, candidate['pixel'])
      if not (0 <= row < depth.shape[0] and 0 <= col < depth.shape[1]):
        continue
      label = int(labels[row, col])
      if label == 0:
        continue
      other = surface & (labels != label)
      if np.any(other):
        distance_to_other = ndimage.distance_transform_edt(~other)
        separation = float(np.min(distance_to_other[labels == label]) *
                           PIXEL_SIZE)
        isolation = float(np.clip(separation /
                                  max(self.clearance_scale, 1e-6), 0, 1))
      else:
        isolation = 1.0
      scores = np.asarray([
          self._score(depth, labels, label, candidate, perturbation, isolation)
          for perturbation in self._PERTURBATIONS[:passes]
      ], dtype=np.float32)
      mean = float(np.mean(scores))
      uncertainty = float(np.std(scores))
      ranked.append({
          **candidate,
          'probability': mean,
          'uncertainty': uncertainty,
          'quality': mean - float(uncertainty_weight) * uncertainty,
      })
    return sorted(ranked, key=lambda item: item['quality'], reverse=True)
