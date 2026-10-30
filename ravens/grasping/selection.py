"""Action selection from dense planar grasp-quality maps."""

import numpy as np
from scipy import ndimage


def component_center_mask(surface_mask, min_pixels=5):
  """Reduce a surface mask to one representative pixel per component."""
  surface_mask = np.asarray(surface_mask, dtype=bool)
  if surface_mask.ndim != 2:
    raise ValueError('surface_mask must be two-dimensional')
  labels, count = ndimage.label(surface_mask)
  centers = np.zeros_like(surface_mask)
  for label in range(1, count + 1):
    pixels = np.argwhere(labels == label)
    if len(pixels) < min_pixels:
      continue
    centroid = np.mean(pixels, axis=0)
    distance = np.sum((pixels - centroid) ** 2, axis=1)
    row, col = pixels[int(np.argmin(distance))]
    centers[row, col] = True
  return centers


def component_grasp_candidates(surface_mask, pixel_size, min_pixels=5,
                               width_margin=0.008):
  """Create center, principal-axis yaw, and width from depth components."""
  surface_mask = np.asarray(surface_mask, dtype=bool)
  labels, count = ndimage.label(surface_mask)
  candidates = []
  for label in range(1, count + 1):
    pixels = np.argwhere(labels == label)
    if len(pixels) < min_pixels:
      continue
    centroid = np.mean(pixels, axis=0)
    distance = np.sum((pixels - centroid) ** 2, axis=1)
    row, col = map(int, pixels[int(np.argmin(distance))])
    centered = pixels - centroid
    covariance = centered.T @ centered / max(len(pixels) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    yaw = float(np.arctan2(major[0], major[1]) % np.pi)
    minor = vectors[:, int(np.argmin(values))]
    projected = centered @ minor
    width = float((np.max(projected) - np.min(projected) + 1) *
                  pixel_size + width_margin)
    candidates.append({
        'pixel': (row, col),
        'yaw': yaw,
        'width': width,
        'quality': None,
    })
  return candidates


def filter_supported_candidates(candidates, action_space, valid_actions=None):
  if valid_actions is None:
    return list(candidates)
  valid_actions = np.asarray(valid_actions, dtype=bool)
  supported = []
  for candidate in candidates:
    action_bin = action_space.encode(candidate['yaw'], candidate['width'])
    rotation_bin, width_bin = divmod(action_bin, action_space.n_widths)
    if valid_actions[rotation_bin, width_bin]:
      supported.append(candidate)
  return supported


def rank_grasp_candidates(quality, candidates, action_space,
                          valid_actions=None):
  """Return the highest-Q physically proposed grasp candidate."""
  candidates = filter_supported_candidates(
      candidates, action_space, valid_actions)
  if not candidates:
    raise ValueError('no supported grasp candidates')
  best = None
  for candidate in candidates:
    action_bin = action_space.encode(candidate['yaw'], candidate['width'])
    rotation_bin, width_bin = divmod(action_bin, action_space.n_widths)
    row, col = candidate['pixel']
    score = float(quality[row, col, rotation_bin, width_bin])
    if best is None or score > best['quality']:
      best = {**candidate, 'quality': score}
  return best


def select_best_grasp(quality, action_space, valid_mask, valid_actions=None):
  """Select the highest-quality action whose pixel is valid."""
  quality = np.asarray(quality, dtype=np.float32)
  valid_mask = np.asarray(valid_mask, dtype=bool)
  expected = (quality.shape[0], quality.shape[1])
  if quality.ndim != 4 or valid_mask.shape != expected:
    raise ValueError('expected quality (H,W,R,D) and valid_mask (H,W)')
  if (quality.shape[2] != action_space.n_rotations or
      quality.shape[3] != action_space.n_widths):
    raise ValueError('quality action dimensions do not match action_space')
  if not np.any(valid_mask):
    raise ValueError('valid_mask contains no candidate pixels')
  masked = np.where(valid_mask[..., None, None], quality, -np.inf)
  if valid_actions is not None:
    valid_actions = np.asarray(valid_actions, dtype=bool)
    if valid_actions.shape != quality.shape[2:]:
      raise ValueError('valid_actions must have shape (R,D)')
    if not np.any(valid_actions):
      raise ValueError('valid_actions contains no action bins')
    masked = np.where(valid_actions[None, None, ...], masked, -np.inf)
  index = np.unravel_index(np.argmax(masked), masked.shape)
  row, col, rotation_bin, width_bin = map(int, index)
  action_bin = rotation_bin * action_space.n_widths + width_bin
  yaw, width = action_space.decode(action_bin)
  return {
      'pixel': (row, col),
      'yaw': yaw,
      'width': width,
      'quality': float(masked[index]),
  }


def select_random_grasp(rng, action_space, valid_mask, valid_actions=None):
  """Sample a uniformly random valid pixel and discretized grasp action."""
  valid_pixels = np.argwhere(np.asarray(valid_mask, dtype=bool))
  if not len(valid_pixels):
    raise ValueError('valid_mask contains no candidate pixels')
  row, col = valid_pixels[int(rng.integers(len(valid_pixels)))]
  if valid_actions is None:
    action_bin = int(rng.integers(action_space.size))
  else:
    valid_actions = np.asarray(valid_actions, dtype=bool)
    if valid_actions.shape != (action_space.n_rotations,
                               action_space.n_widths):
      raise ValueError('valid_actions must have shape (R,D)')
    action_bins = np.flatnonzero(valid_actions.reshape(-1))
    if not len(action_bins):
      raise ValueError('valid_actions contains no action bins')
    action_bin = int(rng.choice(action_bins))
  yaw, width = action_space.decode(action_bin)
  return {
      'pixel': (int(row), int(col)),
      'yaw': yaw,
      'width': width,
      'quality': None,
  }
