"""Point-cloud conversion and external 6-DoF grasp adapter utilities."""

import os
import subprocess
import tempfile

import numpy as np

from ravens.dataset import BOUNDS
from ravens.dataset import CAMERA_CONFIG
from ravens.dataset import PIXEL_SIZE
from ravens.utils import utils


def heightmap_pointcloud(image, min_surface_height=0.015,
                         bounds=BOUNDS, pixel_size=PIXEL_SIZE):
  """Convert a fused orthographic RGB-D image to a world-frame point cloud."""
  image = np.asarray(image)
  if image.ndim != 3 or image.shape[2] < 4:
    raise ValueError('image must have shape (height, width, at least 4)')
  height = np.asarray(image[..., 3], dtype=np.float32)
  rows, cols = np.nonzero(height >= float(min_surface_height))
  points = np.column_stack((
      bounds[0, 0] + cols * pixel_size,
      bounds[1, 0] + rows * pixel_size,
      bounds[2, 0] + height[rows, cols],
  )).astype(np.float32)
  colors = np.clip(image[rows, cols, :3], 0, 255).astype(np.uint8)
  pixels = np.column_stack((rows, cols)).astype(np.int32)
  return {'points': points, 'colors': colors, 'pixels': pixels}


def planar_grasp_candidates(proposals, max_tilt_degrees=20,
                            min_width=0.015, max_width=0.15,
                            bounds=BOUNDS, pixel_size=PIXEL_SIZE):
  """Filter world-frame 6-DoF proposals to reachable top-down grasps.

  Each proposal must contain position, approach, closing_axis, width, and
  score. The approach vector points in the direction of gripper travel.
  """
  cosine_limit = np.cos(np.deg2rad(float(max_tilt_degrees)))
  candidates = []
  for proposal in proposals:
    position = np.asarray(proposal['position'], dtype=np.float64)
    approach = np.asarray(proposal['approach'], dtype=np.float64)
    closing = np.asarray(proposal['closing_axis'], dtype=np.float64)
    width = float(proposal['width'])
    score = float(proposal['score'])
    uncertainty = float(proposal.get('uncertainty', 0.0))
    if (position.shape != (3,) or approach.shape != (3,) or
        closing.shape != (3,)):
      raise ValueError('proposal vectors must have shape (3,)')
    if not np.all(np.isfinite(np.concatenate(
        (position, approach, closing, (width, score, uncertainty))))):
      continue
    if not min_width <= width <= max_width:
      continue
    if np.any(position < bounds[:, 0]) or np.any(position > bounds[:, 1]):
      continue
    approach_norm = np.linalg.norm(approach)
    closing_xy_norm = np.linalg.norm(closing[:2])
    if approach_norm < 1e-8 or closing_xy_norm < 1e-8:
      continue
    approach = approach / approach_norm
    if float(approach @ np.asarray((0., 0., -1.))) < cosine_limit:
      continue
    closing_xy = closing[:2] / closing_xy_norm
    yaw = float(np.arctan2(closing_xy[1], closing_xy[0]) % np.pi)
    pixel = utils.xyz_to_pix(position, bounds, pixel_size)
    if not (0 <= pixel[0] < int(np.ceil(
        (bounds[1, 1] - bounds[1, 0]) / pixel_size)) and
            0 <= pixel[1] < int(np.ceil(
                (bounds[0, 1] - bounds[0, 0]) / pixel_size))):
      continue
    probability = float(np.clip(score, 0, 1))
    candidates.append({
        'pixel': pixel,
        'yaw': yaw,
        'width': width,
        'probability': probability,
        'uncertainty': max(0.0, uncertainty),
        'quality': probability - max(0.0, uncertainty),
        'position': position,
    })
  return sorted(candidates, key=lambda item: item['quality'], reverse=True)


class NpzCommandGraspBackend:
  """Run an isolated grasp model using a small NPZ file protocol."""

  def __init__(self, command, timeout=120):
    if not command:
      raise ValueError('command must contain at least one argument')
    self.command = list(command)
    self.timeout = float(timeout)

  def predict(self, points, colors):
    """Return proposals emitted by an external GPD/Contact-GraspNet wrapper."""
    with tempfile.TemporaryDirectory(prefix='ravens-grasp-') as directory:
      input_path = os.path.join(directory, 'scene.npz')
      output_path = os.path.join(directory, 'grasps.npz')
      np.savez_compressed(input_path, points=np.asarray(points, np.float32),
                          colors=np.asarray(colors, np.uint8),
                          frame=np.asarray('world'),
                          camera_positions=np.asarray([
                              config['position'] for config in CAMERA_CONFIG
                          ], dtype=np.float32))
      subprocess.run(
          self.command + ['--input', input_path, '--output', output_path],
          check=True, timeout=self.timeout)
      if not os.path.isfile(output_path):
        raise ValueError('external grasp backend did not create its output')
      with np.load(output_path) as result:
        required = ('positions', 'approaches', 'closing_axes', 'widths',
                    'scores')
        missing = [name for name in required if name not in result]
        if missing:
          raise ValueError(f'external grasp output missing: {missing}')
        arrays = {name: np.asarray(result[name]) for name in required}
        uncertainties = (np.asarray(result['uncertainties'])
                         if 'uncertainties' in result else
                         np.zeros_like(arrays['scores'], dtype=np.float32))
    count = len(arrays['scores'])
    if any(len(value) != count for value in arrays.values()):
      raise ValueError('external grasp output arrays have different lengths')
    if len(uncertainties) != count:
      raise ValueError('external grasp uncertainties have the wrong length')
    return [{
        'position': arrays['positions'][index],
        'approach': arrays['approaches'][index],
        'closing_axis': arrays['closing_axes'][index],
        'width': arrays['widths'][index],
        'score': arrays['scores'][index],
        'uncertainty': uncertainties[index],
    } for index in range(count)]
