"""On-disk dataset for positive and negative RGB-D grasp attempts."""

from dataclasses import dataclass
import json
import os

import numpy as np


@dataclass(frozen=True)
class GraspSample:
  """A single planar grasp attempt expressed in a fused heightmap."""

  image: np.ndarray
  pixel: np.ndarray
  yaw: float
  width: float
  success: bool
  metadata: dict

  def validate(self):
    image = np.asarray(self.image)
    pixel = np.asarray(self.pixel)
    if image.ndim != 3 or image.shape[2] not in (4, 6):
      raise ValueError('image must have shape (height, width, 4 or 6)')
    if pixel.shape != (2,):
      raise ValueError('pixel must have shape (2,)')
    if not (0 <= pixel[0] < image.shape[0] and
            0 <= pixel[1] < image.shape[1]):
      raise ValueError('pixel is outside the image')
    if not np.isfinite(self.yaw):
      raise ValueError('yaw must be finite')
    if not np.isfinite(self.width) or self.width <= 0:
      raise ValueError('width must be finite and positive')
    if not isinstance(self.metadata, dict):
      raise ValueError('metadata must be a dictionary')


class GraspDataset:
  """Directory of compressed, independently readable grasp attempts."""

  def __init__(self, path):
    self.path = os.path.abspath(path)
    os.makedirs(self.path, exist_ok=True)
    self._refresh()

  def _refresh(self):
    self.files = sorted(
        os.path.join(self.path, name) for name in os.listdir(self.path)
        if name.endswith('.npz'))

  def __len__(self):
    return len(self.files)

  def add(self, sample):
    sample.validate()
    index = len(self.files)
    path = os.path.join(self.path, f'{index:08d}.npz')
    while os.path.exists(path):
      index += 1
      path = os.path.join(self.path, f'{index:08d}.npz')
    np.savez_compressed(
        path,
        image=np.asarray(sample.image, dtype=np.float32),
        pixel=np.asarray(sample.pixel, dtype=np.int32),
        yaw=np.float32(sample.yaw),
        width=np.float32(sample.width),
        success=np.uint8(bool(sample.success)),
        metadata=np.asarray(json.dumps(sample.metadata, sort_keys=True)))
    self.files.append(path)
    return path

  def load(self, index):
    with np.load(self.files[index], allow_pickle=False) as data:
      sample = GraspSample(
          image=np.asarray(data['image'], dtype=np.float32),
          pixel=np.asarray(data['pixel'], dtype=np.int32),
          yaw=float(data['yaw']),
          width=float(data['width']),
          success=bool(data['success']),
          metadata=json.loads(str(data['metadata'])))
    sample.validate()
    return sample

  def sample(self, rng=None):
    if not self.files:
      raise ValueError(f'grasp dataset is empty: {self.path}')
    rng = rng or np.random.default_rng()
    return self.load(int(rng.integers(len(self.files))))

  def success_rate(self):
    if not self.files:
      return float('nan')
    return float(np.mean([self.load(i).success for i in range(len(self))]))
