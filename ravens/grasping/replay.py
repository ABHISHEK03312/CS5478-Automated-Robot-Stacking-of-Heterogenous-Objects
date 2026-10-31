"""Sequential RGB-D grasp replay data for compact offline Q-learning."""

from dataclasses import dataclass
import json
import os

import numpy as np


@dataclass(frozen=True)
class GraspTransition:
  image: np.ndarray
  pixel: np.ndarray
  yaw: float
  width: float
  reward: float
  next_image: np.ndarray
  done: bool
  metadata: dict

  def validate(self):
    image = np.asarray(self.image)
    next_image = np.asarray(self.next_image)
    pixel = np.asarray(self.pixel)
    if image.ndim != 3 or image.shape[2] not in (4, 6):
      raise ValueError('image must have shape (height, width, 4 or 6)')
    if next_image.shape != image.shape:
      raise ValueError('next_image must have the same shape as image')
    if pixel.shape != (2,) or not (
        0 <= pixel[0] < image.shape[0] and
        0 <= pixel[1] < image.shape[1]):
      raise ValueError('pixel is outside the image')
    if not np.isfinite(self.yaw):
      raise ValueError('yaw must be finite')
    if not np.isfinite(self.width) or self.width <= 0:
      raise ValueError('width must be finite and positive')
    if not np.isfinite(self.reward):
      raise ValueError('reward must be finite')
    if not isinstance(self.metadata, dict):
      raise ValueError('metadata must be a dictionary')


class GraspReplayDataset:
  """Directory of independently readable grasp transitions."""

  def __init__(self, path):
    self.path = os.path.abspath(path)
    os.makedirs(self.path, exist_ok=True)
    self.files = sorted(
        os.path.join(self.path, name) for name in os.listdir(self.path)
        if name.endswith('.npz'))

  def __len__(self):
    return len(self.files)

  def add(self, transition):
    transition.validate()
    index = len(self.files)
    path = os.path.join(self.path, f'{index:08d}.npz')
    while os.path.exists(path):
      index += 1
      path = os.path.join(self.path, f'{index:08d}.npz')
    np.savez_compressed(
        path,
        image=np.asarray(transition.image, dtype=np.float32),
        pixel=np.asarray(transition.pixel, dtype=np.int32),
        yaw=np.float32(transition.yaw),
        width=np.float32(transition.width),
        reward=np.float32(transition.reward),
        next_image=np.asarray(transition.next_image, dtype=np.float32),
        done=np.uint8(bool(transition.done)),
        metadata=np.asarray(json.dumps(transition.metadata, sort_keys=True)))
    self.files.append(path)
    return path

  def load(self, index):
    with np.load(self.files[index], allow_pickle=False) as data:
      transition = GraspTransition(
          image=np.asarray(data['image'], dtype=np.float32),
          pixel=np.asarray(data['pixel'], dtype=np.int32),
          yaw=float(data['yaw']), width=float(data['width']),
          reward=float(data['reward']),
          next_image=np.asarray(data['next_image'], dtype=np.float32),
          done=bool(data['done']),
          metadata=json.loads(str(data['metadata'])))
    transition.validate()
    return transition
