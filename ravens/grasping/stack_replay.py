"""Replay storage for RGB-D pick-and-place stacking actions."""

from dataclasses import dataclass
import json
import os

import numpy as np


@dataclass(frozen=True)
class StackActionTransition:
  image: np.ndarray
  pick_pixel: np.ndarray
  pick_yaw: float
  width: float
  place_position: np.ndarray
  place_yaw: float
  reward: float
  next_image: np.ndarray
  done: bool
  metadata: dict

  def validate(self):
    image, next_image = np.asarray(self.image), np.asarray(self.next_image)
    pixel = np.asarray(self.pick_pixel)
    place = np.asarray(self.place_position)
    if image.ndim != 3 or image.shape[2] not in (4, 6):
      raise ValueError('image must have shape (height, width, 4 or 6)')
    if next_image.shape != image.shape:
      raise ValueError('next_image must have the same shape as image')
    if pixel.shape != (2,) or not (
        0 <= pixel[0] < image.shape[0] and
        0 <= pixel[1] < image.shape[1]):
      raise ValueError('pick_pixel is outside the image')
    if place.shape != (3,) or not np.all(np.isfinite(place)):
      raise ValueError('place_position must be a finite 3-vector')
    scalars = (self.pick_yaw, self.width, self.place_yaw, self.reward)
    if not np.all(np.isfinite(scalars)) or self.width <= 0:
      raise ValueError('action scalars must be finite and width positive')
    if not isinstance(self.metadata, dict):
      raise ValueError('metadata must be a dictionary')


class StackActionReplayDataset:
  """Directory of independently readable stacking transitions."""

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
        path, image=np.asarray(transition.image, np.float32),
        pick_pixel=np.asarray(transition.pick_pixel, np.int32),
        pick_yaw=np.float32(transition.pick_yaw),
        width=np.float32(transition.width),
        place_position=np.asarray(transition.place_position, np.float32),
        place_yaw=np.float32(transition.place_yaw),
        reward=np.float32(transition.reward),
        next_image=np.asarray(transition.next_image, np.float32),
        done=np.uint8(bool(transition.done)),
        metadata=np.asarray(json.dumps(transition.metadata, sort_keys=True)))
    self.files.append(path)
    return path

  def load(self, index):
    with np.load(self.files[index], allow_pickle=False) as data:
      transition = StackActionTransition(
          image=np.asarray(data['image'], np.float32),
          pick_pixel=np.asarray(data['pick_pixel'], np.int32),
          pick_yaw=float(data['pick_yaw']), width=float(data['width']),
          place_position=np.asarray(data['place_position'], np.float32),
          place_yaw=float(data['place_yaw']), reward=float(data['reward']),
          next_image=np.asarray(data['next_image'], np.float32),
          done=bool(data['done']),
          metadata=json.loads(str(data['metadata'])))
    transition.validate()
    return transition
