"""Discretization helpers for planar parallel-jaw grasps."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraspActionSpace:
  """Discretize yaw and jaw width for a dense image-space Q function.

  Parallel-jaw grasps are pi-periodic: rotating a grasp by pi exchanges the
  two identical fingers without changing the action.
  """

  n_rotations: int = 16
  n_widths: int = 5
  min_width: float = 0.015
  max_width: float = 0.12

  def __post_init__(self):
    if self.n_rotations <= 0 or self.n_widths <= 0:
      raise ValueError('n_rotations and n_widths must be positive')
    if not 0 < self.min_width < self.max_width:
      raise ValueError('expected 0 < min_width < max_width')

  @property
  def size(self):
    return self.n_rotations * self.n_widths

  def rotation_bin(self, yaw):
    yaw = float(yaw) % np.pi
    return int(np.floor(yaw * self.n_rotations / np.pi)) % self.n_rotations

  def width_bin(self, width):
    width = float(np.clip(width, self.min_width, self.max_width))
    scaled = ((width - self.min_width) /
              (self.max_width - self.min_width))
    return min(int(np.floor(scaled * self.n_widths)), self.n_widths - 1)

  def encode(self, yaw, width):
    return self.rotation_bin(yaw) * self.n_widths + self.width_bin(width)

  def decode(self, action_bin):
    action_bin = int(action_bin)
    if not 0 <= action_bin < self.size:
      raise ValueError(f'action_bin must be in [0, {self.size})')
    rotation_bin, width_bin = divmod(action_bin, self.n_widths)
    yaw = (rotation_bin + 0.5) * np.pi / self.n_rotations
    width = self.min_width + ((width_bin + 0.5) / self.n_widths) * (
        self.max_width - self.min_width)
    return float(yaw), float(width)
