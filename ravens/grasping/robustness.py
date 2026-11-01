"""Deterministic perturbation and slip model for parallel-jaw evaluation."""

from dataclasses import asdict
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GraspRobustnessResult:
  """One reproducible robustness trial for a proposed grasp."""

  success: bool
  reason: str
  quality: float
  slip_probability: float
  center_error_m: float
  yaw_error_rad: float
  jaw_width_m: float
  position_noise_xy_m: tuple

  def to_dict(self):
    return asdict(self)


class ParallelJawRobustnessModel:
  """Apply seeded execution noise and quality-dependent slip failures.

  Random draws are keyed by episode seed and dataset object ID. Therefore,
  matched policies see the same disturbance for the same object even if their
  proposed grasp poses differ.
  """

  def __init__(self, seed, position_noise_mm=0.0, yaw_noise_degrees=0.0,
               slip_scale=0.0, open_width=0.15, capture_radius=0.028):
    self.seed = int(seed)
    self.position_sigma = max(0.0, float(position_noise_mm)) / 1000.0
    self.yaw_sigma = math.radians(max(0.0, float(yaw_noise_degrees)))
    self.slip_scale = float(np.clip(slip_scale, 0.0, 1.0))
    self.open_width = float(open_width)
    self.capture_radius = float(capture_radius)

  @property
  def enabled(self):
    return bool(self.position_sigma or self.yaw_sigma or self.slip_scale)

  def evaluate(self, object_meta, local_position, local_yaw):
    object_id = int(object_meta['id'])
    rng = np.random.default_rng(np.random.SeedSequence(
        [self.seed & 0xffffffff, object_id & 0xffffffff]))
    noise_xy = rng.normal(0.0, self.position_sigma, size=2)
    yaw_error = float(rng.normal(0.0, self.yaw_sigma))
    slip_draw = float(rng.random())

    effective_x = float(local_position[0]) + float(noise_xy[0])
    effective_y = float(local_position[1]) + float(noise_xy[1])
    center_error = math.hypot(effective_x, effective_y)
    length, width, _ = (float(value) for value in object_meta['dims'])
    if object_meta['shape'] == 'cylinder':
      jaw_width = length
    else:
      yaw = float(local_yaw) + yaw_error
      jaw_width = abs(length * math.cos(yaw)) + abs(width * math.sin(yaw))

    width_margin = self.open_width - jaw_width - 0.002
    center_quality = math.exp(
        -0.5 * (center_error / max(self.capture_radius * 0.55, 1e-6)) ** 2)
    margin_quality = float(np.clip(width_margin / 0.035, 0.0, 1.0))
    shape_quality = {
        'cube': 1.0,
        'rectangle': 1.0,
        'cylinder': 0.90,
        'triangle': 0.82,
    }.get(object_meta['shape'], 0.85)
    quality = float(np.clip(
        center_quality * (0.35 + 0.65 * margin_quality) * shape_quality,
        0.0, 1.0))
    slip_probability = self.slip_scale * (1.0 - quality)

    if center_error > self.capture_radius:
      success, reason = False, 'missed_center'
    elif width_margin < 0.0:
      success, reason = False, 'jaw_width_exceeded'
    elif slip_draw < slip_probability:
      success, reason = False, 'slip'
    else:
      success, reason = True, 'secured'
    return GraspRobustnessResult(
        success=success,
        reason=reason,
        quality=quality,
        slip_probability=float(slip_probability),
        center_error_m=float(center_error),
        yaw_error_rad=yaw_error,
        jaw_width_m=float(jaw_width),
        position_noise_xy_m=(float(noise_xy[0]), float(noise_xy[1])))
