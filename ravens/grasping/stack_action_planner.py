"""Adapter from planner pick/place poses to a joint action critic."""

import numpy as np

from ravens.dataset import BOUNDS
from ravens.dataset import PIXEL_SIZE
from ravens.grasping.observations import fused_heightmap
from ravens.grasping.stack_replay import StackActionTransition
from ravens.utils import utils


class StackActionQScorer:
  """Score complete planner actions against the current RGB-D observation."""

  def __init__(self, model, width=0.048):
    self.models = list(model) if isinstance(model, (list, tuple)) else [model]
    self.width = float(width)
    self.image = None
    self.rerank_calls = 0
    self.rerank_changes = 0
    self.rerank_abstentions = 0

  def prepare(self, obs):
    self.image = fused_heightmap(obs)

  def _transition(self, pick_pose, place_pose):
    if self.image is None:
      raise ValueError('prepare(obs) must be called before quality')
    pixel = np.asarray(utils.xyz_to_pix(
        pick_pose[0], BOUNDS, PIXEL_SIZE), dtype=np.int32)
    if not (0 <= pixel[0] < self.image.shape[0] and
            0 <= pixel[1] < self.image.shape[1]):
      return None
    pick_yaw = utils.quatXYZW_to_eulerXYZ(pick_pose[1])[2]
    place_yaw = utils.quatXYZW_to_eulerXYZ(place_pose[1])[2]
    return StackActionTransition(
        image=self.image, pick_pixel=pixel, pick_yaw=float(pick_yaw),
        width=self.width, place_position=np.asarray(place_pose[0]),
        place_yaw=float(place_yaw), reward=0.0,
        next_image=self.image, done=False, metadata={})
  def quality_stats(self, pick_pose, place_pose):
    """Return ensemble mean and standard deviation for one action."""
    transition = self._transition(pick_pose, place_pose)
    if transition is None:
      return 0.0, 1.0
    values = np.asarray([
        float(model.predict([transition])[0]) for model in self.models])
    return float(np.mean(values)), float(np.std(values))

  def quality(self, pick_pose, place_pose):
    return self.quality_stats(pick_pose, place_pose)[0]

  def rerank(self, pick_pose, place_poses, base_scores, weight=0.1,
             max_uncertainty=0.05, min_margin=0.05):
    """Conservatively choose among physics-ranked placement candidates."""
    if not place_poses or len(place_poses) != len(base_scores):
      raise ValueError('place_poses and base_scores must have equal length')
    self.rerank_calls += 1
    stats = [self.quality_stats(pick_pose, pose) for pose in place_poses]
    means = np.asarray([item[0] for item in stats])
    uncertainty = np.asarray([item[1] for item in stats])
    combined = np.asarray(base_scores, dtype=np.float64) + weight * means
    selected = int(np.argmax(combined))
    confident = (uncertainty[selected] <= max_uncertainty and
                 means[selected] - means[0] >= min_margin)
    if selected == 0:
      return 0
    if not confident:
      self.rerank_abstentions += 1
      return 0
    self.rerank_changes += 1
    return selected


def create_stack_action_scorer(checkpoint):
  """Load a placement-aware scorer, or disable it for an empty path."""
  checkpoints = ([checkpoint] if isinstance(checkpoint, str) and checkpoint
                 else [] if isinstance(checkpoint, str) else list(checkpoint))
  if not checkpoints:
    return None
  from ravens.models.stack_action_q import StackActionQModel
  models = []
  for path in checkpoints:
    model = StackActionQModel()
    model.load(path)
    models.append(model)
  return StackActionQScorer(models)
