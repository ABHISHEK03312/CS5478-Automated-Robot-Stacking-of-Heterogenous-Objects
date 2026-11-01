"""Planner adapter for candidate-relative stacking rankers."""

import numpy as np

from ravens.dataset import BOUNDS
from ravens.dataset import PIXEL_SIZE
from ravens.grasping.observations import fused_heightmap
from ravens.grasping.stack_replay import StackActionTransition
from ravens.utils import utils


class StackActionPairScorer:
  """Compare top CEM placements relative to its physics winner."""

  def __init__(self, model, width=0.048):
    self.model = model
    self.width = float(width)
    self.image = None
    self.rerank_calls = 0
    self.rerank_changes = 0
    self.rerank_abstentions = 0

  def prepare(self, obs):
    self.image = fused_heightmap(obs)

  def _transition(self, pick_pose, place_pose):
    pixel = np.asarray(utils.xyz_to_pix(
        pick_pose[0], BOUNDS, PIXEL_SIZE), dtype=np.int32)
    if not (0 <= pixel[0] < self.image.shape[0] and
            0 <= pixel[1] < self.image.shape[1]):
      return None
    return StackActionTransition(
        image=self.image, pick_pixel=pixel,
        pick_yaw=float(utils.quatXYZW_to_eulerXYZ(pick_pose[1])[2]),
        width=self.width, place_position=np.asarray(place_pose[0]),
        place_yaw=float(utils.quatXYZW_to_eulerXYZ(place_pose[1])[2]),
        reward=0.0, next_image=self.image, done=False, metadata={})

  def rerank(self, pick_pose, place_poses, physics_scores,
             min_probability=0.65):
    self.rerank_calls += 1
    reference = self._transition(pick_pose, place_poses[0])
    candidates = [self._transition(pick_pose, pose) for pose in place_poses[1:]]
    if reference is None or any(item is None for item in candidates):
      self.rerank_abstentions += 1
      return 0
    probabilities = self.model.predict_pairs(
        candidates, [reference] * len(candidates), physics_scores[1:],
        [physics_scores[0]] * len(candidates))
    selected = int(np.argmax(probabilities)) + 1
    if probabilities[selected - 1] < min_probability:
      self.rerank_abstentions += 1
      return 0
    self.rerank_changes += 1
    return selected


def create_stack_pair_scorer(checkpoint):
  if not checkpoint:
    return None
  from ravens.models.stack_pair_rank import StackActionPairRankModel
  model = StackActionPairRankModel()
  model.load(checkpoint)
  return StackActionPairScorer(model)
