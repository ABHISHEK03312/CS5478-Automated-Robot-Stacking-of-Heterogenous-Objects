"""Adapter from candidate grasp Q scores to simulator planner pick poses."""

import numpy as np
import pybullet as p
from ravens.dataset import BOUNDS, PIXEL_SIZE
from ravens.grasping.observations import fused_heightmap
from ravens.grasping.selection import component_grasp_candidates
from ravens.utils import utils


class CandidateGraspPlanner:
  """Cache the best RGB-D grasp proposal for each visible rigid object."""

  def __init__(self, model, min_surface_height=0.015, mc_samples=8,
               uncertainty_weight=1.0, max_match_distance=0.06,
               refine_pose=False, min_probability=0.65,
               max_uncertainty=0.08, min_margin=0.15):
    self.model = model
    self.min_surface_height = min_surface_height
    self.mc_samples = mc_samples
    self.uncertainty_weight = uncertainty_weight
    self.max_match_distance = max_match_distance
    self.refine_pose = refine_pose
    self.min_probability = min_probability
    self.max_uncertainty = max_uncertainty
    self.min_margin = min_margin
    self._by_object = {}
    self.selection_calls = 0
    self.selection_changes = 0
    self.selection_abstentions = 0

  def prepare(self, obs, object_ids):
    image = fused_heightmap(obs)
    mask = image[..., 3] >= self.min_surface_height
    candidates = component_grasp_candidates(mask, PIXEL_SIZE)
    ranked = self.model.score_candidates(
        image, candidates, self.mc_samples, self.uncertainty_weight)
    positions = {
        oid: np.asarray(p.getBasePositionAndOrientation(oid)[0], dtype=float)
        for oid in object_ids}
    self._by_object = {}
    for candidate in ranked:
      xyz = np.asarray(utils.pix_to_xyz(
          candidate['pixel'], image[..., 3], BOUNDS, PIXEL_SIZE), dtype=float)
      if not positions:
        break
      oid = min(positions, key=lambda key: np.linalg.norm(
          positions[key][:2] - xyz[:2]))
      if np.linalg.norm(positions[oid][:2] - xyz[:2]) > self.max_match_distance:
        continue
      if oid not in self._by_object:
        self._by_object[oid] = {**candidate, 'position': xyz}
    return self._by_object

  def quality(self, object_id):
    candidate = self._by_object.get(object_id)
    return candidate['quality'] if candidate is not None else 0.0

  def select_object(self, object_ids, fallback=None):
    """Select a confidently better object, or abstain to the fallback."""
    object_ids = list(object_ids)
    self.selection_calls += 1
    scored = [(oid, self._by_object[oid]) for oid in object_ids
              if oid in self._by_object]
    scored.sort(key=lambda item: item[1]['quality'], reverse=True)
    if not scored:
      self.selection_abstentions += 1
      return fallback
    top_id, top = scored[0]
    runner_up = scored[1][1]['quality'] if len(scored) > 1 else 0.0
    confident = (top['probability'] >= self.min_probability and
                 top['uncertainty'] <= self.max_uncertainty and
                 top['quality'] - runner_up >= self.min_margin)
    if not confident:
      self.selection_abstentions += 1
      return fallback
    if fallback is None or top_id != fallback:
      self.selection_changes += 1
    return top_id

  def pick_pose(self, object_id, fallback):
    if not self.refine_pose:
      return fallback
    candidate = self._by_object.get(object_id)
    if candidate is None:
      return fallback
    rotation = utils.eulerXYZ_to_quatXYZW((0, 0, candidate['yaw']))
    return np.float64(candidate['position']), np.float64(rotation)
