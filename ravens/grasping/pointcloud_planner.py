"""Planner adapter for external point-cloud grasp proposal models."""

import numpy as np
import pybullet as p

from ravens.grasping.observations import fused_heightmap
from ravens.grasping.planner import CandidateGraspPlanner
from ravens.grasping.pointcloud import heightmap_pointcloud
from ravens.grasping.pointcloud import planar_grasp_candidates


class PointCloudGraspPlanner(CandidateGraspPlanner):
  """Adapt GPD/Contact-GraspNet-style proposals to planar Ravens planners."""

  def __init__(self, backend, max_tilt_degrees=20, **kwargs):
    super().__init__(model=None, **kwargs)
    self.backend = backend
    self.max_tilt_degrees = float(max_tilt_degrees)

  def prepare(self, obs, object_ids):
    image = fused_heightmap(obs)
    cloud = heightmap_pointcloud(
        image, min_surface_height=self.min_surface_height)
    proposals = self.backend.predict(cloud['points'], cloud['colors'])
    ranked = planar_grasp_candidates(
        proposals, max_tilt_degrees=self.max_tilt_degrees)
    positions = {
        oid: np.asarray(p.getBasePositionAndOrientation(oid)[0], dtype=float)
        for oid in object_ids}
    self._by_object = {}
    for candidate in ranked:
      if not positions:
        break
      xyz = np.asarray(candidate['position'], dtype=float)
      oid = min(positions, key=lambda key: np.linalg.norm(
          positions[key][:2] - xyz[:2]))
      if np.linalg.norm(positions[oid][:2] - xyz[:2]) > self.max_match_distance:
        continue
      if oid not in self._by_object:
        self._by_object[oid] = candidate
    return self._by_object
