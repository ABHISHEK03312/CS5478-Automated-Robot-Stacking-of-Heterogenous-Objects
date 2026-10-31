"""Factory for grasp policies that share the candidate-scoring interface."""

from ravens.grasping.analytic import RobustAnalyticGraspModel
from ravens.grasping.planner import CandidateGraspPlanner


POLICIES = ('geometry', 'candidate_q', 'analytic')


def resolve_policy(policy, checkpoint=''):
  """Preserve the old checkpoint-only CLI while making policy explicit."""
  if policy not in POLICIES:
    raise ValueError(f'unknown grasp policy: {policy}')
  if policy == 'geometry' and checkpoint:
    return 'candidate_q'
  return policy


def create_grasp_planner(policy='geometry', checkpoint='', mc_samples=8,
                         refine_pose=False, min_probability=0.65,
                         max_uncertainty=0.08, min_margin=0.15):
  """Construct a planner adapter, or None for native geometry."""
  policy = resolve_policy(policy, checkpoint)
  if policy == 'geometry':
    return None
  if policy == 'candidate_q':
    if not checkpoint:
      raise ValueError('--grasp_checkpoint is required for candidate_q')
    from ravens.models.grasp_candidate_q import GraspCandidateQModel
    model = GraspCandidateQModel()
    model.load(checkpoint)
  elif policy == 'analytic':
    model = RobustAnalyticGraspModel()
  return CandidateGraspPlanner(
      model, mc_samples=mc_samples, refine_pose=refine_pose,
      min_probability=min_probability, max_uncertainty=max_uncertainty,
      min_margin=min_margin)
