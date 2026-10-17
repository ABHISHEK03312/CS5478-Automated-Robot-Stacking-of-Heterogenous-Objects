"""Execution loop joining a symbolic VLM plan to a pick/place policy."""

from vlm_stacking.ravens_policy import RavensTransporterPolicy
from vlm_stacking.ravens_sim import RavensSimulationExecutor
from vlm_stacking.ravens_sim import RavensSimulationSceneProvider


class LanguageStackPipeline:
  """Plan object order, then execute through injected robot skills.

  `planner` can be the zero-shot VLM baseline or a search-based planner that
  implements the same `plan(instruction, image_bytes, objects)` interface.
  `scene_provider` returns `(image_bytes, object_catalog, perception_state)`.
  Each catalog item has `object_id`, measured `dimensions`, and optional
  semantic metadata. `policy.make_action` must restrict its pick search to the
  requested object's mask and return Ravens' `{pose0, pose1}` action. The
  caller's executor should perform one action and return the next observation.
  """

  def __init__(self, planner,
               scene_provider: RavensSimulationSceneProvider,
               policy: RavensTransporterPolicy,
               executor: RavensSimulationExecutor):
    self.planner = planner
    self.scene_provider = scene_provider
    self.policy = policy
    self.executor = executor

  def run(self, instruction):
    image_bytes, objects, state = self.scene_provider()
    # The planner owns validation and retries, so callers consistently receive
    # either a valid plan or a final StackPlanError.
    plan = self.planner.plan(instruction, image_bytes, objects)

    steps = []
    previous_id = None
    for object_id in plan["stack_order"]:
      # The target is either the table anchor or the top face of the preceding
      # planned object. The scene adapter recalculates the target from current
      # depth after each completed movement.
      action = self.policy.make_action(
          object_id=str(object_id), support_id=previous_id, state=state)
      result = self.executor(action, object_id=str(object_id),
                             support_id=previous_id)
      steps.append({"object_id": str(object_id), "support_id": previous_id,
                    "result": result})
      if result.get("success") is False:
        raise RuntimeError("Execution stopped after object %s: %s" % (
            object_id, result))
      # Re-localize after each placement. Do not keep using stale pixel masks
      # or stack heights after the scene has changed.
      image_bytes, objects, state = self.scene_provider()
    return {"plan": plan, "steps": steps}
