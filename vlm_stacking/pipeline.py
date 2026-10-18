"""Execution loop joining a symbolic VLM plan to a pick/place policy."""

from vlm_stacking.ravens_policy import RavensTransporterPolicy
from vlm_stacking.ravens_sim import RavensSimulationExecutor
from vlm_stacking.ravens_sim import RavensSimulationSceneProvider


class LanguageStackPipeline:
  """Plan object order, then execute through injected robot skills.

  `planner` returns the VLM's validated object order. An optional
  `search_agent` can consume that order and produce its own simulated
  pick/place actions; otherwise `policy.make_action` executes each planned
  object with the baseline Transporter. `scene_provider` returns
  `(image_bytes, object_catalog, perception_state)`. The caller's executor
  performs one action and returns the next observation.
  """

  def __init__(self, planner,
               scene_provider: RavensSimulationSceneProvider,
               policy: RavensTransporterPolicy,
               executor: RavensSimulationExecutor):
    self.planner = planner
    self.scene_provider = scene_provider
    self.policy = policy
    self.executor = executor

  def run(self, instruction, search_agent=None):
    image_bytes, objects, state = self.scene_provider()
    # The planner owns validation and parsing retries.
    plan = self.planner.plan(instruction, image_bytes, objects)

    if search_agent is not None:
      search_agent.set_preferred_order(plan["stack_order"])
      steps = []
      for expected_id in plan["stack_order"]:
        action = search_agent.act(state["obs"], state.get("info"))
        object_id = getattr(search_agent, "last_object_id", None)
        if action is None or object_id is None:
          raise RuntimeError("Search agent stopped before placing VLM object %s." %
                             expected_id)
        if str(object_id) != str(expected_id):
          raise RuntimeError("Search agent selected object %s; VLM order "
                             "requires %s." % (object_id, expected_id))
        result = self.executor(action, object_id=str(object_id))
        steps.append({"object_id": str(object_id), "result": result})
        if result.get("success") is False:
          raise RuntimeError("Execution stopped after object %s: %s" %
                             (object_id, result))
        image_bytes, objects, state = self.scene_provider()
      return {"plan": plan, "steps": steps,
              "search_agent": search_agent.__class__.__name__,
              "search_rollouts": getattr(search_agent, "n_rollouts", None)}

    if self.policy is None:
      raise ValueError("A Transporter policy is required when no search agent is supplied.")
    steps = []
    previous_id = None
    for object_id in plan["stack_order"]:
      action = self.policy.make_action(
          object_id=str(object_id), support_id=previous_id, state=state)
      result = self.executor(action, object_id=str(object_id),
                             support_id=previous_id)
      steps.append({"object_id": str(object_id), "support_id": previous_id,
                    "result": result})
      if result.get("success") is False:
        raise RuntimeError("Execution stopped after object %s: %s" %
                           (object_id, result))
      image_bytes, objects, state = self.scene_provider()
    return {"plan": plan, "steps": steps}
