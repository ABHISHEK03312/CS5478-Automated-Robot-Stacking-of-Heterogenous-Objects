"""Execution loop joining a symbolic VLM plan to a pick/place policy."""

from vlm_stacking.ravens_policy import RavensTransporterPolicy
from vlm_stacking.ravens_sim import RavensSimulationExecutor
from vlm_stacking.ravens_sim import RavensSimulationSceneProvider
from vlm_stacking.ravens_sim import is_object_at_target


class LanguageStackPipeline:
  """Assign objects to task slots, then execute through injected robot skills.

  `planner` returns validated object-to-slot assignments. An optional search
  agent optimizes placements around those task-defined slots; otherwise the
  Transporter executes each assigned object at its slot. The scene provider
  returns `(image_bytes, object_catalog, vlm_catalog, state)`.
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
    image_bytes, objects, objects_for_prompt, state = self.scene_provider()
    # The planner owns validation and parsing retries.
    slots = self.scene_provider.slot_catalog(state)
    plan = self.planner.plan(instruction, image_bytes, objects_for_prompt, slots)
    placements = plan["placements"]

    if search_agent is not None:
      search_agent.set_preferred_placements(placements, state["target_slots"])
      steps = []
      for placement in placements:
        expected_id = str(placement["object_id"])
        slot_id = placement["slot_id"]
        object_id = expected_id
        result = None
        for attempt in range(3):
          slot = state["target_slots"][slot_id]
          if is_object_at_target(
              state["task"], expected_id, slot["pose"]):
            result = {"success": True, "already_placed": True,
                      "done": False}
            break
          action = search_agent.act(state["obs"], state.get("info"))
          object_id = getattr(search_agent, "last_object_id", None)
          if action is None or object_id is None:
            raise RuntimeError("Search agent stopped before placing VLM object %s." %
                               expected_id)
          if str(object_id) != expected_id:
            if is_object_at_target(
                state["task"], expected_id, slot["pose"]):
              result = {"success": True, "already_placed": True,
                        "done": False}
              break
            raise RuntimeError("Search agent selected object %s; VLM order "
                               "requires %s." % (object_id, expected_id))
          result = self.executor(
              action, object_id=str(object_id), support_id=slot_id,
              target_pose=slot["pose"],
              support_ids=slot["support_object_ids"])
          if result.get("done") or result.get("success"):
            break
          if attempt == 2:
            raise RuntimeError("Object %s missed slot %s after 3 attempts." %
                               (object_id, slot_id))
          _, _, _, state = self.scene_provider()
        steps.append({"object_id": str(object_id),
                      "slot_id": placement["slot_id"], "result": result})
        if result.get("success") is False:
          raise RuntimeError("Execution stopped after object %s: %s" %
                             (object_id, result))
        image_bytes, objects, objects_for_prompt, state = self.scene_provider()
      return {"plan": plan, "steps": steps,
              "search_agent": search_agent.__class__.__name__,
              "search_rollouts": getattr(search_agent, "n_rollouts", None)}

    if self.policy is None:
      raise ValueError("A Transporter policy is required when no search agent is supplied.")
    steps = []
    for placement in placements:
      object_id = str(placement["object_id"])
      slot_id = placement["slot_id"]
      result = None
      for attempt in range(3):
        slot = state["target_slots"][slot_id]
        action = self.policy.make_action(
            object_id=object_id, support_id=slot_id, state=state)
        result = self.executor(
            action, object_id=object_id, support_id=slot_id,
            target_pose=slot["pose"],
            support_ids=slot["support_object_ids"])
        if result.get("done") or result.get("success"):
          break
        if attempt == 2:
          raise RuntimeError("Object %s missed slot %s after 3 attempts." %
                             (object_id, slot_id))
        _, _, _, state = self.scene_provider()
      steps.append({"object_id": object_id, "slot_id": slot_id,
                    "result": result})
      if result.get("success") is False:
        raise RuntimeError("Execution stopped after object %s: %s" %
                           (object_id, result))
      image_bytes, objects, objects_for_prompt, state = self.scene_provider()
    return {"plan": plan, "steps": steps}
