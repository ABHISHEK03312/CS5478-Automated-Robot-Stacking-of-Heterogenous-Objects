"""Evaluate VLM stacking strategies on seed-matched Ravens test scenes.

The Hanoi dataset is used only to obtain its held-out episode seeds. Each seed
initializes the parallel-jaw block-pyramid task so all three strategies see
identical, reproducible pyramid scenes. Hanoi images and actions are not used.
"""

import argparse
import json
import math
import os
import time
import traceback

import numpy as np
from scipy import stats

from ravens import tasks
from ravens.agents.beam import BeamSearchAgent
from ravens.agents.cem import CEMAgent
from ravens.agents.transporter import OriginalTransporterAgent
from ravens.dataset import Dataset
from ravens.environments.environment import Environment
from vlm_stacking.planner import OllamaStackPlanner
from vlm_stacking.ravens_policy import RavensTransporterPolicy
from vlm_stacking.ravens_sim import RavensSimulationExecutor
from vlm_stacking.ravens_sim import RavensSimulationMaskProvider
from vlm_stacking.ravens_sim import RavensSimulationSceneProvider
from vlm_stacking.ravens_sim import RavensSimulationTargetProvider
from vlm_stacking.ravens_sim import is_object_at_target
from vlm_stacking.ravens_sim import target_slot_catalog

TASK_NAME = "stack-block-pyramid-parallel-jaw"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SEED_DATASET = os.path.join(
    REPO_ROOT, "data", "parallel-jaw-v2", TASK_NAME + "-test")


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--seed-dataset", default=DEFAULT_SEED_DATASET,
      help="Ravens test dataset folder; its episode seeds define pyramid scenes")
  parser.add_argument("--assets-root", default="ravens/environments/assets")
  parser.add_argument("--models-root", default="models/parallel-jaw-v2")
  parser.add_argument(
      "--checkpoint", default="stack-block-pyramid-parallel-jaw-transporter-100-0")
  parser.add_argument("--checkpoint-step", type=int, default=40000)
  parser.add_argument("--instruction", default=(
      "Build a stable pyramid using all the blocks."))
  parser.add_argument("--model", default=None,
                      help="Ollama model; defaults to OLLAMA_VLM_MODEL")
  parser.add_argument("--temperature", type=float, default=0.0001,
                      help="Ollama sampling temperature (default: 0.0001)")
  parser.add_argument("--num-plans", type=int, default=1,
                      help="Validated VLM plans to compare when temperature "
                           "is not 0.0001")
  parser.add_argument("--techniques", nargs="+",
                      choices=("zero-shot", "beam", "cem", "all"),
                      default=["all"])
  parser.add_argument("--num-test-cases", "--max-episodes",
                      dest="max_episodes", type=int, default=0,
                      help="Number of test cases; 0 evaluates the entire test set")
  parser.add_argument("--seed-offset", type=int, default=0,
                      help="Offset added to each source dataset seed")
  parser.add_argument("--beam-width", type=int, default=3)
  parser.add_argument("--beam-horizon", type=int, default=2)
  parser.add_argument("--beam-placements", type=int, default=6)
  parser.add_argument("--cem-samples", type=int, default=24)
  parser.add_argument("--cem-elites", type=int, default=5)
  parser.add_argument("--cem-iterations", type=int, default=3)
  parser.add_argument("--verify-k", type=int, default=3)
  parser.add_argument("--rollout", choices=("teleport", "full"),
                      default="teleport")
  parser.add_argument("--seed", type=int, default=11,
                      help="Search-agent sampling seed offset")
  parser.add_argument("--disp", action="store_true")
  parser.add_argument("--output", default="vlm_stacking_eval.json",
                      help="JSON file for per-episode results and summaries")
  parser.add_argument("--diagnostics-dir", default="vlm_stacking_diagnostics",
                      help="Directory for separate failure-category JSON logs")
  args = parser.parse_args()
  if not math.isfinite(args.temperature) or args.temperature < 0:
    parser.error("--temperature must be a finite, non-negative number")
  if args.num_plans < 1:
    parser.error("--num-plans must be at least 1")
  return args


def dataset_seeds(path, max_episodes=0, seed_offset=0):
  if not os.path.isdir(path):
    raise FileNotFoundError("Seed dataset not found: %s" % path)
  dataset = Dataset(path)
  if dataset.n_episodes == 0:
    raise ValueError("No episodes found in seed dataset: %s" % path)
  count = min(max_episodes, dataset.n_episodes) if max_episodes > 0 else dataset.n_episodes
  seeds = [dataset.load(i, images=False)[1] + seed_offset
           for i in range(count)]
  return seeds


def make_search_agent(technique, env, task, args, seed):
  if technique == "beam":
    return BeamSearchAgent(
        env, task, objective="target", beam_width=args.beam_width,
        horizon=args.beam_horizon, n_place=args.beam_placements,
        verify_k=args.verify_k, rollout="teleport", seed=args.seed + seed)
  if technique == "cem":
    return CEMAgent(
        env, task, init="target", n_samples=args.cem_samples, n_elites=args.cem_elites,
        n_iters=args.cem_iterations, verify_k=args.verify_k,
        rollout=args.rollout, seed=args.seed + seed)
  return None


def _pose_dict(pose):
  if pose is None:
    return None
  return {
      "position_m": [float(value) for value in pose[0]],
      "quaternion_xyzw": [float(value) for value in pose[1]],
  }


def _object_poses(env, colors=None):
  import pybullet as p
  colors = colors or {}
  objects = {}
  for object_id in env.obj_ids["rigid"]:
    position, rotation = p.getBasePositionAndOrientation(object_id)
    bounds = p.getAABB(object_id)
    objects[str(object_id)] = {
        "color": colors.get(str(object_id), "unknown"),
        "position_m": [float(value) for value in position],
        "quaternion_xyzw": [float(value) for value in rotation],
        "aabb_min_m": [float(value) for value in bounds[0]],
        "aabb_max_m": [float(value) for value in bounds[1]],
    }
  return objects


def _goal_snapshot(task):
  """Report per-object target matches for all currently active goal layers."""
  import pybullet as p
  groups = []
  for group_index, goal in enumerate(task.goals):
    objs, matches, targets, _, _, metric, _, max_reward = goal
    object_rows = []
    matched_count = 0
    for object_index, (object_id, (symmetry, _)) in enumerate(objs):
      pose = p.getBasePositionAndOrientation(object_id)
      matching_targets = [target_index for target_index in
                          np.argwhere(matches[object_index, :]).reshape(-1)
                          if task.is_match(pose, targets[target_index], symmetry)]
      matched = bool(matching_targets)
      matched_count += int(matched)
      object_rows.append({
          "object_id": str(object_id),
          "matched": matched,
          "matching_target_indices": [int(index) for index in matching_targets],
          "position_m": [float(value) for value in pose[0]],
      })
    groups.append({
        "active_group_index": group_index,
        "metric": metric,
        "max_reward": float(max_reward),
        "matched_objects": matched_count,
        "total_objects": len(objs),
        "reward_if_scored_now": (float(max_reward) * matched_count / len(objs)
                                  if objs and metric == "pose" else None),
        "objects": object_rows,
        "target_poses": [_pose_dict(target) for target in targets],
    })
  return groups


class CountingExecutor:
  """Count executed robot trials and retain per-step diagnostic state."""

  def __init__(self, env, task, record):
    self.env = env
    self.task = task
    self.record = record
    self.executor = RavensSimulationExecutor(env, task)
    self.trials = 0
    self.reward = 0.0
    self.colors = {}

  def __call__(self, action, object_id=None, support_id=None,
               target_pose=None, support_ids=None):
    before = _object_poses(self.env, self.colors)
    result = self.executor(action, object_id=object_id,
                           support_id=support_id, target_pose=target_pose,
                           support_ids=support_ids)
    self.trials += 1
    self.reward += result["reward"]
    self.record["steps"].append({
        "trial": self.trials,
        "object_id": str(object_id) if object_id is not None else None,
        "support_id": str(support_id) if support_id is not None else None,
        "action": {
            "pick_pose": _pose_dict(action.get("pose0")),
            "place_pose": _pose_dict(action.get("pose1")),
        },
        "objects_before": before,
        "objects_after": _object_poses(self.env, self.colors),
        "reward_delta": float(result["reward"]),
        "cumulative_reward": float(self.reward),
        "done": bool(result["done"]),
        "timed_out": bool(result.get("timed_out", False)),
        "selected_object_moved": result.get("moved"),
        "object_moved": result.get("moved"),
        "placement_verified": result.get("placement_verified"),
        "placement_error_xy_m": result.get("position_error_xy_m"),
        "placement_error_z_m": result.get("position_error_z_m"),
        "task_progress": float(self.task.progress),
        "task_reward_state": float(self.task._rewards),
        "active_goal_matches": _goal_snapshot(self.task),
    })
    return result


def _initial_goal_spec(task):
  return [{
      "group_index": index,
      "object_ids": [str(object_id) for object_id, _ in goal[0]],
      "max_reward": float(goal[7]),
      "target_poses": [_pose_dict(target) for target in goal[2]],
  } for index, goal in enumerate(task.goals)]


def write_diagnostic_files(records_by_technique, directory):
  records = [record for technique_records in records_by_technique.values()
             for record in technique_records]
  categories = {
      "partial_towers.json": [record for record in records
                               if record["outcome"] == "partial_pyramid"],
      "error_cases.json": [record for record in records
                           if record["error"] is not None],
      "zero_reward_cases.json": [record for record in records
                                 if record["outcome"] == "zero_reward"],
  }
  os.makedirs(directory, exist_ok=True)
  written = {}
  for filename, cases in categories.items():
    destination = os.path.abspath(os.path.join(directory, filename))
    with open(destination, "w", encoding="utf-8") as output_file:
      json.dump({"case_count": len(cases), "cases": cases}, output_file,
                indent=2)
    written[filename] = destination
  return written


def _plan_selection_score(record):
  """Compare complete candidate runs by task reward and final scene quality."""
  matched = sum(group["matched_objects"]
               for group in record.get("remaining_goal_matches", []))
  timeouts = sum(bool(step.get("timed_out"))
                 for step in record.get("steps", []))
  failed = int(record.get("error") is not None)
  return (float(record.get("reward", 0.0)), matched, -failed,
          -timeouts, -int(record.get("trials", 0)))


def _plan_selection_details(record):
  score = _plan_selection_score(record)
  return {
      "task_reward": score[0],
      "matched_goal_objects": score[1],
      "episode_error": -score[2],
      "timeouts": -score[3],
      "trials": -score[4],
      "priority": ["task_reward", "matched_goal_objects",
                   "no_episode_error", "fewer_timeouts", "fewer_trials"],
  }


def sample_plan_candidates(env, task, seed, planner, args):
  """Sample one validated plan batch from a reproducible initial scene."""
  started = time.time()
  np.random.seed(seed)
  env.seed(seed)
  env.set_task(task)
  env.reset()
  scene = RavensSimulationSceneProvider(env, task, debug_image_path=None)
  image, objects, objects_for_prompt, _ = scene()
  slots = target_slot_catalog(task, objects_for_prompt)
  sample_count = args.num_plans if args.temperature != 0.0001 else 1
  plans = []
  sampling_errors = []
  for candidate_index in range(sample_count):
    try:
      plan = planner.plan(args.instruction, image, objects_for_prompt, slots)
      plans.append((candidate_index, plan))
    except Exception as error:  # One bad sample should not discard valid plans.
      sampling_errors.append({
          "candidate_index": candidate_index,
          "error": "%s: %s" % (type(error).__name__, error),
      })
  return plans, sampling_errors, round(time.time() - started, 3)


def run_episode(technique, env, task, seed, planner, policy, args,
                sampled_plans=None, sampling_errors=None,
                sampling_seconds=0.0):
  """Evaluate a shared VLM plan batch from one seed and keep the best."""
  if sampled_plans is None:
    sampled_plans, sampling_errors, sampling_seconds = sample_plan_candidates(
        env, task, seed, planner, args)
  sampling_errors = sampling_errors or []
  sample_count = args.num_plans if args.temperature != 0.0001 else 1
  candidate_started = time.time()

  if not sampled_plans:
    np.random.seed(seed)
    env.seed(seed)
    env.set_task(task)
    env.reset()
    scene = RavensSimulationSceneProvider(env, task, debug_image_path=None)
    _, objects, _, _ = scene()
    colors = {str(obj["object_id"]): obj.get("color", "unknown")
              for obj in objects}
    record = {
        "seed": int(seed), "technique": technique,
        "instruction": args.instruction, "trials": 0, "reward": 0.0,
        "plan": None, "error": "All VLM plan samples failed.",
        "failure_stage": "VLM planning", "object_catalog": objects,
        "steps": [], "plan_candidates": [],
        "plan_sampling_errors": sampling_errors,
        "temperature": float(args.temperature),
        "requested_plan_samples": int(args.num_plans),
        "effective_plan_samples": int(sample_count),
        "evaluated_plan_samples": 0,
        "initial_object_poses": _object_poses(env, colors),
        "goal_specification": _initial_goal_spec(task),
    }
    record["error"] += " " + "; ".join(
        item["error"] for item in sampling_errors)
    record["plan_sampling_seconds"] = float(sampling_seconds)
    record["candidate_search_seconds"] = round(
        time.time() - candidate_started, 3)
    record["seconds"] = round(
        record["plan_sampling_seconds"] + record["candidate_search_seconds"], 3)
    record["final_object_poses"] = _object_poses(env, colors)
    record["final_task_progress"] = float(task.progress)
    record["final_task_reward_state"] = float(task._rewards)
    record["remaining_goal_matches"] = _goal_snapshot(task)
    record["outcome"] = "zero_reward"
    return record

  candidates = []
  candidate_records = []
  for original_index, plan in sampled_plans:
    candidate = _run_episode_with_plan(
        technique, env, task, seed, planner, policy, args,
        plan, original_index)
    candidate_records.append(candidate)
    candidates.append({
        "candidate_index": original_index,
        "plan": plan,
        "score": _plan_selection_details(candidate),
        "reward": candidate["reward"],
        "trials": candidate["trials"],
        "outcome": candidate["outcome"],
        "failure_stage": candidate["failure_stage"],
        "error": candidate["error"],
        "seconds": candidate["seconds"],
        "final_task_progress": candidate["final_task_progress"],
        "remaining_goal_matches": candidate["remaining_goal_matches"],
        "timeout_count": sum(bool(step.get("timed_out"))
                              for step in candidate["steps"]),
        "steps": [{
            "trial": step["trial"],
            "object_id": step["object_id"],
            "support_id": step["support_id"],
            "placement_verified": step["placement_verified"],
            "timed_out": step["timed_out"],
            "reward_delta": step["reward_delta"],
            "placement_error_xy_m": step["placement_error_xy_m"],
        } for step in candidate["steps"]],
    })
  best = max(candidate_records, key=_plan_selection_score)
  best["plan_candidates"] = candidates
  best["plan_sampling_errors"] = sampling_errors
  best["selected_plan_index"] = best.pop("plan_candidate_index")
  best["plan_selection_score"] = _plan_selection_details(best)
  best["temperature"] = float(args.temperature)
  best["requested_plan_samples"] = int(args.num_plans)
  best["effective_plan_samples"] = int(sample_count)
  best["evaluated_plan_samples"] = len(candidate_records)
  best["unique_assignments"] = len({
      tuple((str(row["object_id"]), row["slot_id"])
            for row in candidate_plan["placements"])
      for _, candidate_plan in sampled_plans
  })
  best["selected_plan_trials"] = int(best["trials"])
  best["candidate_evaluation_trials"] = sum(
      int(candidate["trials"]) for candidate in candidate_records)
  best["candidate_evaluation_seconds"] = round(sum(
      float(candidate["seconds"]) for candidate in candidate_records), 3)
  best["plan_sampling_seconds"] = float(sampling_seconds)
  best["candidate_search_seconds"] = round(
      time.time() - candidate_started, 3)
  best["seconds"] = round(
      best["plan_sampling_seconds"] + best["candidate_search_seconds"], 3)
  return best


def _run_episode_with_plan(technique, env, task, seed, planner, policy, args,
                           plan, candidate_index):
  np.random.seed(seed)
  env.seed(seed)
  env.set_task(task)
  obs = env.reset()
  scene = RavensSimulationSceneProvider(
      env, task, debug_image_path=None)
  record = {"seed": int(seed), "technique": technique,
            "instruction": args.instruction, "trials": 0,
            "reward": 0.0, "plan": None, "error": None,
            "failure_stage": None, "object_catalog": None,
            "steps": []}
  executor = CountingExecutor(env, task, record)
  started = time.time()
  current_stage = "scene perception"

  try:
    image, objects, objects_for_prompt, state = scene()
    record["object_catalog"] = objects
    executor.colors = {str(obj["object_id"]): obj.get("color", "unknown")
                       for obj in objects}
    record["initial_object_poses"] = _object_poses(env, executor.colors)
    record["goal_specification"] = _initial_goal_spec(task)
    record["plan"] = plan
    record["plan_candidate_index"] = candidate_index
    placements = plan["placements"]

    search_agent = make_search_agent(technique, env, task, args, seed)
    if search_agent is not None:
      search_agent.set_preferred_placements(placements, state["target_slots"])
      placements_by_object = {
          str(placement["object_id"]): placement
          for placement in placements
      }
      repair_actions = 0
      max_repair_actions = max(3, 2 * len(placements))
      info = None
      for placement in placements:
        expected_id = str(placement["object_id"])
        slot_id = placement["slot_id"]
        slot = state["target_slots"][slot_id]
        result = None
        attempt = 0
        while attempt < 3:
          if is_object_at_target(task, expected_id, slot["pose"]):
            result = {"success": True, "already_placed": True,
                      "done": False}
            break
          current_stage = "search action selection"
          action = search_agent.act(obs, info)
          object_id = getattr(search_agent, "last_object_id", None)
          if action is None or object_id is None:
            raise RuntimeError("Search agent stopped before its next planned pick.")
          selected_id = str(object_id)
          if selected_id != expected_id:
            repair = placements_by_object.get(selected_id)
            if repair is None:
              raise RuntimeError(
                  "Search agent selected unplanned object %s while plan "
                  "requires %s." % (selected_id, expected_id))
            if repair_actions >= max_repair_actions:
              raise RuntimeError(
                  "Search agent exceeded the %d-action corrective placement "
                  "budget while plan requires object %s." %
                  (max_repair_actions, expected_id))

            repair_slot_id = repair["slot_id"]
            repair_slot = state["target_slots"][repair_slot_id]
            if is_object_at_target(task, selected_id, repair_slot["pose"]):
              # The scene may have changed since the previous search step.
              # Refresh it so beam does not act on a stale candidate.
              repair_actions += 1
              _, _, _, state = scene()
              obs, info = state["obs"], state.get("info")
              slot = state["target_slots"][slot_id]
              continue

            current_stage = "search corrective action execution"
            result = executor(
                action, object_id=selected_id, support_id=repair_slot_id,
                target_pose=repair_slot["pose"],
                support_ids=repair_slot["support_object_ids"])
            record["steps"][-1]["corrective_pick_for_object"] = expected_id
            record["steps"][-1]["planned_slot_id"] = repair_slot_id
            repair_actions += 1
            if result.get("done") and not result.get("timed_out"):
              break
            # Corrective picks do not consume the current object's normal
            # three placement attempts. Re-observe before resuming the plan.
            _, _, _, state = scene()
            obs, info = state["obs"], state.get("info")
            slot = state["target_slots"][slot_id]
            continue

          current_stage = "search action execution"
          result = executor(
              action, object_id=object_id, support_id=slot_id,
              target_pose=slot["pose"],
              support_ids=slot["support_object_ids"])
          if ((result.get("done") and not result.get("timed_out")) or
              result.get("success")):
            break
          attempt += 1
          if attempt == 3:
            raise RuntimeError(
                "Object %s missed slot %s after 3 placement attempts." %
                (object_id, slot_id))
          _, _, _, state = scene()
          obs, info = state["obs"], state.get("info")
          slot = state["target_slots"][slot_id]
        if result.get("done") and not result.get("timed_out"):
          break
        _, _, _, state = scene()
        obs, info = state["obs"], state.get("info")
    else:
      mask_provider = RavensSimulationMaskProvider()
      target_provider = RavensSimulationTargetProvider()
      policy = RavensTransporterPolicy(
          policy.transporter, mask_provider, target_provider)
      for placement in placements:
        object_id = str(placement["object_id"])
        slot_id = placement["slot_id"]
        result = None
        for attempt in range(3):
          slot = state["target_slots"][slot_id]
          current_stage = "Transporter action selection"
          action = policy.make_action(
              object_id=object_id, support_id=slot_id, state=state)
          current_stage = "Transporter action execution"
          result = executor(
              action, object_id=object_id, support_id=slot_id,
              target_pose=slot["pose"],
              support_ids=slot["support_object_ids"])
          if ((result.get("done") and not result.get("timed_out")) or
              result.get("success")):
            break
          if attempt == 2:
            raise RuntimeError(
                "Object %s missed slot %s after 3 placement attempts." %
                (object_id, slot_id))
          _, _, _, state = scene()
        if result.get("done") and not result.get("timed_out"):
          break
        _, objects, _, state = scene()

  except Exception as error:  # Keep failed episodes in the aggregate.
    record["failure_stage"] = current_stage
    record["error"] = "%s: %s" % (type(error).__name__, error)
    traceback.print_exc()

  record["trials"] = executor.trials
  record["reward"] = float(np.clip(executor.reward, 0.0, 1.0))
  record["seconds"] = round(time.time() - started, 3)
  record["final_object_poses"] = _object_poses(env, executor.colors)
  record["final_task_progress"] = float(task.progress)
  record["final_task_reward_state"] = float(task._rewards)
  record["remaining_goal_matches"] = _goal_snapshot(task)
  if record["reward"] >= 0.99:
    record["outcome"] = "full_pyramid"
  elif record["reward"] > 0.0:
    record["outcome"] = "partial_pyramid"
  else:
    record["outcome"] = "zero_reward"
  return record


def summarize(records):
  rewards = np.asarray([row["reward"] for row in records], dtype=float)
  trials = np.asarray([row["trials"] for row in records], dtype=float)
  candidate_trials = np.asarray([
      row.get("candidate_evaluation_trials", row["trials"])
      for row in records], dtype=float)
  n = int(len(rewards))
  mean_reward = float(np.mean(rewards)) if n else 0.0
  if n > 1:
    margin = float(stats.t.ppf(0.975, n - 1) *
                   stats.sem(rewards))
  else:
    margin = 0.0
  return {
      "episodes": n,
      "average_trials": float(np.mean(trials)) if n else 0.0,
      "average_candidate_evaluation_trials": (
          float(np.mean(candidate_trials)) if n else 0.0),
      "mean_reward": mean_reward,
      "mean_reward_95_ci": [max(0.0, mean_reward - margin),
                             min(1.0, mean_reward + margin)],
      "full_pyramids": sum(row["outcome"] == "full_pyramid" for row in records),
      "partial_pyramids": sum(row["outcome"] == "partial_pyramid" for row in records),
      "zero_reward": sum(row["outcome"] == "zero_reward" for row in records),
      "errors": sum(row["error"] is not None for row in records),
  }


def main():
  args = parse_args()
  techniques = ["zero-shot", "beam", "cem"] if "all" in args.techniques else args.techniques
  seeds = dataset_seeds(args.seed_dataset, args.max_episodes, args.seed_offset)
  task = tasks.names[TASK_NAME]()
  task.mode = "test"
  env = Environment(args.assets_root, disp=args.disp, hz=480)
  try:
    policy = None
    if "zero-shot" in techniques:
      agent = OriginalTransporterAgent(args.checkpoint, TASK_NAME, args.models_root)
      agent.models_dir = os.path.join(args.models_root, "checkpoints", args.checkpoint)
      agent.load(args.checkpoint_step)
      policy = RavensTransporterPolicy(agent, None, None)

    planner = OllamaStackPlanner(
        model=args.model, temperature=args.temperature)
    plan_batches = {}
    for episode_index, seed in enumerate(seeds):
      print("VLM plans: episode %d/%d (seed %d)" % (
          episode_index + 1, len(seeds), seed), flush=True)
      plan_batches[episode_index] = sample_plan_candidates(
          env, task, seed, planner, args)

    summaries, all_records = {}, {}
    for technique in techniques:
      records = []
      for index, seed in enumerate(seeds, start=1):
        print("%s: episode %d/%d (seed %d)" %
              (technique, index, len(seeds), seed), flush=True)
        (sampled_plans, sampling_errors,
         sampling_seconds) = plan_batches[index - 1]
        record = run_episode(
            technique, env, task, seed, planner, policy, args,
            sampled_plans=sampled_plans,
            sampling_errors=sampling_errors,
            sampling_seconds=sampling_seconds)
        records.append(record)
        print("  reward=%.3f trials=%d outcome=%s%s" % (
            record["reward"], record["trials"], record["outcome"],
            (" error=" + record["error"]) if record["error"] else ""),
              flush=True)
        if record.get("evaluated_plan_samples", 1) > 1:
          print("  selected plan %d/%d score=%s" % (
              record["selected_plan_index"] + 1,
              record["effective_plan_samples"],
              record["plan_selection_score"]), flush=True)
      summaries[technique] = summarize(records)
      all_records[technique] = records

    output = {
        "benchmark": {
            "source_seed_dataset": os.path.abspath(args.seed_dataset),
            "source_episode_count": len(seeds),
            "evaluated_task": TASK_NAME,
            "seed_offset": args.seed_offset,
            "instruction": args.instruction,
            "vlm_model": planner.model,
            "vlm_temperature": args.temperature,
            "requested_plan_samples": args.num_plans,
            "plan_sampling_rule": (
                "one plan at temperature 0.0001; otherwise --num-plans "
                "plans sampled once per seed and reused across strategies; "
                "each plan is run from the same seed and ranked by simulator "
                "reward, matched goals, episode errors, timeouts, then trials"),
            "reward_definition": "Cumulative Ravens pyramid task reward, clipped to [0, 1].",
            "trial_definition": "One executed pick-and-place action.",
            "average_trials_definition": (
                "Actions in the selected plan run; average_candidate_"
                "evaluation_trials includes all sampled plan runs."),
            "confidence_interval": "Two-sided 95% Student t interval for mean episode reward.",
        },
        "summary": summaries,
        "episodes": all_records,
    }
    output_path = os.path.abspath(args.output)
    output_dir = os.path.dirname(output_path)
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
      json.dump(output, output_file, indent=2)
    diagnostic_files = write_diagnostic_files(
        all_records, args.diagnostics_dir)
    output["benchmark"]["diagnostic_files"] = diagnostic_files
    with open(output_path, "w", encoding="utf-8") as output_file:
      json.dump(output, output_file, indent=2)
    print("\nSummary")
    print(json.dumps(summaries, indent=2))
    print("Saved per-episode results to %s" % output_path)
    print("Saved diagnostic case files:")
    print(json.dumps(diagnostic_files, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
