"""Evaluate VLM stacking strategies on seed-matched Ravens test scenes.

The Hanoi dataset is used only to obtain its held-out episode seeds. Each seed
initializes the parallel-jaw block-pyramid task so all three strategies see
identical, reproducible pyramid scenes. Hanoi images and actions are not used.
"""

import argparse
import json
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
  return parser.parse_args()


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


def run_episode(technique, env, task, seed, planner, policy, args):
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
    current_stage = "VLM planning"
    slots = target_slot_catalog(task, objects_for_prompt)
    plan = planner.plan(args.instruction, image, objects_for_prompt, slots)
    record["plan"] = plan
    placements = plan["placements"]

    search_agent = make_search_agent(technique, env, task, args, seed)
    if search_agent is not None:
      search_agent.set_preferred_placements(placements, state["target_slots"])
      info = None
      for placement in placements:
        expected_id = str(placement["object_id"])
        slot_id = placement["slot_id"]
        slot = state["target_slots"][slot_id]
        result = None
        for attempt in range(3):
          if is_object_at_target(task, expected_id, slot["pose"]):
            result = {"success": True, "already_placed": True,
                      "done": False}
            break
          current_stage = "search action selection"
          action = search_agent.act(obs, info)
          object_id = getattr(search_agent, "last_object_id", None)
          if action is None or object_id is None:
            raise RuntimeError("Search agent stopped before its next planned pick.")
          if str(object_id) != expected_id:
            if is_object_at_target(task, expected_id, slot["pose"]):
              result = {"success": True, "already_placed": True,
                        "done": False}
              break
            raise RuntimeError("Search agent selected %s; plan requires %s." %
                               (object_id, expected_id))
          current_stage = "search action execution"
          result = executor(
              action, object_id=object_id, support_id=slot_id,
              target_pose=slot["pose"],
              support_ids=slot["support_object_ids"])
          if result.get("done") or result.get("success"):
            break
          if attempt == 2:
            raise RuntimeError(
                "Object %s missed slot %s after 3 placement attempts." %
                (object_id, slot_id))
          _, _, _, state = scene()
          obs, info = state["obs"], state.get("info")
          slot = state["target_slots"][slot_id]
        if result.get("done"):
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
          if result.get("done") or result.get("success"):
            break
          if attempt == 2:
            raise RuntimeError(
                "Object %s missed slot %s after 3 placement attempts." %
                (object_id, slot_id))
          _, _, _, state = scene()
        if result.get("done"):
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

    summaries, all_records = {}, {}
    for technique in techniques:
      planner = OllamaStackPlanner(model=args.model)
      records = []
      for index, seed in enumerate(seeds, start=1):
        print("%s: episode %d/%d (seed %d)" %
              (technique, index, len(seeds), seed), flush=True)
        record = run_episode(
            technique, env, task, seed, planner, policy, args)
        records.append(record)
        print("  reward=%.3f trials=%d outcome=%s%s" % (
            record["reward"], record["trials"], record["outcome"],
            (" error=" + record["error"]) if record["error"] else ""),
              flush=True)
      summaries[technique] = summarize(records)
      all_records[technique] = records

    output = {
        "benchmark": {
            "source_seed_dataset": os.path.abspath(args.seed_dataset),
            "source_episode_count": len(seeds),
            "evaluated_task": TASK_NAME,
            "seed_offset": args.seed_offset,
            "instruction": args.instruction,
            "reward_definition": "Cumulative Ravens pyramid task reward, clipped to [0, 1].",
            "trial_definition": "One executed pick-and-place action.",
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
