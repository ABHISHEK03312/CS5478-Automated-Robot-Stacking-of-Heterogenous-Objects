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
from ravens.agents.cem import FreeStackCEMAgent
from ravens.agents.transporter import OriginalTransporterAgent
from ravens.dataset import Dataset
from ravens.environments.environment import Environment
from vlm_stacking.planner import OllamaStackPlanner
from vlm_stacking.ravens_policy import RavensTransporterPolicy
from vlm_stacking.ravens_sim import RavensSimulationExecutor
from vlm_stacking.ravens_sim import RavensSimulationMaskProvider
from vlm_stacking.ravens_sim import RavensSimulationSceneProvider
from vlm_stacking.ravens_sim import RavensSimulationTargetProvider

TASK_NAME = "stack-block-pyramid-parallel-jaw"


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--seed-dataset", default="../ravens_data/towers-of-hanoi/towers-of-hanoi-test",
      help="Hanoi test dataset folder; its episode seeds define pyramid scenes")
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
        env, task, objective="free", beam_width=args.beam_width,
        horizon=args.beam_horizon, n_place=args.beam_placements,
        verify_k=args.verify_k, rollout="teleport", seed=args.seed + seed)
  if technique == "cem":
    return FreeStackCEMAgent(
        env, task, n_samples=args.cem_samples, n_elites=args.cem_elites,
        n_iters=args.cem_iterations, verify_k=args.verify_k,
        rollout=args.rollout, seed=args.seed + seed)
  return None


class CountingExecutor:
  """Count executed robot trials and retain environment rewards."""

  def __init__(self, env):
    self.executor = RavensSimulationExecutor(env)
    self.trials = 0
    self.reward = 0.0

  def __call__(self, action, object_id=None, support_id=None):
    result = self.executor(action, object_id=object_id, support_id=support_id)
    self.trials += 1
    self.reward += result["reward"]
    return result


def run_episode(technique, env, task, seed, planner, policy, args):
  np.random.seed(seed)
  env.seed(seed)
  env.set_task(task)
  obs = env.reset()
  scene = RavensSimulationSceneProvider(
      env, task, debug_image_path=None)
  executor = CountingExecutor(env)
  record = {"seed": int(seed), "technique": technique, "trials": 0,
            "reward": 0.0, "plan": None, "error": None}
  started = time.time()

  try:
    image, objects, state = scene()
    plan = planner.plan(args.instruction, image, objects)
    record["plan"] = plan
    order = [str(item) for item in plan["stack_order"]]

    search_agent = make_search_agent(technique, env, task, args, seed)
    if search_agent is not None:
      search_agent.set_preferred_order(order)
      info = None
      for expected_id in order:
        action = search_agent.act(obs, info)
        object_id = getattr(search_agent, "last_object_id", None)
        if action is None or object_id is None:
          raise RuntimeError("Search agent stopped before its next planned pick.")
        if str(object_id) != expected_id:
          raise RuntimeError("Search agent selected %s; plan requires %s." %
                             (object_id, expected_id))
        result = executor(action, object_id=object_id)
        obs, info = state["obs"], state.get("info")
        if result.get("done"):
          break
        _, _, state = scene()
        obs = state["obs"]
        info = state.get("info")
        if result.get("success") is False:
          raise RuntimeError("The selected object did not move: %s" % object_id)
    else:
      previous_id = None
      mask_provider = RavensSimulationMaskProvider()
      anchor = env.obj_ids["fixed"][0]
      import pybullet as p
      base_position = p.getBasePositionAndOrientation(anchor)[0]
      target_provider = RavensSimulationTargetProvider(
          anchor_xy=base_position[:2])
      policy = RavensTransporterPolicy(
          policy.transporter, mask_provider, target_provider)
      for object_id in order:
        action = policy.make_action(
            object_id=object_id, support_id=previous_id, state=state)
        result = executor(action, object_id=object_id,
                          support_id=previous_id)
        previous_id = object_id
        if result.get("done"):
          break
        if result.get("success") is False:
          raise RuntimeError("The selected object did not move: %s" % object_id)
        _, objects, state = scene()

  except Exception as error:  # Keep failed episodes in the aggregate.
    record["error"] = "%s: %s" % (type(error).__name__, error)
    traceback.print_exc()

  record["trials"] = executor.trials
  record["reward"] = float(np.clip(executor.reward, 0.0, 1.0))
  record["seconds"] = round(time.time() - started, 3)
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
    print("\nSummary")
    print(json.dumps(summaries, indent=2))
    print("Saved per-episode results to %s" % output_path)
  finally:
    env.close()


if __name__ == "__main__":
  main()
