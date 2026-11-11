"""Select a reproducible VLM stacking run, then replay it in the PyBullet GUI.

The requested seed and VLM configuration are evaluated in headless PyBullet
first. If at least one sampled plan completes the stack, the highest-scoring
plan is replayed once in a visible GUI environment for screen recording.
"""

import argparse
import json
import os
import time

from ravens import tasks
from ravens.agents.transporter import OriginalTransporterAgent
from ravens.environments.environment import Environment

from vlm_stacking import evaluate
from vlm_stacking.planner import OllamaStackPlanner
from vlm_stacking.ravens_policy import RavensTransporterPolicy


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK_NAME = "stack-block-pyramid-parallel-jaw"
DEFAULT_ASSETS_ROOT = os.path.join(REPO_ROOT, "ravens", "environments", "assets")
DEFAULT_MODELS_ROOT = os.path.join(REPO_ROOT, "models", "parallel-jaw-v2")


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seed", type=int, required=True,
                      help="Exact PyBullet scene seed to demonstrate")
  parser.add_argument("--technique", choices=("zero-shot", "beam", "cem"),
                      default="beam",
                      help="VLM plan execution configuration (default: beam)")
  parser.add_argument("--instruction", default="Build a stable pyramid using all the blocks.")
  parser.add_argument("--model", default=None,
                      help="Ollama VLM model; defaults to OLLAMA_VLM_MODEL")
  parser.add_argument("--temperature", type=float, default=0.0001,
                      help="Ollama sampling temperature (default: 0.0001)")
  parser.add_argument("--num-plans", type=int, default=1,
                      help="Plans to sample/search when temperature != 0.0001")
  parser.add_argument("--assets-root", default=DEFAULT_ASSETS_ROOT)
  parser.add_argument("--models-root", default=DEFAULT_MODELS_ROOT,
                      help="Transporter checkpoint directory for zero-shot mode")
  parser.add_argument("--checkpoint",
                      default="stack-block-pyramid-parallel-jaw-transporter-100-0")
  parser.add_argument("--checkpoint-step", type=int, default=40000)
  parser.add_argument("--search-seed", type=int, default=11,
                      help="Random seed for Beam/CEM sampling")
  parser.add_argument("--beam-width", type=int, default=3)
  parser.add_argument("--beam-horizon", type=int, default=2)
  parser.add_argument("--beam-placements", type=int, default=6)
  parser.add_argument("--cem-samples", type=int, default=24)
  parser.add_argument("--cem-elites", type=int, default=5)
  parser.add_argument("--cem-iterations", type=int, default=3)
  parser.add_argument("--verify-k", type=int, default=3)
  parser.add_argument("--rollout", choices=("teleport", "full"), default="teleport")
  parser.add_argument("--allow-partial", action="store_true",
                      help="Replay the best plan even if headless reward is below 0.99")
  parser.add_argument("--no-pause", action="store_true",
                      help="Close the GUI immediately after replay finishes")
  parser.add_argument("--output", default=None,
                      help="JSON result path (default: vlm_demo_seed_<seed>.json)")
  args = parser.parse_args()
  if args.temperature < 0:
    parser.error("--temperature must be non-negative")
  if args.num_plans < 1:
    parser.error("--num-plans must be at least 1")
  if args.rollout == "full" and args.technique != "cem":
    parser.error("--rollout full is supported only with --technique cem")
  return args


def make_eval_args(args):
  """Adapt the demo CLI to the evaluator's shared candidate-run interface."""
  return argparse.Namespace(
      instruction=args.instruction,
      temperature=args.temperature,
      num_plans=args.num_plans,
      seed=args.search_seed,
      beam_width=args.beam_width,
      beam_horizon=args.beam_horizon,
      beam_placements=args.beam_placements,
      cem_samples=args.cem_samples,
      cem_elites=args.cem_elites,
      cem_iterations=args.cem_iterations,
      verify_k=args.verify_k,
      rollout=args.rollout,
  )


def load_policy(args):
  if args.technique != "zero-shot":
    return None
  agent = OriginalTransporterAgent(
      args.checkpoint, TASK_NAME, args.models_root)
  agent.models_dir = os.path.join(args.models_root, "checkpoints", args.checkpoint)
  agent.load(args.checkpoint_step)
  return RavensTransporterPolicy(agent, None, None)


def main():
  args = parse_args()
  eval_args = make_eval_args(args)
  task = tasks.names[TASK_NAME]()
  task.mode = "test"
  planner = OllamaStackPlanner(model=args.model, temperature=args.temperature)
  policy = load_policy(args)

  # Search/evaluate all sampled plans without showing intermediate candidates.
  headless_env = Environment(args.assets_root, disp=False, hz=480)
  try:
    plans, sampling_errors, sampling_seconds = evaluate.sample_plan_candidates(
        headless_env, task, args.seed, planner, eval_args)
    if not plans:
      raise RuntimeError("No valid VLM plans were produced: %s" % sampling_errors)
    selected = evaluate.run_episode(
        args.technique, headless_env, task, args.seed, planner, policy,
        eval_args, sampled_plans=plans, sampling_errors=sampling_errors,
        sampling_seconds=sampling_seconds)
  finally:
    headless_env.close()

  print("Headless candidate selection:")
  print(json.dumps({
      "seed": args.seed,
      "technique": args.technique,
      "model": planner.model,
      "temperature": args.temperature,
      "requested_plans": args.num_plans,
      "evaluated_plans": selected.get("evaluated_plan_samples", 0),
      "selected_plan_index": selected.get("selected_plan_index"),
      "reward": selected.get("reward"),
      "outcome": selected.get("outcome"),
      "trials": selected.get("trials"),
      "error": selected.get("error"),
  }, indent=2))

  if selected.get("reward", 0.0) < 0.99 and not args.allow_partial:
    raise SystemExit(
        "No full stack was found for this seed/configuration. Choose a seed "
        "that succeeds in the evaluation results, increase --num-plans, or "
        "pass --allow-partial to replay the best partial result.")

  # Reuse the selected plan; do not make another VLM request for the recording.
  gui_task = tasks.names[TASK_NAME]()
  gui_task.mode = "test"
  gui_env = Environment(args.assets_root, task=gui_task, disp=True, hz=480)
  try:
    replay = evaluate._run_episode_with_plan(
        args.technique, gui_env, gui_task, args.seed, planner, policy,
        eval_args, selected["plan"], selected.get("selected_plan_index", 0))
    replay["demo_config"] = {
        "seed": args.seed,
        "technique": args.technique,
        "vlm_model": planner.model,
        "temperature": args.temperature,
        "requested_plans": args.num_plans,
        "selected_plan_index": selected.get("selected_plan_index"),
        "selection_reward": selected.get("reward"),
        "selection_outcome": selected.get("outcome"),
        "search_seed": args.search_seed,
        "instruction": args.instruction,
    }
    print("\nVisible replay result:")
    print(json.dumps(replay, indent=2, default=str))
    output_path = args.output or "vlm_demo_seed_%d.json" % args.seed
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
      json.dump(replay, output_file, indent=2, default=str)
    print("\nSaved demo record to %s" % output_path)
    if not args.no_pause:
      input("Demo complete. Leave the final scene visible; press Enter to close PyBullet. ")
  finally:
    gui_env.close()


if __name__ == "__main__":
  main()
