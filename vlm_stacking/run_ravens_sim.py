"""Run one VLM-guided stacking episode in Ravens' PyBullet simulator."""

import argparse
import json
import os

from ravens import tasks
from ravens.agents.transporter import OriginalTransporterAgent
from ravens.environments.environment import Environment

from vlm_stacking.pipeline import LanguageStackPipeline
from vlm_stacking.planner import OllamaStackPlanner
from vlm_stacking.search_planner import HybridStackPlanner
from vlm_stacking.cem_planner import CEMStackPlanner
from vlm_stacking.ravens_sim import RavensSimulationStackEvaluator
from vlm_stacking.ravens_policy import RavensTransporterPolicy
from vlm_stacking.ravens_sim import RavensSimulationExecutor
from vlm_stacking.ravens_sim import RavensSimulationMaskProvider
from vlm_stacking.ravens_sim import RavensSimulationSceneProvider
from vlm_stacking.ravens_sim import RavensSimulationTargetProvider


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--instruction", required=True)
  parser.add_argument("--assets-root", default="ravens/environments/assets")
  parser.add_argument("--models-root", required=True,
                      help="Folder containing the Ravens checkpoints directory")
  parser.add_argument("--checkpoint",
                      default="stack-block-pyramid-transporter-1000-0")
  parser.add_argument("--checkpoint-step", type=int, default=20000)
  parser.add_argument("--seed", type=int, default=11)
  parser.add_argument("--disp", action="store_true",
                      help="Open the PyBullet viewer while running")
  parser.add_argument("--model", default=None,
                      help="Ollama model; defaults to OLLAMA_VLM_MODEL")
  parser.add_argument("--dry-run", action="store_true",
                      help="Ask for and validate a plan without moving the arm")
  parser.add_argument("--planner", choices=("zero-shot", "hybrid", "cem"),
                      default="zero-shot",
                      help="Choose the VLM-only baseline, beam search, or CEM")
  parser.add_argument("--beam-width", type=int, default=12,
                      help="Number of partial orders retained at each beam depth")
  parser.add_argument("--cem-population", type=int, default=12)
  parser.add_argument("--cem-iterations", type=int, default=3)
  parser.add_argument("--cem-elite-fraction", type=float, default=0.25)
  parser.add_argument("--cem-smoothing", type=float, default=0.7)
  parser.add_argument("--cem-seed", type=int, default=0)
  args = parser.parse_args()

  task = tasks.names["stack-block-pyramid"]()
  env = Environment(args.assets_root, task=task, disp=args.disp, hz=480)
  try:
    env.seed(args.seed)
    env.reset()

    scene_provider = RavensSimulationSceneProvider(env, task)
    vlm_planner = OllamaStackPlanner(model=args.model)
    planner = vlm_planner
    if args.planner == "hybrid":
      planner = HybridStackPlanner(
          vlm_planner,
          evaluator=RavensSimulationStackEvaluator(env),
          beam_width=args.beam_width)
    elif args.planner == "cem":
      planner = CEMStackPlanner(
          vlm_planner,
          evaluator=RavensSimulationStackEvaluator(env),
          population_size=args.cem_population,
          iterations=args.cem_iterations,
          elite_fraction=args.cem_elite_fraction,
          smoothing=args.cem_smoothing,
          seed=args.cem_seed)
    if args.dry_run:
      image_bytes, objects, _ = scene_provider()
      plan = planner.plan(args.instruction, image_bytes, objects)
      print(json.dumps(plan, indent=2))
      return

    # pybullet env
    import pybullet as p
    agent = OriginalTransporterAgent(
        args.checkpoint, "stack-block-pyramid", args.models_root)
    agent.models_dir = os.path.join(args.models_root, "checkpoints",
                                    args.checkpoint)
    agent.load(args.checkpoint_step)
    # Use the known fixed base as the geometric anchor for this simulator task.
    base_pose = p.getBasePositionAndOrientation(env.obj_ids["fixed"][0])
    target_provider = RavensSimulationTargetProvider(
        anchor_xy=base_pose[0][:2])
    policy = RavensTransporterPolicy(
        agent, RavensSimulationMaskProvider(), target_provider)
    pipeline = LanguageStackPipeline(
        planner, scene_provider, policy, RavensSimulationExecutor(env))
    result = pipeline.run(args.instruction)
    print(json.dumps(result, indent=2, default=str))
  finally:
    env.close()


if __name__ == "__main__":
  main()
