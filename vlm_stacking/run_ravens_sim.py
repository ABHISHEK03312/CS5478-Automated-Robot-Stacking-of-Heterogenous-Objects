"""Run one VLM-guided stacking episode in Ravens' PyBullet simulator."""

import argparse
import json
import os

from ravens import tasks
from ravens.agents.transporter import OriginalTransporterAgent
from ravens.environments.environment import Environment

from vlm_stacking.pipeline import LanguageStackPipeline
from vlm_stacking.planner import OllamaStackPlanner
from ravens.agents.beam import BeamSearchAgent
from ravens.agents.cem import FreeStackCEMAgent
from vlm_stacking.ravens_policy import RavensTransporterPolicy
from vlm_stacking.ravens_sim import RavensSimulationExecutor
from vlm_stacking.ravens_sim import RavensSimulationMaskProvider
from vlm_stacking.ravens_sim import RavensSimulationSceneProvider
from vlm_stacking.ravens_sim import RavensSimulationTargetProvider


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--instruction", required=True)
  parser.add_argument("--assets-root", default="ravens/environments/assets")
  parser.add_argument("--models-root", default=None,
                      help="Checkpoint folder (required only for zero-shot baseline)")
  parser.add_argument("--checkpoint",
                      default="stack-block-pyramid-parallel-jaw-transporter-100-0")
  parser.add_argument("--checkpoint-step", type=int, default=40000)
  parser.add_argument("--seed", type=int, default=11)
  parser.add_argument("--disp", action="store_true",
                      help="Open the PyBullet viewer while running")
  parser.add_argument("--model", default=None,
                      help="Ollama model; defaults to OLLAMA_VLM_MODEL")
  parser.add_argument("--dry-run", action="store_true",
                      help="Ask for and validate a plan without moving the arm")
  parser.add_argument("--planner", choices=("zero-shot", "beam", "cem"),
                      default="zero-shot",
                      help="Use the Transporter baseline or Ravens beam/CEM agent")
  parser.add_argument("--beam-width", type=int, default=3)
  parser.add_argument("--beam-horizon", type=int, default=2)
  parser.add_argument("--beam-placements", type=int, default=6)
  parser.add_argument("--cem-samples", type=int, default=24)
  parser.add_argument("--cem-elites", type=int, default=5)
  parser.add_argument("--cem-iterations", type=int, default=3)
  parser.add_argument("--verify-k", type=int, default=3)
  parser.add_argument("--rollout", choices=("teleport", "full"),
                      default="teleport")
  args = parser.parse_args()
  if args.planner == "zero-shot" and not args.models_root and not args.dry_run:
    parser.error("--models-root is required for --planner zero-shot")
  if args.rollout == "full" and args.planner != "cem":
    parser.error("--rollout full is supported only with --planner cem")

  task_name = "stack-block-pyramid-parallel-jaw"
  task = tasks.names[task_name]()
  env = Environment(args.assets_root, task=task, disp=args.disp, hz=480)
  try:
    env.seed(args.seed)
    env.reset()

    scene_provider = RavensSimulationSceneProvider(env, task)
    vlm_planner = OllamaStackPlanner(model=args.model)
    if args.dry_run:
      image_bytes, objects, _ = scene_provider()
      plan = vlm_planner.plan(args.instruction, image_bytes, objects)
      print(json.dumps(plan, indent=2))
      return

    search_agent = None
    policy = None
    if args.planner == "beam":
      search_agent = BeamSearchAgent(
          env, task, objective="free", beam_width=args.beam_width,
          horizon=args.beam_horizon, n_place=args.beam_placements,
          verify_k=args.verify_k, rollout="teleport", seed=args.seed)
    elif args.planner == "cem":
      search_agent = FreeStackCEMAgent(
          env, task, n_samples=args.cem_samples, n_elites=args.cem_elites,
          n_iters=args.cem_iterations, verify_k=args.verify_k,
          rollout=args.rollout, seed=args.seed)
    else:
      # Zero-shot baseline keeps the original learned Transporter action path.
      import pybullet as p
      agent = OriginalTransporterAgent(
          args.checkpoint, task_name, args.models_root)
      agent.models_dir = os.path.join(args.models_root, "checkpoints",
                                      args.checkpoint)
      agent.load(args.checkpoint_step)
      base_pose = p.getBasePositionAndOrientation(env.obj_ids["fixed"][0])
      target_provider = RavensSimulationTargetProvider(
          anchor_xy=base_pose[0][:2])
      policy = RavensTransporterPolicy(
          agent, RavensSimulationMaskProvider(), target_provider)
    pipeline = LanguageStackPipeline(
        vlm_planner, scene_provider, policy, RavensSimulationExecutor(env))
    result = pipeline.run(args.instruction, search_agent=search_agent)
    print(json.dumps(result, indent=2, default=str))
  finally:
    env.close()


if __name__ == "__main__":
  main()
