# Language-guided stacking (Ravens integration)

This folder adds a high-level VLM planning layer without modifying Ravens' core
agent, demos, datasets, or checkpoints. It sends a language instruction, a
scene image, and measured object metadata to a vision-capable model. The model
returns stable object IDs ordered **bottom to top**. The pipeline checks that
the IDs match the detected scene objects and rejects obvious footprint support
violations before asking the Transporter model for a masked pick and a place
near the planned support.

## What is present in the supplied assets

The adjacent `ravens_data` directory contains Towers of Hanoi demonstration
episodes. `ravens_models/checkpoints` includes Hanoi, block-pyramid, and other
task-specific Transporter checkpoints. These are not general heterogeneous
object grasp models. The adapter here uses the existing Transporter API, but a
task-specific checkpoint should only be used on scenes similar to its training
task. For arbitrary heterogeneous shapes, collect demonstrations and train a
matching grasp/place policy.

The current Ravens simulator exposes object IDs, dimensions, and poses through
`info`; this is a **simulation-only privileged-state bridge**. For a real arm,
provide a scene provider backed by RGB-D segmentation and calibrated
localization. A VLM label or bounding box alone is not a metric grasp pose.

## Files

- `planner.py`: Ollama vision request, JSON parsing, and plan validation.
- `pipeline.py`: re-observe / plan / execute loop with explicit adapters.
- `ravens_policy.py`: applies object masks to Transporter pick scores and
  constrains placement to the geometry chosen by the high-level plan.

## Configure the VLM

Start Ollama locally and pull a vision-language model, for example
`ollama pull qwen2.5vl:7b`. The planner defaults to `qwen2.5vl:7b`; set
`OLLAMA_VLM_MODEL` or pass `--model` to select a different local model. The
scene image and measured object catalog are sent to the local Ollama service.
If the planner's plan validator rejects the ordering, it retries generation
three times, sending the rejected plan and validator error with each repair
request. The initial plan request is additional to those three retries.

## Run the included simulator example

From the repository root, start the Ollama service and use the Ravens
environment. The example uses the available block-pyramid Transporter
checkpoint and creates the matching Ravens task scene:

```bash
export OLLAMA_VLM_MODEL="qwen2.5vl:7b"
python -m vlm_stacking.run_ravens_sim \
  --models-root "../ravens_models" \
  --instruction "Build a stable tower with the largest block at the bottom."
```

Add `--dry-run` to request and print a validated plan without executing robot
actions. The supplied `ravens_data` currently contains Hanoi demonstrations,
while this runnable example uses the matching block-pyramid scene and
checkpoint. The demo dataset is not needed for inference; it is used when
training or evaluating a learned Ravens policy.

## Connect to your environment

Construct `LanguageStackPipeline` with four project-specific adapters:

1. `scene_provider()` returns `(image_bytes, objects, state)`. Each object must
   include a stable `object_id` and measured `dimensions` in meters. Keep the
   ID mapping from segmentation through execution.
2. `mask_provider(object_id, state)` returns a boolean mask aligned to the
   agent's 320x160 fused heightmap. In simulation, derive this from Ravens'
   segmentation image; on hardware, use your segmentation and depth pipeline.
3. `target_provider(support_id, state)` returns a world `(x, y)` for the table
   anchor or the preceding object's top support. Recompute support height from
   depth after each action.
4. `executor(action)` runs one Ravens `env.step(action)` or the equivalent robot
   skill and returns a result dictionary.

Example wiring (after creating the adapters and loading a checkpoint):

```python
from vlm_stacking import LanguageStackPipeline, OllamaStackPlanner
from vlm_stacking.ravens_policy import RavensTransporterPolicy

policy = RavensTransporterPolicy(
    transporter, mask_provider, target_provider)
pipeline = LanguageStackPipeline(
    OllamaStackPlanner(), scene_provider, policy, executor)
result = pipeline.run("Stack the largest object at the bottom and the smallest on top.")
```

## Limits to account for

- The plan validator checks object IDs and horizontal footprint fit. It does
  not prove stability, infer mass/center of gravity, detect collisions, or
  replace a motion planner.
- Every planned object must be individually visible and pickable at its turn.
  Stacked source piles need a task-specific staging/rearrangement planner.
- The Transporter checkpoint still determines the grasp pixel and local
  orientation. The VLM does not output motor commands.
- The supplied data and checkpoints do not include an arbitrary-object stack
  task or a hardware camera/gripper driver. This folder provides the planner
  and policy bridge; implement the scene and robot adapters for your hardware.
