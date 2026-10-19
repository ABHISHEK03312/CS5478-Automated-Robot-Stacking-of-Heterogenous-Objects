# Language-guided stacking (Ravens integration)

This folder adds a high-level VLM planning layer and adapters for Ravens'
existing simulator search agents. It does not modify demos, datasets, or
checkpoints. It sends a language instruction, a
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
If the model output cannot be parsed as JSON or a parsed plan fails
validation, it retries generation up to three times and includes the previous
response or plan and the parse or validation error in the repair prompt. The
initial plan request is additional to those retries. The planner remains
zero-shot; it does not train or update the model.

## Run the included simulator example

From the repository root, start the Ollama service and use the Ravens
environment. The example uses the included parallel-jaw block-pyramid Transporter
checkpoint and creates the matching parallel-jaw Ravens task scene:

```bash
export OLLAMA_VLM_MODEL="qwen2.5vl:7b"
python -m vlm_stacking.run_ravens_sim \
  --models-root "./models/parallel-jaw-v2" \
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


## Compare the zero-shot baseline and hybrid planner

The default `--planner zero-shot` keeps the VLM-only order and Transporter
policy as the baseline. Its three retries apply when model output cannot be
parsed or when a parsed plan fails validation, including missing or extra
object IDs and unsafe footprint order. Each repair prompt includes the previous
response or plan and the error. These retries are still zero-shot inference and
do not train or update the model.

The optional `beam` and `cem` modes import the concrete Ravens agents from
`ravens/agents/beam.py` and `ravens/agents/cem.py`. The VLM still produces the
object order. Beam search then explores placement sequences for the next VLM
object, while CEM optimizes continuous `(x, y, yaw)` placement parameters for
the next object. Both use the agents' PyBullet snapshot, rollout, and restore
logic. Beam search uses fast teleported rollouts and arm-checks the top
`--verify-k` candidates. CEM defaults to teleported rollouts with top-candidate
arm checks; `--rollout full` runs every CEM candidate through the arm and is
slower. These modes do not load a Transporter checkpoint.

```bash
python -m vlm_stacking.run_ravens_sim \
  --planner beam --beam-width 3 --beam-horizon 2 \
  --instruction "Build a stable tower with the largest block at the bottom."

python -m vlm_stacking.run_ravens_sim \
  --planner cem --cem-samples 24 --cem-elites 5 --cem-iterations 3 \
  --instruction "Build a stable tower with the largest block at the bottom."
```

`--dry-run` prints just the validated VLM order without running the search
agent. The included Ravens task uses identical cubes and a stand; evaluate
heterogeneous shapes with a matching task and scene adapter before interpreting
search results as general heterogeneous stacking performance.

## Evaluate all three stacking strategies

The Ravens Hanoi test set supplies 100 held-out episode seeds. The evaluation
script reuses those seeds to initialize the parallel-jaw block-pyramid task,
so zero-shot, beam, and CEM are compared on the same reproducible pyramid
scenes. Hanoi RGB-D frames and actions are not used; the Hanoi task itself is
not compatible with these stacking planners.

Run all techniques from the repository root with Ollama available and the
included parallel-jaw checkpoint present:

```bash
python -m vlm_stacking.evaluate \
  --seed-dataset ../ravens_data/towers-of-hanoi/towers-of-hanoi-test \
  --models-root ./models/parallel-jaw-v2 \
  --output ./vlm_stacking_eval.json
```

Use `--num-test-cases 5` for a short run. The default value `0` runs the full test set; `--max-episodes` remains an alias. The report includes per-episode reward,
pick-and-place trial count, errors, and a summary for each technique. A full
pyramid has reward at least 0.99, a partial pyramid has reward above zero and
below 0.99, and zero reward is reported separately. The 95% confidence interval
uses the Student t distribution. Beam and CEM defaults match the simulator
runner; tune them only on a separate training seed set.
