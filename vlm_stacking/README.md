# Language-guided stacking (Ravens integration)

This folder adds a high-level VLM planning layer and adapters for Ravens'
existing simulator search agents. It does not modify demos, datasets, or
checkpoints. It sends a language instruction, a scene image, measured object
metadata, and named target slots to a vision-capable model. The model returns
an object-to-slot assignment in build order. Slot poses come from the task's
goal specification; the VLM does not invent robot coordinates. The planner
validates object and slot coverage, layer ordering, and the task's allowed
colors for each layer before execution.

## What is present in the supplied assets

The repository's `data/parallel-jaw-v2` directory contains parallel-jaw
block-pyramid train and test demonstrations, and `models/parallel-jaw-v2`
contains the matching Transporter checkpoint. These are not general
heterogeneous object grasp models. The adapter here uses the existing
Transporter API, but a task-specific checkpoint should only be used on scenes
similar to its training task. For arbitrary heterogeneous shapes, collect
demonstrations and train a matching grasp/place policy.

The current Ravens simulator exposes object IDs, dimensions, and poses through
`info`; this is a **simulation-only privileged-state bridge**. For a real arm,
provide a scene provider backed by RGB-D segmentation and calibrated
localization. A VLM label or bounding box alone is not a metric grasp pose.

## Files

- `planner.py`: Ollama vision request, JSON parsing, and plan validation.
- `pipeline.py`: re-observe / plan / execute loop with explicit adapters.
- `ravens_policy.py`: applies object masks to Transporter pick scores and
  constrains placement to the assigned target slot.

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
  --instruction "Build a stable pyramid following the block colors."
```

Add `--dry-run` to request and print a validated plan without executing robot
actions. The supplied `ravens_data` currently contains Hanoi demonstrations,
while this runnable example uses the matching block-pyramid scene and
checkpoint. The demo dataset is not needed for inference; it is used when
training or evaluating a learned Ravens policy.

## Connect to your environment

Construct `LanguageStackPipeline` with four project-specific adapters:

1. `scene_provider()` returns the image, full object metadata, the simplified
   VLM catalog, and execution state. Each object must include a stable
   `object_id`, readable color, and measured dimensions. Provide a target slot
   catalog plus a slot-to-pose mapping as well.
2. `mask_provider(object_id, state)` returns a boolean mask aligned to the
   agent's 320x160 fused heightmap. In simulation, derive this from Ravens'
   segmentation image; on hardware, use your segmentation and depth pipeline.
3. `target_provider(slot_id, state)` maps the planned slot to a world `(x, y)`.
   The robot motion planner must determine a collision-free approach and safe
   release height for that slot.
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
result = pipeline.run("Build a stable pyramid following the block colors.")
```

## Limits to account for

- The plan validator checks object and slot coverage, layer ordering, and
  allowed colors. It does not prove stability, infer mass/center of gravity,
  detect collisions, or replace a motion planner.
- Every planned object must be individually visible and pickable at its turn.
  Stacked source piles need a task-specific staging/rearrangement planner.
- The Transporter checkpoint still determines the grasp pixel and local
  orientation. The VLM does not output motor commands.
- The supplied data and checkpoints do not include an arbitrary-object stack
  task or a hardware camera/gripper driver. This folder provides the planner
  and policy bridge; implement the scene and robot adapters for your hardware.


## Compare the zero-shot baseline and hybrid planner

The default `--planner zero-shot` keeps VLM slot assignment and the Transporter
policy as the baseline. Its three retries apply when model output cannot be
parsed or when a parsed plan fails validation, including missing or extra
object IDs, repeated slots, invalid colors, or invalid layer order. Each repair
prompt includes the previous response or plan and the validation error. These
retries are still zero-shot inference and do not train or update the model.

The optional `beam` and `cem` modes import the concrete Ravens agents from
`ravens/agents/beam.py` and `ravens/agents/cem.py`. The VLM still produces the
slot assignments. Beam search explores placements around each assigned target,
while CEM optimizes continuous `(x, y, yaw)` parameters around that target.
Both use the agents' PyBullet snapshot, rollout, and restore
logic. Beam search uses fast teleported rollouts and arm-checks the top
`--verify-k` candidates. CEM defaults to teleported rollouts with top-candidate
arm checks; `--rollout full` runs every CEM candidate through the arm and is
slower. These modes do not load a Transporter checkpoint.

```bash
python -m vlm_stacking.run_ravens_sim \
  --planner beam --beam-width 3 --beam-horizon 2 \
  --instruction "Build a stable pyramid following the block colors."

python -m vlm_stacking.run_ravens_sim \
  --planner cem --cem-samples 24 --cem-elites 5 --cem-iterations 3 \
  --instruction "Build a stable pyramid following the block colors."
```

`--dry-run` prints the validated object-to-slot assignments without running the search
agent. The included Ravens task uses identical cubes and a stand; evaluate
heterogeneous shapes with a matching task and scene adapter before interpreting
search results as general heterogeneous stacking performance.

## Evaluate all three stacking strategies

The repository's parallel-jaw block-pyramid test set supplies 100 held-out
episode seeds. The evaluation script reads it from `data/parallel-jaw-v2` by
default and reuses those seeds to initialize the matching task, so zero-shot,
beam, and CEM are compared on the same reproducible pyramid scenes. The
default dataset path is resolved from the script location, so it does not
depend on the current working directory. Pass `--seed-dataset` to use another
Ravens dataset.

Run all techniques from the repository root with Ollama available and the
included parallel-jaw checkpoint present:

```bash
python -m vlm_stacking.evaluate \
  --models-root ./models/parallel-jaw-v2 \
  --output ./vlm_stacking_eval.json
```

Use `--num-test-cases 5` for a short run. The default value `0` runs the full test set; `--max-episodes` remains an alias. The report includes per-episode reward,
pick-and-place trial count, errors, and a summary for each technique. A full
pyramid has reward at least 0.99, a partial pyramid has reward above zero and
below 0.99, and zero reward is reported separately. The 95% confidence interval
uses the Student t distribution. Beam and CEM defaults match the simulator
runner; tune them only on a separate training seed set.

The evaluator also writes `partial_towers.json`, `error_cases.json`, and
`zero_reward_cases.json` under `--diagnostics-dir` (default:
`./vlm_stacking_diagnostics`). Each case includes the VLM plan and object/color
catalog, initial and final block poses, action pick/place poses, per-step reward,
and per-goal target-match details. A case with an error can also appear in the
partial or zero-reward file if it accumulated reward before failing.

## Results
### Simple block stacks
| Method | Episodes | Avg. Trials | Mean Reward | 95% CI | Full Pyramids | Partial Pyramids | Zero Reward | Errors |
|-----------|:--------:|:-----------:|:-----------:|:------------------:|:-------------:|:----------------:|:-----------:|:------:|
| Zero-shot VLM | 100 | 5.24 | 0.745 | [0.668, 0.822] | 66 | 20 | 14 | 0 |
| VLM + Beam | 100 | 5.79 | 0.850 | [0.784, 0.916] | 81 | 9 | 10 | 19 |
| VLM + CEM | 100 | 5.82 | 0.852 | [0.788, 0.915] | 80 | 12 | 8 | 20 |