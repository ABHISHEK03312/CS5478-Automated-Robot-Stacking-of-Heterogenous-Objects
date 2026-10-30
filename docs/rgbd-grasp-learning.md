# RGB-D Grasp Learning

This experiment learns a dense planar grasp value function from executed
parallel-jaw attempts. It predicts grasp quality over image position, yaw, and
jaw width:

```text
Q[row, column, rotation_bin, width_bin]
```

Unlike the heterogeneous stacking demonstrations, these labels record both
successful and failed physical attempts in the simulator. Simulator poses are
used only to sample a balanced mix of candidates; the model receives the same
fused RGB-D heightmap representation as Transporter Networks.

## Collect attempts

Run from the repository root:

```powershell
python -m ravens.collect_grasp_data `
  --assets_root=./ravens/environments/assets/ `
  --task=stack-block-pyramid-parallel-jaw `
  --attempts=1000 `
  --output_dir=./data/rgbd-grasp-attempts
```

Each compressed sample contains the fused heightmap, grasp pixel, yaw, desired
jaw width, observed lift success, timeout state, grasped object ID, measured
lift height, and whether the grasp remained secured during transport. A grasp
is successful only when it remains attached and clears the minimum lift-height
threshold.

## Train

```powershell
python -m ravens.train_grasp_q `
  --data_dir=./data/rgbd-grasp-attempts `
  --root_dir=. `
  --steps=40000 `
  --n_rotations=16 `
  --n_widths=5 `
  --gpu_limit=5
```

Checkpoints are written to `checkpoints/grasp-q/`. The next integration step is
to use the learned Q values as the grasp term in the existing CEM and beam
planners. The same model can become an off-policy RL critic once collection
records complete `(observation, action, reward, next_observation)` transitions.

## Evaluate on held-out attempts

Collect a separate directory with different seeds, then evaluate without using
those samples for training:

```powershell
python -m ravens.eval_grasp_q `
  --data_dir=./data/rgbd-grasp-attempts-validation `
  --checkpoint=./checkpoints/grasp-q/grasp-q-ckpt-40000.h5 `
  --output=./results/grasp-q-validation.json `
  --gpu_limit=5
```

The report includes accuracy, balanced accuracy, precision, recall,
specificity, ROC AUC, log loss, and the confusion-matrix counts.

To test whether predicted values improve executed grasp choice, run the paired
selection benchmark on identical unseen scenes:

```powershell
python -m ravens.eval_grasp_selection `
  --checkpoint=./checkpoints/grasp-q/grasp-q-ckpt-500.h5 `
  --episodes=20 `
  --seed=30000 `
  --output=./results/grasp-selection.json `
  --gpu_limit=5
```

Depth connected components propose object-center pixels, principal-axis yaw,
and approximate jaw width because training only supervises attempted actions,
not every possible pixel and orientation. The benchmark then compares a random
depth proposal, a privileged simulator-geometry baseline, and the proposal
with the highest learned Q value. For small datasets, yaw/width bins with fewer
than `--min_action_samples` training examples are excluded.

The first end-to-end pilot and its limitations are recorded in the
[`pilot results`](rgbd-grasp-pilot-results.md).

## Current simulator caveat

The Ravens parallel-jaw gripper still uses geometry gating followed by a fixed
constraint to carry an accepted object. Consequently, `success` currently means
"accepted and lifted by the Ravens grasp model," not force-closure under
realistic friction. Contact-rich validation should replace this approximation
before interpreting results as real-world grasp reliability.
