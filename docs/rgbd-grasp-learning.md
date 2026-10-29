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
jaw width, observed grasp success, timeout state, and collection metadata.

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

## Current simulator caveat

The Ravens parallel-jaw gripper still uses geometry gating followed by a fixed
constraint to carry an accepted object. Consequently, `success` currently means
"accepted and lifted by the Ravens grasp model," not force-closure under
realistic friction. Contact-rich validation should replace this approximation
before interpreting results as real-world grasp reliability.
