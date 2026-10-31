# Grasp-policy benchmark

## Goal

Compare grasp-policy families through one candidate-scoring contract and the
same held-out scene seeds. Every compatible scorer returns a probability,
uncertainty, and conservative quality for each proposed grasp. Beam and CEM
can then consume the scores without policy-specific planner code.

## Compatibility

| Policy family | Current status | Required adaptation |
|---|---|---|
| Native geometry | Complete | None |
| Candidate grasp Q | Complete | Candidate-conditioned RGB-D checkpoint |
| Robust analytic | Complete | Planar perturbation scoring |
| Dex-Net GQ-CNN | Planned | Model/environment isolation and planar adapter |
| Point-cloud adapter | Complete | World-frame NPZ subprocess protocol |
| GPD | Adapter ready; backend unavailable | Isolated Linux/PCL environment and wrapper |
| Contact-GraspNet | Adapter ready; backend unavailable | Isolated legacy TensorFlow environment and wrapper |
| Compact QT-Opt-style RL | Pipeline complete; smoke tested | Larger replay dataset and evaluation |
| Tactile / multi-finger | Blocked by platform | New sensors, end effector, action space, and task dynamics |

The robust analytic policy is inspired by Dex-Net's robustness objective but
is not a reproduction of Dex-Net or its GQ-CNN. It scores candidate centering,
surface flatness, object isolation, and jaw-width fit under deterministic pixel
and yaw perturbations. Its conservative quality is the mean score minus the
score standard deviation.

## Reproducible evaluation

Use the same seeds and planner settings as the existing geometry and learned-Q
experiments. The policy name is included in output filenames so results do not
overwrite one another.

Long evaluations can be sharded without changing scene or planner seeds. For
example, `--start_index=10 --n=30` evaluates held-out seed indices 10 through
39. Nonzero start indices are included in output filenames.

```bash
python -m ravens.beam_test \
  --task=stack-block-pyramid-parallel-jaw --split=test --n=100 \
  --beam_width=2 --horizon=1 --n_blocks=3 --n_place=3 --verify_k=1 \
  --grasp_policy=analytic --grasp_gate=true

python -m ravens.cem_test \
  --task=stack-block-pyramid-parallel-jaw --split=test --n=100 \
  --n_samples=24 --n_elites=5 --n_iters=3 --verify_k=3 \
  --grasp_policy=analytic --grasp_rank_objects=true
```

## Analytic evaluation

The ten-seed safety screen completed without errors and qualified both planner
integrations for the complete 100-seed paired evaluation. All full-run seeds
matched the existing geometry result sets exactly, and no episode errored.

| Planner | Geometry | + analytic | Delta | Wins / ties / losses | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Reduced beam | 0.515 | 0.522 | +0.007 | 7 / 88 / 5 | [-0.020, 0.035] |
| Standard CEM | 0.673 | **0.710** | +0.037 | 15 / 77 / 8 | [-0.003, 0.080] |

Beam + analytic produced 36 full, 54 partial, and 10 zero-reward episodes. Its
gate accepted 304 of 1018 decisions and abstained on 714. Average planning cost
was 60.8 rollouts per episode, effectively matching geometry beam's 61.3.

CEM + analytic produced 59 full, 32 partial, and 9 zero-reward episodes. Its
gate accepted 258 of 925 decisions, abstained on 667, and changed CEM's object
choice 55 times. It averaged 703.0 rollouts per episode versus 714.4 for CEM.

The initial beam gain shrank to near zero at 100 seeds. CEM + analytic has the
best observed mean and improved more seeds than it regressed, but its confidence
interval narrowly includes zero. It is therefore promising rather than a
statistically established improvement and remains disabled by default.

## Point-cloud adapter test

The next integration layer converts the fused RGB-D heightmap into a
world-frame `N x 3` point cloud with aligned RGB values. An isolated subprocess
receives `scene.npz` and writes `grasps.npz`; this keeps GPD or Contact-GraspNet
dependencies out of the working Ravens environment.

The input protocol contains:

- `points`: float32 world coordinates, shape `(N, 3)`;
- `colors`: uint8 RGB values, shape `(N, 3)`;
- `frame`: the string `world`;
- `camera_positions`: the three configured camera origins in world coordinates.

The backend output contains `positions`, `approaches`, `closing_axes`,
`widths`, and `scores`, plus optional `uncertainties`. The adapter rejects
non-finite, out-of-workspace, excessive-width, and non-top-down grasps before
matching proposals to objects. This restriction is necessary because the
current parallel-jaw primitive cannot execute arbitrary 6-DoF approaches.

The subprocess protocol, frame conversion, filtering, and object matching all
pass unit tests. A live simulator test supplied six proposals through the same
adapter, matched all six visible objects, and completed the selected lift with
attachment and no timeout.

Actual GPD and Contact-GraspNet inference is not yet available on this machine.
The current environment is Windows, Python 3.10, and TensorFlow 2.10; PCL,
CMake, Open3D, PyTorch, and both model packages are absent. WSL is installed as
a command but has no usable distribution or virtualization support. The next
external-model step therefore requires a separate compatible Linux machine or
enabling WSL2/virtualization, plus pretrained model weights.

Once a wrapper is available, both planners can invoke it without importing the
external framework. Each command argument is supplied separately:

```bash
python -m ravens.cem_test \
  --grasp_policy=gpd --grasp_rank_objects=true --grasp_refine_pose=true \
  --grasp_backend_arg=python \
  --grasp_backend_arg=/path/to/gpd_npz_wrapper.py

python -m ravens.cem_test \
  --grasp_policy=contact_graspnet --grasp_rank_objects=true \
  --grasp_refine_pose=true \
  --grasp_backend_arg=python \
  --grasp_backend_arg=/path/to/contact_graspnet_npz_wrapper.py
```

The wrapper receives `--input <scene.npz> --output <grasps.npz>` automatically.
Without `--grasp_refine_pose=true`, external scores can rank objects but the
planner deliberately preserves its native center grasp.

## Compact QT-Opt-style test

A separate replay dataset now records
`(RGB-D state, grasp, reward, next RGB-D state, done)` transitions. Collection
is closed-loop: a successful lift removes that object from the active workspace,
the next observation contains the remaining objects, and each newly cleared
object earns `1 / initial_object_count` reward.

The TD trainer uses a slowly copied target network and candidate maximization:

```text
target = reward + gamma * max_next_candidate Q_target(next_state, candidate)
```

This is a compact QT-Opt-style experiment, not a reproduction of QT-Opt. It
uses discrete depth-derived candidates instead of continuous-action CEM and is
trained entirely in the Ravens simulator.

```bash
python -m ravens.collect_grasp_replay \
  --episodes=100 --output_dir=./data/rgbd-grasp-replay

python -m ravens.train_grasp_td \
  --replay_dir=./data/rgbd-grasp-replay \
  --root_dir=./runs/grasp-qt-opt --steps=1500

python -m ravens.beam_test \
  --grasp_policy=qt_opt \
  --grasp_checkpoint=./runs/grasp-qt-opt/checkpoints/grasp-qt-opt-1500.h5
```

The end-to-end smoke test collected eight sequential transitions: five
object-clearing rewards and three failures. Ten Bellman updates produced a
loadable checkpoint. Reduced beam then completed three held-out seeds with
mean reward 0.389 and no errors, exactly matching geometry because the safety
gate abstained on all 33 decisions. This validates collection, Bellman targets,
checkpoint loading, and planner integration only; it provides no evidence of
RL performance from such a tiny replay set.

### Complete policy comparison

| Policy | Mean reward | Full | Partial | Zero |
|---|---:|---:|---:|---:|
| Beam | 0.515 | 37 | 53 | 10 |
| Beam + gated Q | 0.510 | 37 | 50 | 13 |
| Beam + analytic | 0.522 | 36 | 54 | 10 |
| CEM | 0.673 | 55 | 35 | 10 |
| CEM + gated Q | 0.678 | 58 | 31 | 11 |
| **CEM + analytic** | **0.710** | **59** | 32 | **9** |

CEM + analytic exceeded CEM + gated Q by +0.032, but that paired bootstrap
interval also included zero: [-0.023, 0.088]. The robust conclusion remains
that the CEM planner family outperforms reduced beam; the choice among CEM
grasp-ranking variants is not yet resolved at 95% confidence.

## Evaluation sequence

1. Run a 10-seed safety screen for each new adapter.
2. Reject adapters that introduce execution errors or a large paired loss.
3. Run surviving policies on all 100 held-out seeds.
4. Report reward, full/partial/zero episodes, rollouts, latency, gate behavior,
   paired wins/ties/losses, and a paired bootstrap confidence interval.
5. Compare 6-DoF methods separately until the robot controller supports their
   full action space; filtering them to top-down poses is an adapter benchmark,
   not a complete evaluation of those methods.
