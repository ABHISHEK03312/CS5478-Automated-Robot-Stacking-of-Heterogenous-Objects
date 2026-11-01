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
| Compact QT-Opt-style RL | 10-seed safety screen complete | No full run: 0 wins, 1 loss |
| Joint pick-place return critic | Gated ensemble safety screen complete | Collect hard CEM candidate pairs |
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

The subsequent experiment collected 100 episodes in four deterministic shards:
1,054 transitions covering seeds 42000 through 42099, with 570 positive rewards,
484 zero-reward actions, and exactly 100 terminal transitions. The critic was
trained for 1,500 Bellman updates with target-network copies every 100 updates.

On the matched 10-seed reduced-beam safety screen, QT-style ranking scored
0.483 versus 0.500 for geometry. The paired delta was -0.017 with bootstrap
interval [-0.050, 0.000]: zero wins, nine ties, and one loss. The gate accepted
34 of 104 decisions and abstained on 70. QT-style beam produced 4 full, 3
partial, and 3 zero-reward episodes; all episodes completed without errors.

This checkpoint does not qualify for a 100-seed run. Its sequential object-
clearing objective validates Bellman learning but is not sufficiently aligned
with downstream stacking value. A future RL version should include placement
in the action and train on actual stacking returns rather than lift-and-clear
rewards.

## Joint pick-and-place return critic

The next RL formulation conditions on the complete action: a grasp-aligned
RGB-D crop, jaw width, placement XYZ, and placement yaw. Its replay collector
mixes oracle and perturbed placements and records actual Ravens stacking reward
and the next RGB-D observation. Discounted episode returns train a separate
joint-action critic.

The full collection used four independent 25-episode shards and produced 958
valid transitions across 100 episodes: 176 positive, 766 zero, and 16 negative
reward steps. The intervention mix was nearly even, with 483 oracle actions and
475 perturbed actions. Off-image oracle picks are logged and end only the
affected episode rather than aborting a collection shard. The joint critic was
trained for 1,500 updates.

The critic is an optional bonus over each planner's physics score; it does not
replace Beam or CEM search. On the matched ten-seed safety screen, weight 0.5
produced the following results:

| Planner | Geometry | + joint action Q | Delta | Wins / ties / losses | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Reduced Beam | 0.500 | **0.600** | +0.100 | 3 / 5 / 2 | [-0.150, 0.350] |
| Standard CEM | **0.683** | 0.600 | -0.083 | 2 / 6 / 2 | [-0.300, 0.083] |

Both integrations completed without errors. Beam + joint action Q produced four
full, five partial, and one zero-reward episode, so it qualifies for a complete
100-seed evaluation, although the ten-seed interval is wide. CEM + joint action
Q produced four full and six partial episodes. It is not promoted at weight 0.5
because its point estimate regressed; a smaller weight or uncertainty gate
should be screened before a full CEM run.

### Conservative CEM ensemble

Three independently seeded joint-action critics were then used to inspect only
CEM's top five physics candidates. The adapter keeps native CEM's top-three arm
verification unchanged whenever Q abstains. A Q-selected candidate is added to
that verification set only when ensemble uncertainty is at most 0.05 and its
predicted margin over CEM's winner is at least 0.05; physics still makes the
final execution choice.

On the same ten seeds, both weight 0.1 and weight 0.5 exactly reproduced native
CEM: mean reward 0.683, five full and five partial episodes, and no errors. At
weight 0.1 the ensemble checked 101 decisions, made zero changes, and recorded
one explicit uncertainty abstention. Per-seed rewards, trials, and rollouts
matched native CEM. This proves the fallback is safe, but the critic does not
separate CEM's near-tied top candidates strongly enough to improve planning.

The next training set should therefore contain paired rollouts of CEM's top
candidates, especially candidates with similar physics scores but different
executed stacking returns. A pairwise ranking loss is more appropriate for
that data than further tuning the global Q weight on these test seeds.

### Hard-candidate pairwise pilot

The counterfactual collector now exposes CEM's top five unique proposals and
executes each one in an isolated simulator snapshot. It records actual task
reward and restores the scene before testing the next proposal or advancing
the native CEM episode. This produced informative groups such as
`[0, 0.167, 0, 0, -0.167]` without contaminating the behavior policy.

Twenty reduced-CEM training episodes produced 995 candidate transitions and
298 ordered preference pairs. Splitting complete decision groups rather than
individual candidates yielded 238 training and 60 validation pairs. Starting
from the 100-episode joint-action checkpoint, pairwise fine-tuning at `1e-5`
improved held-out ranking accuracy from 0.517 to 0.550 (training accuracy
0.555). A `1e-4` run overfit, ending at 0.483 validation accuracy. The better
checkpoint preserved native CEM on a standard one-seed diagnostic but made no
ranking changes.

The counterfactual data pipeline is therefore valid, but the current scalar-Q
architecture is not yet accurate enough to control CEM. The next model should
compare candidates directly using relative placement deltas and physics-score
deltas; forcing changes by weakening the gate would not be evidence-based.

A candidate-relative model was subsequently implemented with a shared RGB-D
crop encoder and explicit first/second action features, scaled placement
deltas, yaw delta, and physics-score delta. It trains on both pair directions
and integrates through a calibrated challenger-vs-CEM-winner probability gate.
The scratch model reached 0.656 training accuracy but only 0.500 on held-out
decision groups. Initializing its RGB-D backbone from scalar Q increased
training accuracy to 0.693 while validation fell to 0.483. At a conservative
0.80 gate, the standard one-seed planner diagnostic exactly preserved native
CEM and made no changes.

The replay was then expanded to 60 independent scenes. Validation now holds out
entire scene seeds, preventing candidates or decisions from the same simulated
scene from appearing on both sides of the split. With 769 training pairs and
148 pairs from 12 held-out scenes, the pretrained relative ranker reached 0.649
validation accuracy after 2,000 updates (training accuracy 0.550). The lower
training score is acceptable here because scene-held-out accuracy is the
qualification metric and the held-out set is relatively small.

The qualified model completed a full-stack one-seed smoke test. On the standard
ten-seed CEM benchmark, both the conservative 0.80 gate and the intended 0.65
gate exactly matched native CEM: 0.683 mean reward, 5 full stacks, 5 partial
stacks, and no zero-reward episodes or errors. Thus added scene diversity fixed
the chance-level offline result, but the ranker still abstains on live CEM
candidates at safe thresholds. It remains disabled; standard CEM remains the
recommended policy until probability calibration or additional informative
hard pairs produce a measured online gain.

```bash
python -m ravens.collect_cem_pair_replay \
  --episodes=20 --candidates=5 --output_dir=./data/cem-pair-replay

python -m ravens.train_stack_action_rank \
  --replay_dir=./data/cem-pair-replay \
  --init_checkpoint=./runs/stack-action-q/checkpoints/stack-action-q-1500.h5 \
  --root_dir=./runs/stack-action-rank --steps=1000

python -m ravens.train_stack_pair_rank \
  --pair_replay_dir=./data/cem-pair-replay \
  --pair_init_checkpoint=./runs/stack-action-q/checkpoints/stack-action-q-1500.h5 \
  --root_dir=./runs/stack-pair-rank --steps=2000 \
  --validation_fraction=0.2
```

```bash
python -m ravens.collect_stack_action_replay \
  --episodes=100 --output_dir=./data/stack-action-replay

python -m ravens.train_stack_action_q \
  --replay_dir=./data/stack-action-replay \
  --root_dir=./runs/stack-action-q --steps=1500

python -m ravens.beam_test \
  --n=10 --beam_width=2 --horizon=1 --n_blocks=3 --n_place=3 --verify_k=1 \
  --stack_action_checkpoint=./runs/stack-action-q/checkpoints/stack-action-q-1500.h5

python -m ravens.cem_test \
  --n=10 --n_samples=24 --n_elites=5 --n_iters=3 --verify_k=3 \
  --stack_action_checkpoint=./runs/stack-action-q/checkpoints/stack-action-q-1500.h5
```

Collection can be parallelized safely by writing each process to a separate
directory and repeating `--replay_shard` during training. This avoids index
collisions while preserving every transition.

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
