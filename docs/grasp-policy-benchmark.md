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
| GPD | Planned | RGB-D point cloud, 6-DoF frame conversion, reachability filter |
| Contact-GraspNet | Planned | Separate TensorFlow environment and the same 6-DoF filters |
| QT-Opt-style RL | Planned | Simulator replay data and closed-loop action policy |
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

## Ten-seed analytic safety screen

The first ten held-out seeds completed without errors using reduced beam and
standard CEM. Results are paired against the existing geometry runs.

| Planner | Geometry | + analytic | Delta | Wins / ties / losses | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Reduced beam | 0.500 | 0.583 | +0.083 | 1 / 9 / 0 | [0.000, 0.250] |
| Standard CEM | 0.683 | 0.667 | -0.017 | 2 / 5 / 3 | [-0.183, 0.167] |

Beam + analytic produced 5 full, 3 partial, and 2 zero-reward episodes. Its
gate changed the preferred object on 33 of 103 decisions and abstained on 70.
CEM + analytic produced 4 full, 6 partial, and no zero-reward episodes. Its
gate changed the object on 7 of 103 decisions and abstained on 78.

The beam gain came from one seed, while the other nine tied; the normal paired
interval also crossed zero. CEM's small loss was likewise unresolved. These
results pass the integration safety screen but do not establish policy quality.
A 100-seed paired run is required before enabling the policy by default.

## Evaluation sequence

1. Run a 10-seed safety screen for each new adapter.
2. Reject adapters that introduce execution errors or a large paired loss.
3. Run surviving policies on all 100 held-out seeds.
4. Report reward, full/partial/zero episodes, rollouts, latency, gate behavior,
   paired wins/ties/losses, and a paired bootstrap confidence interval.
5. Compare 6-DoF methods separately until the robot controller supports their
   full action space; filtering them to top-down poses is an adapter benchmark,
   not a complete evaluation of those methods.
