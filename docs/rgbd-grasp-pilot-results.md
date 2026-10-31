# RGB-D Grasp-Q Pilot Results

## Purpose

This pilot verifies that the collection, training, checkpoint, and held-out
evaluation pipeline works end to end. It is a small engineering validation,
not a final model comparison.

## Data

| Split | Seeds | Attempts | Successful lifts | Failed lifts |
|---|---:|---:|---:|---:|
| Train | 10000–10079 | 80 | 43 | 37 |
| Validation | 20000–20019 | 20 | 11 | 9 |

An attempt is positive only when an object remains attached and is lifted at
least 5 cm. Training and validation use disjoint simulator seeds.

## Training

- Model: dense RGB-D grasp-Q ResNet
- Action bins: 16 yaw bins × 5 jaw-width bins
- Updates: 500
- Optimizer: Adam, learning rate `1e-4`
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- Checkpoints evaluated: 250 and 500 updates

## Held-out results

| Checkpoint | Accuracy | Balanced accuracy | ROC AUC | Log loss | Precision | Recall | Specificity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 0.600 | 0.576 | 0.848 | 0.446 | 0.600 | 0.818 | 0.333 |
| 500 | **0.800** | **0.788** | **0.939** | **0.334** | **0.769** | **0.909** | **0.667** |

The 500-update confusion matrix is 10 true positives, 6 true negatives,
3 false positives, and 1 false negative.

## Interpretation and limitations

The model learned a signal that transfers to unseen scene seeds, and the
500-update checkpoint improved every reported held-out metric. The sample is
too small for a stable confidence interval or hyperparameter choice, however.

The evaluation scores the grasp action that was actually attempted. It does
not yet test whether selecting the global maximum from the dense Q map improves
robot success. The simulator also uses geometry gating and a temporary fixed
constraint rather than frictional finger contact.

The next experiment should collect at least 1,000 training and 200 validation
attempts, then compare geometry sampling, random candidates, and Q-selected
grasps on the same unseen scenes before integrating the scorer into CEM.

## Expanded 240-attempt follow-up

Four lift-only collection shards added 160 attempts, producing 240 training
attempts in total: 132 successful lifts and 108 failures. All 16 yaw bins and
both object-width bins were represented; 31 of the 32 observed yaw/width
combinations contained at least three examples.

| Checkpoint | Accuracy | Balanced accuracy | ROC AUC | Log loss |
|---:|---:|---:|---:|---:|
| 750 | 0.700 | 0.687 | **0.828** | 1.352 |
| 1500 | 0.700 | 0.687 | 0.808 | **0.622** |

These results use the original 20-attempt validation split. They do not show an
improvement over the 80-attempt pilot, and the validation set is too small for
model selection. They reinforce the need for a larger held-out split and
balanced action coverage.

Direct dense-map argmax was also tested on three unseen robot scenes. It failed
all three because unsupervised pixel/action channels produced overconfident
scores. The inference pipeline therefore now uses connected depth components
to propose center, principal-axis yaw, and jaw width, with grasp-Q used only to
rank those physically plausible candidates.

The geometry-constrained selector initially failed because model preprocessing
normalized the shared raw heightmap in place. This changed an approximately
`0.04 m` pick height into roughly `4.0 m` before pixel-to-world conversion.
Inference now preprocesses a copy, and a regression test protects the raw
heightmap.

After the fix, the 240-attempt, 750-update checkpoint was evaluated on 20 unseen
scene seeds. Grasp-Q ranking, random depth proposals, and privileged geometry
all achieved 20/20 successful lifts with zero timeouts. This verifies execution
but is a saturated benchmark: connected-component centers combined with the
kinematic fixed-constraint gripper are already sufficient for perfect lifting.
The next evaluation must introduce harder clutter, off-center proposals, and
contact/friction-based failures before it can measure the value of learned
ranking.

## Candidate-conditioned Q and planner follow-up

The dense output model was replaced for ranking by a smaller classifier that
only evaluates physically proposed grasps. It rotates a `48 x 48` local RGB-D
crop into the candidate yaw frame, adds normalized jaw width, trains with
class-balanced mini-batches, and estimates uncertainty with Monte Carlo
dropout. Ranking uses `mean_probability - uncertainty`.

| Candidate checkpoint | Accuracy | Balanced accuracy | ROC AUC | Log loss |
|---:|---:|---:|---:|---:|
| 250 | 0.900 | 0.889 | 0.859 | 0.450 |
| 500 | **0.950** | **0.944** | 0.970 | 0.231 |
| 750 | 0.900 | 0.889 | **0.980** | **0.198** |

The 750-update checkpoint is used for ranking because ranking quality and
probability quality, rather than the fixed 0.5 classification threshold, are
the relevant criteria. The validation split still contains only 20 attempts,
so these measurements require confirmation on a larger held-out set.

Planner integration was tested on seeds 1, 3, and 5 with deliberately reduced
search budgets (`8 x 2` CEM and beam width 2, horizon 1). These are smoke tests,
not statistically powered comparisons.

| Policy | Candidate-Q use | Mean reward | Full stacks |
|---|---|---:|---:|
| CEM | none (baseline) | 0.667 | 2/3 |
| Beam | none (baseline) | 0.389 | 1/3 |
| Beam | tie-break weight 0.1 | 0.444 | 1/3 |
| Beam | aggressive weight 1.0 | 0.167 | 0/3 |

CEM selects one object before optimizing its placement, so there is no set of
pick candidates for Q to reorder. Its default integration therefore computes
Q diagnostically but preserves the object-center pick; a seed-1 check exactly
matched the baseline action trace and achieved reward 1.0. Directly replacing
the center pose with image proposals was also tested and rejected: CEM mean
reward fell to 0.056 on three seeds. Beam can rank several first-move objects,
but Q must remain a small tie-break term so it does not overwhelm placement
and stability value.

## Paired 100-seed beam evaluation

The three-seed result was followed by a complete paired evaluation on all 100
held-out pyramid test seeds. Both variants used beam width 2, horizon 1, three
candidate blocks, three placements, and one real-arm verification. Candidate
Q used four dropout passes and weight 0.1. There were no errored episodes.

| Policy | Mean reward | Full | Partial | Zero | Avg. trials |
|---|---:|---:|---:|---:|---:|
| Beam baseline | **0.515** | **37** | **53** | **10** | 10.17 |
| Beam + candidate Q | 0.495 | 35 | 51 | 14 | 10.31 |
| Beam + gated candidate Q | 0.510 | **37** | 50 | 13 | 10.08 |

The paired mean difference (`Q - baseline`) was -0.020. Its normal 95%
confidence interval was [-0.064, 0.024], and a fixed-seed 20,000-sample paired
bootstrap interval was [-0.063, 0.025]. Q improved 18 seeds, tied 58, and
regressed 24. The interval includes zero, so this test does not establish a
statistically significant difference, but it clearly rejects the earlier
three-seed suggestion that Q was already reliably better. The increase from
10 to 14 zero-reward episodes is especially undesirable for robustness.

High held-out grasp ROC AUC therefore did not translate into better stacking.
The current lift-attempt dataset teaches single-grasp classification, while
beam needs reliable *within-scene ranking* among alternative objects and must
balance graspability against downstream placement value. The next model should
train on multiple executed candidates per identical scene using a pairwise
ranking loss, include heterogeneous objects and clutter, and abstain to the
geometry baseline unless its conservative score exceeds a validated margin.

### Uncertainty-gated Q follow-up

A shared abstaining selector was added to beam and CEM. It accepts Q only when
the top proposal has probability at least 0.65, uncertainty at most 0.08, and
a conservative-score margin of at least 0.15; otherwise it preserves the
planner fallback. The beam variant was evaluated on all 100 test seeds.

Gated beam reached 0.510 mean reward with 37 full, 50 partial, and 13 zero
episodes. Relative to geometry-only beam, its paired delta was -0.005 with
normal 95% interval [-0.037, 0.027] and bootstrap interval [-0.037, 0.027]. It
improved 9 seeds, tied 80, and regressed 11. Relative to ungated Q it gained
0.015, but that interval also included zero. The gate accepted 559 of 1008
decisions and abstained on 449; every episode contained at least one accepted
decision.

The gate therefore limits Q damage but does not produce a measurable beam
benefit. The initial 10-seed CEM safety screen also regressed (0.617 versus
0.683), but the requested complete 100-seed run reversed that small-sample
ordering: gated CEM + Q scored 0.678 versus 0.673 for CEM. The +0.005 paired
difference was not statistically resolved, so both hooks remain research
options and CEM without Q remains the simpler recommended policy.

## Alternative algorithm: standard CEM

Because grasp Q did not improve beam, the same 100 held-out seeds were also
evaluated with the standard cross-entropy method planner: 24 samples per
iteration, 5 elites, 3 iterations, teleport rollouts, and real-arm verification
of the top 3 placements. CEM retains the reliable object-center grasp and uses
simulator rollouts to optimize placement instead of adding noisy grasp-Q values.

| Policy | Mean reward | Full | Partial | Zero | Avg. trials | Avg. rollouts |
|---|---:|---:|---:|---:|---:|---:|
| Beam (`K=2`, `H=1`) | 0.515 | 37 | 53 | **10** | 10.17 | 61.3 |
| Beam + gated Q | 0.510 | 37 | 50 | 13 | 10.08 | 61.2 |
| Beam + analytic | 0.522 | 36 | 54 | **10** | 10.18 | 60.8 |
| CEM | 0.673 | 55 | 35 | **10** | 9.40 | 714.4 |
| CEM + gated Q | 0.678 | 58 | 31 | 11 | 9.29 | 706.0 |
| **CEM + analytic** | **0.710** | **59** | 32 | **9** | **9.25** | 703.0 |

CEM improved mean reward over beam by +0.158. The paired normal 95% confidence
interval was [0.086, 0.231], and the paired 20,000-sample bootstrap interval
was [0.087, 0.230]. CEM improved 47 seeds, tied 34, and regressed 19. Relative
to the ungated beam + candidate Q run, its paired gain was +0.178 with
bootstrap interval [0.103, 0.253]. All 100 CEM episodes completed without
evaluation errors.

CEM + gated Q improved 9 seeds, tied 85, and regressed 6 relative to CEM. Its
paired delta was only +0.005: the normal 95% interval was [-0.036, 0.046] and
the bootstrap interval was [-0.038, 0.045]. The gate accepted 442 of 929
object-selection decisions, abstained to CEM's geometry choice on 487, and
actually changed the selected object 81 times. All 100 episodes completed
without errors.

The perturbation-robust analytic scorer was also evaluated on the complete 100
seeds. Beam + analytic was effectively neutral: +0.007 versus beam, with
bootstrap interval [-0.020, 0.035] and 7 wins, 88 ties, and 5 losses. CEM +
analytic reached the highest observed mean, 0.710. Relative to CEM its paired
gain was +0.037 with bootstrap interval [-0.003, 0.080], comprising 15 wins,
77 ties, and 8 losses. The interval narrowly includes zero, so this remains a
promising experimental result rather than a demonstrated improvement.

This establishes the CEM family as the best tested approach in this RGB-D
grasp experiment, but not as a compute-matched winner: it used roughly 11.7
times as many simulator rollouts as reduced beam. A fair efficiency study
should compare reward as a function of rollout budget and include the default
deeper beam configuration. For the current goal of improving task success,
standard CEM remains the conservative default. CEM + analytic has the highest
point estimate, but its gain has not reached 95% paired confidence and should
remain experimental pending replication on heterogeneous-object scenes.
