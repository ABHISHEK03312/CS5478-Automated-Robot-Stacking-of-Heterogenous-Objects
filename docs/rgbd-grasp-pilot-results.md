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
