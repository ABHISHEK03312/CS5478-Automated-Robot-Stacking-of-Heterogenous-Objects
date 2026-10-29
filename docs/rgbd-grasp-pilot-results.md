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
