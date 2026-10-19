# Parallel-Jaw Pyramid Transporter: Completion Report

## Outcome

The `stack-block-pyramid-parallel-jaw` pipeline completed successfully on
19 October 2026. It generated new data using the corrected contact-driven
parallel-jaw controller, trained a Transporter agent from scratch, evaluated
four checkpoints on a common 10-episode subset, and evaluated the two final
candidates on all 100 held-out episodes.

The published default is the **40,000-step checkpoint** because it achieved
the highest standard Ravens mean task reward on the 100-episode evaluation.
The 30,000-step checkpoint achieved more completely solved pyramids, so both
results are retained below as an explicit tradeoff.

## Experimental setup

- Task: `stack-block-pyramid-parallel-jaw`
- Agent: `transporter`
- Training data: 100 successful oracle demonstrations
- Test data: 100 separately seeded successful oracle demonstrations
- Training budget: 40,000 gradient steps
- Checkpoint interval: 1,000 steps
- Training initialization: fresh weights; no v1 checkpoint was resumed
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB VRAM
- TensorFlow: 2.10.1
- GPU memory limit: 5 GiB
- Source implementation commit: `f74a2c9`

Train and test seeds use the Ravens even/odd split. Failed oracle collection
attempts were discarded; only demonstrations with total reward above 0.99
were saved. The model learns pick and place poses, while gripper closing,
contact descent, release, and retreat remain controller logic.

The end-to-end pipeline ran from 18 October 2026 00:10 to 23:54 Singapore
time. Training itself ran from 04:10 to 23:54; concurrent CPU evaluations at
10k, 20k, and 30k temporarily reduced its throughput.

## Ten-episode checkpoint screen

Every checkpoint was evaluated on the same first 10 test episodes.

| Checkpoint | Mean reward | Full pyramids | Partial | Zero reward |
|---:|---:|---:|---:|---:|
| 10k | 0.617 | 3/10 | 7/10 | 0/10 |
| 20k | 0.600 | 4/10 | 6/10 | 0/10 |
| 30k | **0.733** | **6/10** | 2/10 | 2/10 |
| 40k | 0.550 | 3/10 | 5/10 | 2/10 |

The non-monotonic scores show that the last checkpoint is not automatically
the best on a small validation subset.

## Final 100-episode metrics

| Checkpoint | Mean reward | Sample SD | Approx. 95% CI | Full pyramids | Partial | Zero reward |
|---:|---:|---:|---:|---:|---:|---:|
| 30k | 0.717 | 0.353 | 0.647–0.786 | **53/100** | 39/100 | 8/100 |
| 40k | **0.728** | 0.325 | 0.665–0.792 | 48/100 | **47/100** | **5/100** |

Reward is the fraction of the six correctly completed pyramid placements.
The exact reward distribution was:

| Checkpoint | 0/6 | 1/6 | 2/6 | 3/6 | 4/6 | 5/6 | 6/6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30k | 8 | 5 | 12 | 13 | 1 | 8 | 53 |
| 40k | 5 | 7 | 8 | 13 | 8 | 11 | 48 |

On paired episodes, 40k scored higher than 30k on 29 scenes, tied on 37,
and scored lower on 34. Its mean paired advantage was 0.0117 reward, with an
approximate 95% interval of -0.055 to 0.079. The overlap indicates that the
small mean advantage should not be interpreted as a decisive improvement.
The 40k checkpoint is published because mean reward is the standard Ravens
metric and it produced fewer complete failures. For strict full-pyramid
success, the 30k checkpoint was better in this evaluation.

## Published model

The inference-ready checkpoint is stored under:

```text
models/parallel-jaw-v2/checkpoints/
  stack-block-pyramid-parallel-jaw-transporter-100-0/
    attention-ckpt-40000.h5
    transport-ckpt-40000.h5
```

The weights are tracked with Git LFS. Their SHA-256 hashes are:

- Attention: `89F1F450C4EA71743D92895E5D794BFFA7CF6EC461AAA04C7BD1F4C05C73691D`
- Transport: `E49135176BA54582644EF480149C9A48E1E216CD09E2927BB8BD61A20ABB93F5`

See [`models/parallel-jaw-v2/README.md`](../models/parallel-jaw-v2/README.md)
for setup and evaluation commands.

## Completion status and limitations

- Data collection: complete (100 train, 100 test)
- Training: complete (40,000/40,000 steps)
- Checkpoint screening: complete (10k, 20k, 30k, 40k)
- Final evaluation: complete (30k and 40k, 100 episodes each)
- Published inference model: complete (40k, Git LFS)
- Raw datasets, TensorBoard logs, result pickles, and intermediate checkpoints
  remain local and are intentionally excluded from Git.

The task still uses a kinematic temporary constraint to carry objects rather
than learning contact-rich finger dynamics. Results are simulation-only. The
same 100-episode split was used to compare 30k and 40k, and the training loop
also samples the test split for validation loss; a future study should add a
separate validation split and reserve a final untouched test set.
