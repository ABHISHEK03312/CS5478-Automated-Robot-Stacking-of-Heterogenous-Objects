# Parallel-Jaw Pyramid Transporter

This directory contains the selected Transporter checkpoint for
`stack-block-pyramid-parallel-jaw`. The model was trained from scratch with
100 successful oracle demonstrations collected using the contact-driven
parallel-jaw placement controller.

## Checkpoint

- Training step: 40,000
- Attention SHA-256: `89F1F450C4EA71743D92895E5D794BFFA7CF6EC461AAA04C7BD1F4C05C73691D`
- Transport SHA-256: `E49135176BA54582644EF480149C9A48E1E216CD09E2927BB8BD61A20ABB93F5`

The `.h5` files are stored with Git LFS. Install Git LFS before cloning or
pulling this checkpoint:

```shell
git lfs install
git lfs pull
```

## Evaluate

First activate the project environment and generate a local held-out test set
if one is not already available:

```shell
conda activate ravens-transporter
python -m ravens.demos --data_dir=./data/parallel-jaw-v2 --assets_root=./ravens/environments/assets/ --task=stack-block-pyramid-parallel-jaw --mode=test --n=10
```

Run the checkpoint on the generated episodes:

```shell
python -m ravens.test --root_dir=./models/parallel-jaw-v2 --data_dir=./data/parallel-jaw-v2 --assets_root=./ravens/environments/assets/ --task=stack-block-pyramid-parallel-jaw --agent=transporter --n_demos=100 --n_steps=40000 --n_tests=10 --gpu=0 --gpu_limit=5
```

Add `--disp=True` to watch the PyBullet execution. Generated datasets and
evaluation results are intentionally not committed.
