# Ravens - Transporter Networks

Ravens is a collection of simulated tasks in PyBullet for learning vision-based robotic manipulation, with emphasis on pick and place.
It features a Gym-like API with 10 tabletop rearrangement tasks, each with (i) a scripted oracle that provides expert demonstrations (for imitation learning), and (ii) reward functions that provide partial credit (for reinforcement learning).

<img src="docs/tasks.png"><br>

(a) **block-insertion**: pick up the L-shaped red block and place it into the L-shaped fixture.<br>
(b) **place-red-in-green**: pick up the red blocks and place them into the green bowls amidst other objects.<br>
(c) **towers-of-hanoi**: sequentially move disks from one tower to another—only smaller disks can be on top of larger ones.<br>
(d) **align-box-corner**: pick up the randomly sized box and align one of its corners to the L-shaped marker on the tabletop.<br>
(e) **stack-block-pyramid**: sequentially stack 6 blocks into a pyramid of 3-2-1 with rainbow colored ordering.<br>
(f) **palletizing-boxes**: pick up homogeneous fixed-sized boxes and stack them in transposed layers on the pallet.<br>
(g) **assembling-kits**: pick up different objects and arrange them on a board marked with corresponding silhouettes.<br>
(h) **packing-boxes**: pick up randomly sized boxes and place them tightly into a container.<br>
(i) **manipulating-rope**: rearrange a deformable rope such that it connects the two endpoints of a 3-sided square.<br>
(j) **sweeping-piles**: push piles of small objects into a target goal zone marked on the tabletop.<br>

Some tasks require generalizing to unseen objects (d,g,h), or multi-step sequencing with closed-loop feedback (c,e,f,h,i,j).

**Team:** this repository is developed and maintained by [Andy Zeng](https://andyzeng.github.io/), [Pete Florence](http://www.peteflorence.com/), [Daniel Seita](https://people.eecs.berkeley.edu/~seita/), [Jonathan Tompson](https://jonathantompson.github.io/), and [Ayzaan Wahid](https://www.linkedin.com/in/ayzaan-wahid-21676148/). This is the reference repository for the paper:

### Transporter Networks: Rearranging the Visual World for Robotic Manipulation
[Project Website](https://transporternets.github.io/)&nbsp;&nbsp;•&nbsp;&nbsp;[PDF](https://arxiv.org/pdf/2010.14406.pdf)&nbsp;&nbsp;•&nbsp;&nbsp;Conference on Robot Learning (CoRL) 2020

*Andy Zeng, Pete Florence, Jonathan Tompson, Stefan Welker, Jonathan Chien, Maria Attarian, Travis Armstrong,<br>Ivan Krasin, Dan Duong, Vikas Sindhwani, Johnny Lee*

**Abstract.** Robotic manipulation can be formulated as inducing a sequence of spatial displacements: where the space being moved can encompass an object, part of an object, or end effector. In this work, we propose the Transporter Network, a simple model architecture that rearranges deep features to infer spatial displacements from visual input—which can parameterize robot actions. It makes no assumptions of objectness (e.g. canonical poses, models, or keypoints), it exploits spatial symmetries, and is orders of magnitude more sample efficient than our benchmarked alternatives in learning vision-based manipulation tasks: from stacking a pyramid of blocks, to assembling kits with unseen objects; from manipulating deformable ropes, to pushing piles of small objects with closed-loop feedback. Our method can represent complex multi-modal policy distributions and generalizes to multi-step sequential tasks, as well as 6DoF pick-and-place. Experiments on 10 simulated tasks show that it learns faster and generalizes better than a variety of end-to-end baselines, including policies that use ground-truth object poses. We validate our methods with hardware in the real world.

## Installation

**Step 1.** Recommended: install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) with Python 3.7.

```shell
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u
echo $'\nexport PATH=~/miniconda3/bin:"${PATH}"\n' >> ~/.profile  # Add Conda to PATH.
source ~/.profile
conda init
```

**Step 2.** Create and activate Conda environment, then install GCC and Python packages.

```shell
cd ~/ravens
conda create --name ravens python=3.7 -y
conda activate ravens
sudo apt-get update
sudo apt-get -y install gcc libgl1-mesa-dev
pip install -r requirements.txt
python setup.py install --user
```

**Step 3.** Recommended: install GPU acceleration with NVIDIA [CUDA](https://developer.nvidia.com/cuda-toolkit) 10.1 and [cuDNN](https://developer.nvidia.com/cudnn) 7.6.5 for Tensorflow.
```shell
./oss_scripts/install_cuda.sh  #  For Ubuntu 16.04 and 18.04.
conda install cudatoolkit==10.1.243 -y
conda install cudnn==7.6.5 -y
```

### Alternative: Pure `pip`

As an example for Ubuntu 18.04:

```shell
./oss_scipts/install_cuda.sh  #  For Ubuntu 16.04 and 18.04.
sudo apt install gcc libgl1-mesa-dev python3.8-venv
python3.8 -m venv ./venv
source ./venv/bin/activate
pip install -U pip
pip install scikit-build
pip install -r ./requirements.txt
export PYTHONPATH=${PWD}
```

## Getting Started

**Step 1.** Generate training and testing data (saved locally). Note: remove `--disp` for headless mode.

```shell
python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion --mode=train --n=10
python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion --mode=test --n=100
```

To run with shared memory, open a separate terminal window and run `python3 -m pybullet_utils.runServer`. Then add `--shared_memory` flag to the command above.

**Step 2.** Train a model e.g., Transporter Networks model. Model checkpoints are saved to the `checkpoints` directory. Optional: you may exit training prematurely after 1000 iterations to skip to the next step.

```shell
python ravens/train.py --task=block-insertion --agent=transporter --n_demos=10
```

**Step 3.** Evaluate a Transporter Networks agent using the model trained for 1000 iterations. Results are saved locally into `.pkl` files.

```shell
python ravens/test.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion --agent=transporter --n_demos=10 --n_steps=1000
```

**Step 4.** Plot and print results.

```shell
python ravens/plot.py --disp=True --task=block-insertion --agent=transporter --n_demos=10
```

**Optional.** Track training and validation losses with Tensorboard.

```shell
python -m tensorboard.main --logdir=logs  # Open the browser to where it tells you to.
```

## Parallel-Jaw Block Stacking

The `stack-block-pyramid-parallel-jaw` task uses a directly mounted,
two-prismatic-finger kinematic gripper instead of the suction end effector.
Generate demonstrations with:

```shell
python -m ravens.demos --assets_root=./ravens/environments/assets/ --task=stack-block-pyramid-parallel-jaw --mode=train --n=100
python -m ravens.demos --assets_root=./ravens/environments/assets/ --task=stack-block-pyramid-parallel-jaw --mode=test --n=100
```

Train and evaluate a task-specific Transporter checkpoint with:

```shell
python -m ravens.train --task=stack-block-pyramid-parallel-jaw --agent=transporter --n_demos=100
python -m ravens.test --assets_root=./ravens/environments/assets/ --task=stack-block-pyramid-parallel-jaw --agent=transporter --n_demos=100 --n_steps=40000
```

The gripper is mounted directly on the UR5 wrist and its fingers visibly close
to the selected object's width. Because Ravens moves the arm kinematically,
grasp eligibility is checked inside the jaw volume and a temporary PyBullet
constraint carries the object until the fingers open. This avoids unstable
collision impulses while still rejecting picks that are not between the jaws.

The trained parallel-jaw checkpoint, evaluation commands, and final metrics
are available in [`models/parallel-jaw-v2`](models/parallel-jaw-v2) and the
[`completion report`](docs/parallel-jaw-v2-completion-report.md).

### RGB-D Grasp Quality Learning

The current grasping technique is a **hybrid depth-geometry + learned grasp-Q
approach** for top-down parallel-jaw grasps:

```text
RGB-D cameras
    -> fused color heightmap
    -> depth components propose object centers, principal-axis yaw, and width
    -> candidate-Q scores aligned local RGB-D crops and penalizes uncertainty
    -> robot executes the highest-ranked grasp
    -> CEM / beam search plans object order and placement
```

The geometry stage prevents the neural network from selecting arbitrary
background pixels or unsupported yaw/width combinations. The learned stage
uses scene appearance and depth to choose between the proposed grasps.

#### Training labels

The collector samples successful and failed grasp attempts. An attempt is
positive only when an object remains attached and is lifted by at least 5 cm.
Each sample stores the fused RGB-D heightmap, grasp pixel, yaw, jaw width,
grasped object, measured lift height, timeout state, and lift outcome.

#### Commands

```shell
python -m ravens.collect_grasp_data --assets_root=./ravens/environments/assets/ --attempts=1000 --output_dir=./data/rgbd-grasp-attempts-train
python -m ravens.collect_grasp_data --assets_root=./ravens/environments/assets/ --attempts=200 --seed=20000 --output_dir=./data/rgbd-grasp-attempts-validation
python -m ravens.train_grasp_candidate_q --candidate_data_dir=./data/rgbd-grasp-attempts-train --candidate_steps=1500 --gpu_limit=5
python -m ravens.eval_grasp_candidate_q --candidate_eval_data_dir=./data/rgbd-grasp-attempts-validation --candidate_checkpoint=./runs/grasp-candidate-q/checkpoints/grasp-candidate-q-750.h5 --gpu_limit=5
python -m ravens.beam_test --task=stack-block-pyramid-parallel-jaw --n=10 --grasp_checkpoint=./runs/grasp-candidate-q/checkpoints/grasp-candidate-q-750.h5 --grasp_weight=0.1
python -m ravens.cem_test --task=stack-block-pyramid-parallel-jaw --n=100 --n_samples=24 --n_elites=5 --n_iters=3 --verify_k=3
```

The planners now share an explicit grasp-policy adapter. `geometry` preserves
the native object-center pick, `candidate_q` loads the learned checkpoint, and
`analytic` uses a Dex-Net-inspired perturbation-robust geometry score (not the
Dex-Net GQ-CNN):

```bash
python -m ravens.beam_test --grasp_policy=analytic --grasp_gate=true
python -m ravens.cem_test --grasp_policy=analytic --grasp_rank_objects=true
```

The adapter is the integration point for future point-cloud policies such as
GPD and Contact-GraspNet. See the
[`grasp-policy benchmark plan`](docs/grasp-policy-benchmark.md) for scope,
compatibility constraints, and current smoke-test results.

#### Current status

- The first 80-train / 20-validation pilot reached 80% held-out accuracy and
  0.939 ROC AUC at 500 updates.
- Expanding training to 240 attempts improved yaw/width coverage but reached
  70% accuracy on the same small validation set, showing that checkpoint loss
  and dataset size alone do not guarantee better generalization.
- Unrestricted dense-map argmax failed in execution because sparsely supervised
  output locations became overconfident. Inference now ranks depth-derived
  grasp proposals and masks action bins without sufficient training coverage.
- The replacement candidate-conditioned CNN trains on balanced success/failure
  batches, rotates a 48 x 48 RGB-D crop into the proposed grasp frame, and uses
  Monte Carlo dropout to rank by mean success probability minus uncertainty.
  At 750 updates it reached 0.980 ROC AUC and 0.198 log loss on the same small
  20-attempt validation split (11 positive, 9 negative).
- An input-mutation bug initially made learned grasps execute at an invalid
  height; model preprocessing now copies the raw heightmap and has a regression
  test. After the fix, a 20-scene paired benchmark gave 20/20 lifts for learned
  Q ranking, random depth proposals, and privileged geometry. The benchmark is
  therefore saturated under the current kinematic grasp model and does not yet
  demonstrate an advantage for learning.
- Beam search can add candidate Q with a conservative `0.1` tie-break weight.
  The initial three-seed smoke comparison favored Q, but the full paired
  100-seed test did not: baseline mean reward was 0.515 and Q reached 0.495
  (paired delta -0.020, 95% bootstrap CI [-0.063, 0.025]). Q won on 18 seeds,
  tied on 58, and lost on 24; it also increased zero-reward episodes from 10
  to 14. The current grasp Q is therefore not better end-to-end and should be
  considered experimental rather than a default policy improvement.
  An uncertainty-gated variant recovered most of the loss but did not beat
  geometry-only beam: 0.510 versus 0.515 over 100 seeds (paired delta -0.005,
  bootstrap 95% CI [-0.037, 0.027]). It abstained on 449 of 1008 decisions.
  CEM has one predetermined object pick per step, so Q is diagnostic there and
  leaves the proven center pick unchanged by default. Experimental pose
  replacement is available via `--grasp_refine_pose`, but performed poorly.
  Uncertainty-gated Q object reordering is available through
  `--grasp_rank_objects`. Over the complete 100-seed test it reached 0.678
  versus 0.673 for standard CEM (paired delta +0.005, bootstrap 95% CI
  [-0.038, 0.045]). This is not a measurable improvement, so it remains
  disabled by default.
- Standard CEM is the current recommended policy. On the same 100 held-out
  seeds it reached 0.673 mean reward with 55 full stacks, improving over the
  reduced-beam baseline by +0.158 (paired bootstrap 95% CI [0.087, 0.230]).
  CEM + gated Q produced 58 full stacks and mean reward 0.678. The new robust
  analytic scorer reached the highest observed result: 0.710 with 59 full
  stacks and 9 zero-reward episodes. Its +0.037 paired gain over CEM was
  promising but unresolved (bootstrap 95% CI [-0.003, 0.080]), so standard CEM
  remains the default and analytic ranking remains experimental.
  CEM used substantially more planning compute: 714 simulator rollouts per
  episode versus 61 for beam, so this is a better operating point rather than
  a compute-matched algorithm comparison.
- This remains a one-step grasp Q-function (immediate-reward/contextual-bandit
  learning). Sequential off-policy RL remains a future milestone.

See the [`RGB-D grasp learning guide`](docs/rgbd-grasp-learning.md) for the
dataset format, training options, selection benchmark, and simulator
limitations. Pilot measurements are in the
[`RGB-D grasp-Q pilot report`](docs/rgbd-grasp-pilot-results.md).

## Datasets and Pre-Trained Models

Download our generated train and test datasets and pre-trained models.

```shell
wget https://storage.googleapis.com/ravens-assets/checkpoints.zip
wget https://storage.googleapis.com/ravens-assets/block-insertion.zip
wget https://storage.googleapis.com/ravens-assets/place-red-in-green.zip
wget https://storage.googleapis.com/ravens-assets/towers-of-hanoi.zip
wget https://storage.googleapis.com/ravens-assets/align-box-corner.zip
wget https://storage.googleapis.com/ravens-assets/stack-block-pyramid.zip
wget https://storage.googleapis.com/ravens-assets/palletizing-boxes.zip
wget https://storage.googleapis.com/ravens-assets/assembling-kits.zip
wget https://storage.googleapis.com/ravens-assets/packing-boxes.zip
wget https://storage.googleapis.com/ravens-assets/manipulating-rope.zip
wget https://storage.googleapis.com/ravens-assets/sweeping-piles.zip
```

The MDP formulation for each task uses transitions with the following structure:

**Observations:** raw RGB-D images and camera parameters (pose and intrinsics).

**Actions:** a primitive function (to be called by the robot) and parameters.

**Rewards:** total sum of rewards for a successful episode should be =1.

**Info:** 6D poses, sizes, and colors of objects.

## VLM Stacking


Zero-shot VLM with the included parallel-jaw Transporter checkpoint (40,000 steps):

```
python -m vlm_stacking.run_ravens_sim --models-root "./models/parallel-jaw-v2" --planner zero-shot --instruction "Build a stable tower with the largest block at the bottom." --dry-run
```

Beam search planning
```
python -m vlm_stacking.run_ravens_sim \
  --planner beam --beam-width 3 --beam-horizon 2 \
  --instruction "Build a stable tower with the largest block at the bottom."
```

CEM search planning
```
python -m vlm_stacking.run_ravens_sim \
  --planner cem --cem-samples 24 --cem-elites 5 --cem-iterations 3 \
  --instruction "Build a stable tower with the largest block at the bottom."
```
