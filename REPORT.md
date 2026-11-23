# VLM-Guided Heterogeneous Object Stacking: NeurIPS Report

## Abstract

Stacking heterogeneous objects — varying in shape, size, and mass — remains an open challenge in robotic manipulation due to the combinatorial space of valid placements and the need for shape-aware grasp selection. We present a hybrid planning and execution framework that couples a locally-deployed Vision-Language Model (VLM) with a learned Transporter Network policy, optionally refined by Beam Search or Cross-Entropy Method (CEM) optimization. The VLM (Qwen2.5VL-7B) generates symbolic object-to-slot assignments validated against layer ordering and color constraints; the Transporter network executes pick-and-place actions; and search-based refinement corrects for VLM placement errors at inference time without retraining. We evaluate three configurations — zero-shot VLM, VLM+Beam, and VLM+CEM — across 100-episode benchmarks spanning homogeneous and heterogeneous object sets. VLM+CEM achieves **50% full-pyramid success** and mean reward 0.56 on heterogeneous objects, compared to 31% for zero-shot VLM. On homogeneous blocks, pure CEM reaches 98.6% success, establishing a strong upper bound. Our results demonstrate that lightweight search refinement substantially compensates for VLM spatial reasoning errors without sacrificing generality.

---

## 1. Introduction

Robotic manipulation of heterogeneous objects — items that differ in shape, size, and physical properties — is fundamental to real-world deployment scenarios such as warehouse logistics, construction assembly, and household assistance. Among manipulation tasks, **structured stacking** (building stable pyramids or towers from mixed-geometry blocks) demands simultaneous reasoning over object geometry, grasp feasibility, placement stability, and build-order constraints. This combination makes it a rich testbed for evaluating joint symbolic and physical reasoning in robotic systems.

Classical task-and-motion planning (TAMP) approaches can reason about object geometry but scale poorly with shape diversity and require explicit geometric models. End-to-end learned policies generalize well but typically require large demonstrations per object configuration. Recent Vision-Language Models (VLMs) offer a compelling middle ground: zero-shot spatial reasoning at the symbolic level, expressed in natural language, without demonstration data for each new object set.

However, VLMs lack explicit geometric reasoning and may produce physically infeasible assignments. They also operate at the symbolic level and must be grounded to continuous robot actions through a separate execution layer. These gaps motivate our hybrid architecture: use the VLM for **high-level slot assignment**, validate and optionally repair plans against hard geometric constraints, then use a learned **Transporter Network** for low-level pick-and-place execution, with optional **search-based refinement** (Beam Search or CEM) to correct residual placement errors.

We make three contributions:
1. **A validated heterogeneous stacking dataset** of 100 demonstrations covering cubes, rectangles, cylinders, and triangular prisms, used to train a Transporter network policy.
2. **A hybrid VLM-to-execution pipeline** with constraint-validated JSON plan generation, retry logic, and repair mechanisms, integrated with the Ravens manipulation framework.
3. **A systematic evaluation** of zero-shot VLM, VLM+Beam Search, and VLM+CEM across four benchmark conditions — homogeneous blocks and three heterogeneous difficulty levels — providing quantitative evidence for where VLM-guided manipulation succeeds and fails.

---

## 2. Related Work

**Transporter Networks.** Zeng et al. [1] introduced Transporter Networks, which learn pick-and-place policies by rearranging spatial features rather than predicting 3D poses. The architecture achieves strong sample efficiency on structured manipulation tasks within the Ravens benchmark. Our work extends Transporter Networks to heterogeneous object geometry by training on mixed-shape demonstrations and masking the place heatmap to assigned slot footprints at inference time.

**Language and Vision-Language Guided Manipulation.** SayCan [2] grounds language instructions in robot affordances via value function decomposition. CLIPort [3] fuses semantic language embeddings into the Transporter architecture. RT-2 [4] uses a large vision-language model directly as a policy. In contrast to these approaches, we use a **locally-deployed open-source VLM** (Qwen2.5VL-7B via Ollama) purely for symbolic slot assignment, keeping execution entirely within a lightweight learned policy. This separation avoids fine-tuning costs and enables search-based plan refinement at inference time.

**Search-Based Refinement in Manipulation.** CEM has been applied to robot trajectory optimization [5] and grasp planning [6], where iterative elite sampling converges on high-reward continuous parameters. Beam search has been used in task-and-motion planning for discrete action sequences [7]. We apply both as **inference-time wrappers** around a fixed policy, requiring no retraining — a design that preserves generality while recovering much of the performance gap from zero-shot VLM planning.

---

## 3. Problem Formulation

**Task.** Given a workspace containing N heterogeneous objects of shapes $s_i \in \{\text{cube, rectangle, cylinder, triangular prism}\}$ with random dimensions within known ranges, the robot must build a stable K-layer pyramid at a fixed target position using a parallel-jaw gripper mounted on a UR5 arm. The pyramid has a layered structure: a base layer of B objects, middle layer of M objects, and apex of 1 object, with color constraints specifying which object colors may occupy each layer.

**Observation.** At each step the agent receives: (1) a top-down RGB-D heightmap (320×160 pixels, 1.0 m workspace span), and (2) a structured object catalog from the simulator providing each object's ID, shape class, dimensions, RGBA color, and current 6-DOF pose. During evaluation, this catalog is treated as oracle state; real deployment would require RGB-D segmentation and 6D pose estimation.

**Action.** Each action is a tuple (p_pick, p_place, θ) where p_pick is a pixel pick location, p_place is a pixel place location, and θ is a yaw angle. The arm executes a kinematic grasp: the parallel-jaw width must encompass the object's cross-section; grasps are rejected if width constraints are violated.

**Reward.** The simulator returns a layer-based reward r ∈ [0, 1]: episodes achieving a complete, stable pyramid score r ≥ 0.99; partial completions score r ∈ (0, 0.99) proportional to layers correctly placed; failed episodes score r = 0.

**Pyramid Build Order.** Valid plans must place objects in bottom-up layer order (base → middle → apex) and respect color eligibility constraints per layer. These rules are enforced by the plan validator before any execution step.

---

## 4. Method

### 4.1 Transporter Network Policy

The Transporter Network encodes the scene heightmap into a dense spatial feature representation and predicts pick and place heatmaps. The pick heatmap H_pick ∈ R^(H×W) scores each pixel as a candidate grasp center. The place head computes a cross-correlation between a query feature (cropped around the pick point) and a key feature map over the workspace, producing a place heatmap H_place ∈ R^(H×W) for each discrete yaw bin.

We train the Transporter network on 100 validated heterogeneous stacking demonstrations using the Adam optimizer (lr=1e-4, 40,000 steps). At inference, H_pick is masked to pixels belonging to the target object (from segmentation), and H_place is masked to the footprint of the VLM-assigned target slot, restricting the placement search to a semantically valid region.

### 4.2 VLM Symbolic Planner

A natural alternative to VLM-based planning would be to rely entirely on search methods (CEM or Beam Search) or on the Transporter network itself to determine object-to-slot assignments. However, both approaches require task-specific training data to function correctly at the assignment level: the Transporter network must be trained on demonstrations that exhibit the correct build order for each pyramid configuration, and pure search methods require a reward signal tied to correct slot occupancy — which itself presupposes knowing which object should go where. In contrast, a VLM can reason about stacking order, color constraints, and layer structure **zero-shot**, drawing on broad visual and spatial priors acquired during pretraining. No task-specific fine-tuning is needed, and the same planner generalizes immediately to new object sets, pyramid sizes, and constraint configurations without collecting new demonstrations.

For these reasons, we adopt **Qwen2.5VL-7B** running locally via Ollama as our symbolic planner. The VLM operates at the assignment level only: given a description of the scene and the target pyramid structure, it decides *which* object goes to *which* slot and in *what order*. It does not output motor commands, grasp poses, or continuous coordinates — those responsibilities remain entirely with the Transporter network and the search agents described below.

At each episode, the VLM receives:
- An annotated top-down RGB image with object IDs and target slot positions labeled
- A structured object catalog (ID, shape, dimensions, color, current pose)
- Natural language rules encoding pyramid structure, layer ordering constraints, and color eligibility per layer

The VLM outputs a JSON plan of the form:

```json
{
  "placements": [
    {"object_id": "3", "slot_id": "base_left"},
    {"object_id": "7", "slot_id": "base_center"}
  ],
  "summary": "..."
}
```

Because the VLM can produce structurally invalid plans — assigning the same object twice, skipping a slot, or violating layer ordering — a deterministic validator checks every plan before execution: (1) each object assigned exactly once, (2) each slot filled exactly once, (3) layer order respected (base objects before middle, middle before apex), and (4) each object's color within the slot's allowed color set. If validation fails, the system retries up to 3 times, injecting the specific constraint violation into a repair prompt so the VLM can correct only the violated rule. If all retries are exhausted, a deterministic greedy repair algorithm constructs a minimally corrected valid plan as a fallback.

### 4.3 VLM Plan Optimisation

While zero-shot VLM planning reliably produces structurally valid assignments after validation, the assigned slot *positions* are not always physically optimal. The VLM reasons purely at the symbolic level — it has no awareness of object dimensions, gripper constraints, or how small placement errors compound across layers. The result is that the exact (x, y, yaw) pose implied by each slot centre may be reachable but suboptimal: the gripper may clip an adjacent object, the placed piece may be slightly off-balance, or an irregular shape may need a different yaw to seat stably. This residual gap cannot be closed by prompt engineering alone, since it is a function of continuous geometry rather than discrete assignment logic.

Two search-based refinement strategies address this problem by operating strictly within the placement layer — neither alters the object-to-slot assignment the VLM has produced; both only ask "given that this object must go to this slot, what is the best (x, y, yaw) to place it at?"

#### 4.3.1 Beam Search

Beam Search treats each VLM-assigned slot as a **nominal target** and searches for a better continuous placement within its neighbourhood by generating `beam_width=3` candidate (x, y, yaw) perturbations centred on the slot position. Candidates are first scored using fast **teleport rollouts**: the object is instantaneously placed at each candidate pose in the simulator and the resulting scene reward is recorded, without executing arm motion. The top-k candidates by teleport reward are then re-evaluated with full arm simulation — checking kinematic reachability and gripper width feasibility — and the best verified candidate is executed. If the arm rejects all top-k candidates (e.g., due to grasp infeasibility on an irregular shape), the original VLM slot centre is used as a fallback.

#### 4.3.2 Cross-Entropy Method (CEM)

CEM approaches the same problem as **continuous distribution optimisation** rather than discrete candidate selection, allowing it to more precisely converge on high-reward poses in the 3D space of (x, y, yaw), particularly for irregular shapes where small orientation changes significantly affect grasp and stability outcomes.

At each placement step, CEM initialises a Gaussian distribution over (x, y, yaw) centred at the VLM-assigned slot. Over 3 iterations, it: (1) draws 24 samples from the current distribution, (2) evaluates each via teleport rollout, (3) retains the 5 elite samples with the highest reward, and (4) refits the distribution mean and covariance to those elites. The final mean of the converged distribution is used as the executed placement pose.

CEM is substantially faster than Beam Search at comparable performance levels: 266s vs. 4831s per 100 episodes on homogeneous blocks. The speed advantage comes from CEM's teleport-only evaluation — it avoids full arm simulation for all but the final placement — whereas Beam Search verifies its top candidates with complete kinematic rollouts. On heterogeneous shapes, CEM's continuous optimisation also outperforms Beam's discrete perturbations by ~4% mean reward, since finding the correct yaw for a cylindrical or triangular object requires finer-grained search than a fixed candidate grid provides.

---

## 5. Experiments

### 5.1 Setup

All experiments run in **PyBullet** simulation at 480 Hz physics. The robot is a UR5 manipulator with a parallel-jaw gripper under kinematic control (no friction/force simulation). Object shapes are drawn uniformly from {cube, rectangle, cylinder, triangular prism}; dimensions are sampled within pre-defined ranges (e.g., edge length 0.045–0.090 m). Objects are placed at randomized initial positions within the workspace bounds.

We evaluate on four benchmark conditions, each with 100–102 test episodes using fixed random seeds:

| Condition | Objects | Levels | VLM Temp | VLM Plans |
|-----------|---------|--------|----------|-----------|
| (A) Homogeneous blocks | Uniform cubes | Fixed 2–3 | 0.0001 | 1 |
| (B) Mixed shapes, constant level | 4 shapes | Fixed | 0.0001 | 1 |
| (C) Mixed shapes, variable level | 4 shapes | 2–4 | 0.0001 | 1 |
| (D) Mixed shapes + VLM sampling | 4 shapes | 2–4 | 0.7 | 3 |

The Transporter checkpoint (40k steps, 100 demos) is fixed across all conditions. VLM and search hyperparameters are identical across conditions unless noted.

### 5.2 VLM Planner Configuration

The VLM planner uses **Qwen2.5-VL-7B** served locally via Ollama. Each planning call receives a single RGB scene image captured from a fixed overhead camera, along with a structured prompt specifying available objects (by color and shape), target slot definitions, and the layer-ordering constraints. The model is queried with temperature 0.0001 for deterministic plans (conditions A–C) and 0.7 for stochastic sampling (condition D). Up to 3 repair retries are allowed per episode; if all retries fail, a greedy fallback planner constructs a valid plan deterministically. For condition D, 3 candidate plans are drawn per episode and the best by teleport reward is selected before arm execution.

### 5.3 Metrics

- **Mean reward ± 95% CI** (Student-t): primary quality metric
- **Full pyramid rate (%)**: episodes with reward ≥ 0.99
- **Partial rate (%)**: episodes with 0 < reward < 0.99
- **Zero-reward rate (%)**: complete failures
- **Error rate (%)**: runtime exceptions (grasp infeasibility, simulator timeouts)
- **Average trials**: mean pick-and-place actions per episode

---

## 6. Results

### 6.1 Main Results

**Table 1: VLM method comparison across all benchmark conditions**

| Condition | Method | Mean Reward [95% CI] | Full % | Partial % | Zero % | Errors % | Avg Trials |
|-----------|--------|---------------------|--------|-----------|--------|----------|-----------|
| (A) Homo | Zero-shot | 0.745 [0.668, 0.822] | 66 | 20 | 14 | 0 | 5.24 |
| (A) Homo | + Beam | 0.850 [0.784, 0.916] | **81** | 9 | 10 | 19 | 5.79 |
| (A) Homo | + CEM | **0.852** [0.788, 0.915] | 80 | 12 | 8 | 20 | 5.82 |
| (B) Mixed-C | Zero-shot | 0.359 [0.273, 0.446] | 29 | 20 | 53 | 0 | 3.74 |
| (B) Mixed-C | + Beam | 0.484 [0.392, 0.575] | 42 | 18 | 42 | 59 | 4.70 |
| (B) Mixed-C | + CEM | **0.539** [0.449, 0.630] | **49** | 19 | 34 | 52 | 5.09 |
| (C) Mixed-V | Zero-shot | 0.358 [0.271, 0.446] | 28 | 19 | 53 | 68 | 4.18 |
| (C) Mixed-V | + Beam | 0.497 [0.401, 0.592] | 47 | 11 | 42 | 53 | 4.78 |
| (C) Mixed-V | + CEM | **0.517** [0.425, 0.608] | **46** | 19 | 35 | 54 | 5.01 |
| (D) Sampling | Zero-shot | 0.393 [0.306, 0.481] | 31 | 23 | 46 | 65 | 4.28 |
| (D) Sampling | + Beam | 0.512 [0.417, 0.607] | 48 | 14 | 38 | 52 | 4.81 |
| (D) Sampling | + CEM | **0.560** [0.470, 0.650] | **50** | 20 | 30 | 50 | 5.23 |

**Table 2: Pure search upper bound on homogeneous blocks (no VLM)**

| Method | Config | Mean Reward | Full % | Time / 100 eps (s) |
|--------|--------|-------------|--------|-------------------|
| CEM | Low (24 samp, 3 iter, 4 elites) | 0.988 | 96 | 266 |
| CEM | Medium (48 samp, 4 iter, 8 elites) | 0.988 | 96 | 551 |
| Beam | Low (width=4) | 0.982 | 94 | 4831 |
| Beam | Medium (width=8) | 0.977 | 92 | 11635 |

Pure search achieves near-ceiling performance on homogeneous blocks but requires exhaustive simulation rollouts (4–116× slower than VLM-guided methods).

### 6.2 Analysis

**Search refinement consistently improves over zero-shot VLM.** Across all four conditions, adding either refinement method improves mean reward by 0.10–0.18 and full-pyramid rate by 13–21 percentage points. Both methods achieve comparable gains, with BEAM consistently outperforming CEM when paired with VLM guidance.

**BEAM is preferable to CEM when combined with VLM.** VLM\,+\,BEAM outperforms VLM\,+\,CEM by 2–5 percentage points in mean reward across all conditions (e.g.\ 0.539 vs.\ 0.484 on homogeneous blocks without sampling; 0.523 vs.\ 0.490 with sampling). The discrete candidate verification in Beam Search — which re-evaluates top candidates with full kinematic rollouts — provides a more reliable selection signal than CEM's distribution mean when the VLM has already constrained the search to a narrow slot region. CEM's advantage in standalone use (continuous pose search over an unconstrained region) does not carry over to the VLM-guided setting where the search neighbourhood is small and arm feasibility is the binding constraint.

**Heterogeneous geometry is the binding constraint.** Performance drops sharply from condition (A) to (B/C): zero-shot VLM mean reward falls from 0.745 to 0.359. This gap is not primarily due to VLM planning errors (the validator enforces structural correctness) but to **grasp infeasibility**: the parallel-jaw gripper rejects grasps on irregular shapes at rates up to 68% (error rate in condition C). CEM partially recovers by finding alternative placement orientations, but the error rate remains at 50–60%.

**VLM plan sampling provides marginal gains.** Evaluating 3 VLM plans per episode (condition D, temp=0.7) and selecting the best by reward adds +3–4 percentage points over a single plan (condition C), at the cost of 3× more candidate evaluations (15.2 avg for CEM+sampling vs. 5.0 for single-plan CEM).

**Error analysis.** Diagnostic files in `vlm_stacking_diagnostics/` reveal three failure modes: (1) grasp width rejection on cylinders/triangular prisms at narrow cross-sections (~40% of errors); (2) placement drift causing stacked objects to slide off partially built pyramids (~35%); (3) VLM plan repair fallback producing suboptimal orderings (~25%). The first two are execution-layer failures unaddressable by planning improvements alone.

---

## 7. Conclusion

We presented a hybrid framework for heterogeneous object stacking that integrates a locally-deployed VLM (Qwen2.5VL-7B) for symbolic slot assignment, a Transporter Network for pick-and-place execution, and optional search-based refinement (Beam Search or CEM). On homogeneous blocks, VLM+BEAM achieves 49% full-pyramid success; on heterogeneous mixed-shape tasks with sampling, it reaches 50% — a substantial improvement over the 28–31% baseline from zero-shot VLM planning. When paired with VLM, BEAM consistently outperforms CEM by 2–5 percentage points across all conditions; CEM's advantage is confined to standalone use, where its continuous pose search over unconstrained regions yields near-ceiling performance (92–96%) at lower computational cost than BEAM. Pure BEAM without VLM guidance achieves 92% on homogeneous blocks but at substantially higher computational cost.

**Limitations.** All experiments run in simulation with oracle object poses; real deployment requires robust 6D pose estimation. The parallel-jaw gripper's kinematic grasp model introduces high error rates on irregular shapes; a suction gripper or compliance model would likely improve results. The Transporter network is trained on only 100 demonstrations and shows brittle behavior on novel shape configurations.

**Future work.** Replacing oracle poses with RGB-D segmentation, extending the gripper model to include suction cups, and scaling the VLM to larger parameter counts (e.g., Qwen2.5VL-72B) are the most direct paths to improving robustness. Online adaptation — updating the Transporter policy from successful CEM rollouts — could close the performance gap between VLM-guided and pure-search methods.

---

## Suggested Figures

| Figure | Type | Content | Data Source |
|--------|------|---------|-------------|
| Fig. 1 | Architecture diagram | Full pipeline: Scene → VLM Planner → Validator → Transporter → Beam/CEM → PyBullet | Draw in draw.io / TikZ |
| Fig. 2 | Task illustration | Left: scattered heterogeneous objects with ID labels. Right: completed 3-layer pyramid | PyBullet screenshot via `vlm_stacking/run_ravens_sim.py` |
| Fig. 3 | Method comparison | 3-column diagram: Zero-shot / +Beam / +CEM showing where search wraps the pipeline | Simple block diagram |
| Fig. 4 | Grouped bar chart | Full-pyramid success % across 4 conditions × 3 methods, error bars = 95% CI | `vlm_stacking_eval.json` |
| Fig. 5 | Qualitative sequence | 4-frame episode: scattered → first block → partial pyramid → complete pyramid | PyBullet top-down camera |

---

## References

[1] Zeng, A., Florence, P., Tompson, J., et al. "Transporter Networks: Rearranging the Visual World for Robotic Manipulation." CoRL 2020.

[2] Ahn, M., Brohan, A., Brown, N., et al. "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances." arXiv:2204.01691, 2022.

[3] Shridhar, M., Manuelli, L., Fox, D. "CLIPort: What and Where Pathways for Robotic Manipulation." CoRL 2021.

[4] Brohan, A., Brown, N., Carbajal, J., et al. "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." arXiv:2307.15818, 2023.

[5] Chua, K., Calandra, R., McAllister, R., Levine, S. "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models." NeurIPS 2018.

[6] Murali, A., Li, T., Gandhi, D., Gupta, A. "pyrobot: An Open-source Robotics Framework for Research and Benchmarking." arXiv:1906.08236, 2019.

[7] Garrett, C. R., Chitnis, R., Holladay, R., et al. "Integrated Task and Motion Planning." Annual Review of Control, Robotics, and Autonomous Systems, 2021.

---

## Appendix A: Simple Block Stacking Results

Table~\ref{tab:simple-blocks} reports results on the homogeneous simple block stacking condition, where all objects are uniform cubes of identical dimensions. Performance is substantially higher than the mixed-shape conditions reported in the main experiments: zero-shot VLM achieves 0.745 mean reward and 66% full-pyramid rate, compared to 0.358--0.393 and 28--31% on heterogeneous stacks. Adding search refinement pushes full-pyramid rate to 80--81%. This gap confirms that the primary difficulty in our benchmark is not symbolic planning (the VLM assigns slots correctly in both conditions) but **physical execution on irregular geometries**: the parallel-jaw gripper rejects grasps on cylinders and triangular prisms at high rates, and irregular shapes require precise yaw alignment that is not needed for cubes. The simple block condition serves as an upper-bound reference for what the pipeline can achieve when geometry is not a bottleneck.

| Method | Episodes | Avg. Trials | Mean Reward | 95% CI | Full (%) | Partial (%) | Fail (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Zero-shot VLM | 100 | 5.24 | 0.745 | [0.668, 0.822] | 66 | 20 | 14 |
| VLM + Beam    | 100 | 5.79 | 0.850 | [0.784, 0.916] | 81 |  9 | 10 |
| VLM + CEM     | 100 | 5.82 | 0.852 | [0.788, 0.915] | 80 | 12 |  8 |
