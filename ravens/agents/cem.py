"""Cross-Entropy Method (CEM) planner for pick-and-place stacking.

No training. At every step the agent:
  1. picks the next block that still has to be placed,
  2. searches over where to put it, a = (x, y, yaw), with CEM,
  3. scores each candidate by trying it in a copy of the simulation
     (snapshot -> try -> score -> rewind),
  4. returns the best action, in the same {'pose0', 'pose1'} format the
     oracle and the Transporter agent use.

The search distribution is a diagonal Gaussian N(mean, std^2) over (x, y, yaw).
Each CEM iteration samples candidates, keeps the top `n_elites`, and refits the
Gaussian to them.
"""

import copy

import numpy as np
import pybullet as p

from ravens.utils import utils


class CEMAgent:
  """CEM planner that uses the simulator itself as its world model."""

  def __init__(self, env, task,
               n_samples=24,          # candidates per CEM iteration
               n_elites=5,            # top candidates used to refit
               n_iters=3,             # CEM iterations per action
               init_std=None,         # std of (x, y, yaw); None = by init
               min_std=(0.001, 0.001, 0.02),       # stops early collapse
               smoothing=0.2,         # weight kept on the previous mean/std
               rollout='teleport',    # 'teleport' (fast) or 'full' (arm)
               init='target',         # 'target' (warm start) or 'base' (cold)
               verify_k=3,            # re-check top-k with the real arm

               w_target=10.0,         # weight on distance to goal positions
               w_disturb=20.0,        # weight on moving other blocks
               w_rot=0.5,             # weight on yaw error
               seed=0,
               verbose=True):
    self.env = env
    self.task = task
    self.n_samples = n_samples
    self.n_elites = n_elites
    self.n_iters = n_iters
    if init_std is None:
      # Warm start only needs to correct small errors; cold start must
      # cover the whole base.
      init_std = ((0.01, 0.01, np.pi / 8) if init == 'target'
                  else (0.05, 0.05, np.pi / 4))
    self.init_std = np.float64(init_std)
    self.min_std = np.float64(min_std)
    self.smoothing = smoothing
    self.rollout = rollout
    self.init = init
    self.verify_k = verify_k
    self.w_target = w_target
    self.w_disturb = w_disturb
    self.w_rot = w_rot
    self.rng = np.random.RandomState(seed)
    self.verbose = verbose
    self.n_rollouts = 0  # total simulator rollouts, for reporting cost

    # Optional starting mean for the search, e.g. a plan from a VLM
    # (the "Hybrid VLM + search" row). Set to an (x, y, yaw) array.
    self.init_mean = None

  # -------------------------------------------------------------------------
  # Public API (same signature as the other Ravens agents)
  # -------------------------------------------------------------------------

  def act(self, obs, info, goal=None):  # pylint: disable=unused-argument
    if not self._has_goals():
      return None

    obj_id = self._next_object()
    if obj_id is None:
      return None
    pick_pose = self._pick_pose(obj_id)
    symmetry = self._goal_symmetry(obj_id)

    # Initial search distribution.
    if self.init_mean is not None:
      mean = np.float64(self.init_mean)
    else:
      mean = self._default_mean(obj_id)
    std = self.init_std.copy()

    tried = []  # every (score, candidate) evaluated, for the final check
    for it in range(self.n_iters):
      samples = mean + std * self.rng.randn(self.n_samples, 3)
      if it == 0:
        samples[0] = mean  # always try the starting guess itself
      scores = np.array([self._evaluate(obj_id, pick_pose, a)
                         for a in samples])
      tried += list(zip(scores, samples))

      # Keep the elites and refit the Gaussian.
      elite_idx = np.argsort(scores)[-self.n_elites:]
      elites = samples[elite_idx].copy()

      # Symmetric objects look identical at yaw and yaw + symmetry (90 deg for
      # a cube). Without this, elites split across equivalent angles and their
      # average lands half-way between them, which is the worst yaw.
      # Wrap each elite's yaw to the copy closest to the current mean.
      if symmetry > 0:
        d = (elites[:, 2] - mean[2] + symmetry / 2) % symmetry - symmetry / 2
        elites[:, 2] = mean[2] + d
      new_mean = elites.mean(axis=0)
      new_std = np.maximum(elites.std(axis=0), self.min_std)
      mean = self.smoothing * mean + (1 - self.smoothing) * new_mean
      std = self.smoothing * std + (1 - self.smoothing) * new_std

      if self.verbose:
        print(f'  CEM iter {it + 1}/{self.n_iters}: '
              f'best={scores.max():.3f} mean_elite={scores[elite_idx].mean():.3f} '
              f'std={np.round(std, 3)}')

    # The final mean is usually the most reliable point, so add it.
    tried.append((self._evaluate(obj_id, pick_pose, mean), mean))
    tried.sort(key=lambda t: t[0], reverse=True)
    best_a = tried[0][1]

    # Teleport scoring ignores how the arm actually carries and lowers the
    # block (errors of 4-10 mm, against a 10 mm tolerance). Re-score the top
    # few candidates with the real arm motion and pick the best of those.
    if self.rollout == 'teleport' and self.verify_k > 0:
      top = [a for _, a in tried[:self.verify_k]]
      real = [self._evaluate(obj_id, pick_pose, a, rollout='full')
              for a in top]
      best_a = top[int(np.argmax(real))]
      if self.verbose:
        print(f'  arm check of top {len(top)}: {np.round(real, 3)}')

    return {'pose0': pick_pose, 'pose1': self._place_pose(obj_id, best_a)}

  # -------------------------------------------------------------------------
  # Action construction
  # -------------------------------------------------------------------------

  def _has_goals(self):
    return bool(self.task.goals)

  def _goal_symmetry(self, obj_id):
    """Rotational symmetry of the object (pi/2 for a cube)."""
    return dict(self.task.goals[0][0])[obj_id][0]

  def _next_object(self):
    """First object in the current goal that is not yet on a target."""
    objs, _, targs, _, _, _, _, _ = self.task.goals[0]
    for object_id, (symmetry, _) in objs:
      pose = p.getBasePositionAndOrientation(object_id)
      if not any(self.task.is_match(pose, t, symmetry) for t in targs):
        return object_id
    return None

  def _pick_pose(self, obj_id):
    """Grasp at the object's centre, at the height of its top surface."""
    pos, _ = p.getBasePositionAndOrientation(obj_id)
    top_z = p.getAABB(obj_id)[1][2]
    pick_pose = (np.float64([pos[0], pos[1], top_z]), np.float64([0, 0, 0, 1]))
    return self.task.get_pick_pose(obj_id, pick_pose)

  def _place_pose(self, obj_id, a):
    """Turn a search vector a = (x, y, yaw) into a gripper place pose.

    The height is not searched. It is the top of whatever is under (x, y),
    plus the object's own height, so the block is set down gently. This also
    works for objects of different sizes.
    """
    x, y, yaw = a
    aabb_min, aabb_max = p.getAABB(obj_id)
    height = aabb_max[2] - aabb_min[2]
    z = self._support_height(x, y, exclude=obj_id) + height + 0.002
    quat = utils.eulerXYZ_to_quatXYZW((0, 0, yaw))
    return (np.float64([x, y, z]), np.float64(quat))

  def _support_height(self, x, y, exclude, margin=0.01):
    """Highest top surface of any object whose footprint contains (x, y)."""
    top = 0.0  # tabletop
    for oid in self.env.obj_ids['fixed'] + self.env.obj_ids['rigid']:
      if oid == exclude:
        continue
      # Check every link: the stand, for example, is 3 linked plates.
      for link in range(-1, p.getNumJoints(oid)):
        lo, hi = p.getAABB(oid, link)
        if (lo[0] - margin <= x <= hi[0] + margin and
            lo[1] - margin <= y <= hi[1] + margin):
          top = max(top, hi[2])
    return top

  def _default_mean(self, obj_id):
    """Where the search starts.

    'target': at the nearest free goal spot from the task specification
              (warm start; CEM then corrects for placement/arm errors).
    'base':   at the centre of the base (cold start; CEM must find the spot).

    The yaw is the *gripper* yaw. The gripper picks with yaw 0, so the block
    ends up at (gripper yaw + block's current yaw). To reach a goal yaw, the
    gripper must turn by (goal yaw - block yaw).
    """
    obj_pos, obj_rot = p.getBasePositionAndOrientation(obj_id)
    obj_yaw = utils.quatXYZW_to_eulerXYZ(obj_rot)[2]
    if self.init == 'target':
      pos, rot = self._nearest_free_target(obj_pos)
    else:
      pos, rot = p.getBasePositionAndOrientation(self.env.obj_ids['fixed'][0])
    goal_yaw = utils.quatXYZW_to_eulerXYZ(rot)[2]
    return np.float64([pos[0], pos[1], goal_yaw - obj_yaw])

  def _nearest_free_target(self, obj_pos):
    """Closest goal spot in the current layer that no block occupies yet."""
    objs, _, targs, _, _, _, _, _ = self.task.goals[0]
    free = [t for t in targs if not any(
        self.task.is_match(p.getBasePositionAndOrientation(o), t, s)
        for o, (s, _) in objs)]
    free = free or list(targs)
    return min(free, key=lambda t: np.linalg.norm(
        np.float64(t[0][:2]) - np.float64(obj_pos[:2])))

  # -------------------------------------------------------------------------
  # Evaluate one candidate: snapshot -> try -> score -> rewind
  # -------------------------------------------------------------------------

  def _evaluate(self, obj_id, pick_pose, a, rollout=None):
    rollout = rollout or self.rollout
    self.n_rollouts += 1
    snap = self._snapshot()
    before = {oid: np.float64(p.getBasePositionAndOrientation(oid)[0])
              for oid in self.env.obj_ids['rigid']}
    place_pose = self._place_pose(obj_id, a)

    ok = True
    if rollout == 'full':
      # Execute the real arm motion (slow, most faithful).
      timeout = self.task.primitive(self.env.movej, self.env.movep,
                                    self.env.ee, pick_pose, place_pose)
      ok = not timeout
    else:
      # Teleport the object to where the gripper would release it (fast).
      obj_pose = p.getBasePositionAndOrientation(obj_id)
      obj_in_gripper = utils.multiply(utils.invert(pick_pose), obj_pose)
      new_pose = utils.multiply(place_pose, obj_in_gripper)
      p.resetBasePositionAndOrientation(obj_id, new_pose[0], new_pose[1])
      p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])

    self._settle()
    score = self._score(obj_id, before) if ok else -1e3
    self._restore(snap)
    return score

  def _settle(self, max_steps=960):
    """Let physics run until nothing moves (capped, in case of rolling)."""
    for _ in range(max_steps):
      p.stepSimulation()
      if self.env.is_static:
        break

  def _snapshot(self):
    """PyBullet state AND the Python-side bookkeeping PyBullet doesn't track."""
    return {
        'bullet': p.saveState(),
        'goals': copy.deepcopy(self.task.goals),
        'progress': self.task.progress,
        'rewards': self.task._rewards,  # pylint: disable=protected-access
        'fingers': dict(getattr(self.env.ee, 'finger_positions', {})),
    }

  def _restore(self, snap):
    ee = self.env.ee
    # A grasp constraint created during a rollout is not undone by
    # restoreState, so remove it by hand.
    if getattr(ee, 'contact_constraint', None) is not None:
      p.removeConstraint(ee.contact_constraint)
      ee.contact_constraint = None
      ee.activated = False
    p.restoreState(snap['bullet'])
    p.removeState(snap['bullet'])
    self.task.goals = snap['goals']
    self.task.progress = snap['progress']
    self.task._rewards = snap['rewards']  # pylint: disable=protected-access
    if hasattr(ee, 'finger_positions'):
      ee.finger_positions = snap['fingers']

  # -------------------------------------------------------------------------
  # Score function (swap this out for the heterogeneous-objects task)
  # -------------------------------------------------------------------------

  def _score(self, obj_id, before):
    """Higher is better. A smooth version of the task's 0/1 reward.

      - distance from each goal target to its nearest goal object,
      - how much every *other* block moved (knocking things over is bad),
      - yaw error of the placed object (mod its symmetry),
      - a bonus for each goal object that counts as a match in the real reward.
    """
    objs, matches, targs, _, _, _, _, _ = self.task.goals[0]
    goal_ids = [o[0] for o in objs]
    pos = {oid: np.float64(p.getBasePositionAndOrientation(oid)[0])
           for oid in goal_ids}

    target_cost = 0.0
    for j, targ in enumerate(targs):
      cands = [goal_ids[i] for i in range(len(goal_ids)) if matches[i, j]]
      target_cost += min(np.linalg.norm(pos[o] - np.float64(targ[0]))
                         for o in cands)

    disturb = sum(
        np.linalg.norm(np.float64(p.getBasePositionAndOrientation(oid)[0]) -
                       before[oid])
        for oid in self.env.obj_ids['rigid'] if oid != obj_id)

    symmetry = dict(objs)[obj_id][0]
    obj_pose = p.getBasePositionAndOrientation(obj_id)
    yaw = utils.quatXYZW_to_eulerXYZ(obj_pose[1])[2]
    rot_err = np.pi
    for targ in targs:
      t_yaw = utils.quatXYZW_to_eulerXYZ(targ[1])[2]
      d = abs(yaw - t_yaw) % symmetry if symmetry > 0 else 0.0
      rot_err = min(rot_err, min(d, symmetry - d) if symmetry > 0 else 0.0)

    n_match = sum(
        any(self.task.is_match(p.getBasePositionAndOrientation(o), t, s)
            for t in targs) for o, (s, _) in objs)

    return (-self.w_target * target_cost - self.w_disturb * disturb -
            self.w_rot * rot_err + 1.0 * n_match)


# ===========================================================================
# Goal-free version: CEM must work out how to stack by itself.
# ===========================================================================

# Stand footprint in its own frame (3 plates of 5 x 5 cm in a row).
STAND_HALF_X = 0.025
STAND_HALF_Y = 0.075
UPRIGHT_TOL = np.deg2rad(10)


def stack_metrics(env, tol=0.005):
  """Grade the scene WITHOUT any goal positions.

  A block counts as "stacked" if its centre is over the stand's footprint
  (it rests on the stand or on other blocks, not on the table) and it is
  upright (not tipped over).

  Returns:
    n_stacked: number of stacked blocks.
    n_total: number of blocks.
    height: top of the highest stacked block above the table (metres).
    tilts: dict block id -> tilt angle in radians (0 = upright).
  """
  stand = env.obj_ids['fixed'][0]
  s_pos, s_rot = p.getBasePositionAndOrientation(stand)
  world_to_stand = utils.invert((s_pos, s_rot))
  n_stacked, height, tilts = 0, 0.0, {}
  for oid in env.obj_ids['rigid']:
    pos, rot = p.getBasePositionAndOrientation(oid)
    local = utils.multiply(world_to_stand, (pos, rot))[0]
    over = (abs(local[0]) <= STAND_HALF_X + tol and
            abs(local[1]) <= STAND_HALF_Y + tol and
            pos[2] > s_pos[2])
    # A cube is upright if any of its axes points straight up.
    r = np.float64(p.getMatrixFromQuaternion(rot)).reshape(3, 3)
    tilt = np.arccos(np.clip(np.max(np.abs(r[2, :])), -1, 1))
    tilts[oid] = tilt
    if over and tilt < UPRIGHT_TOL:
      n_stacked += 1
      height = max(height, p.getAABB(oid)[1][2])
  return n_stacked, len(env.obj_ids['rigid']), height, tilts


class FreeStackCEMAgent(CEMAgent):
  """CEM with no goal spots: it is only told the stand is where to build.

  Differences from CEMAgent:
    - which block: the unstacked block closest to the stand,
    - search start: the stand's centre, wide spread (cold start),
    - score: blocks stacked + stability, never distance to a goal spot.
  The stand fits only 3 cubes side by side, so placing all 6 forces CEM to
  discover stacking upwards on its own.
  """

  def __init__(self, env, task, init_std=(0.015, 0.05, np.pi / 4),
               w_center=2.0, w_tilt=2.0, **kwargs):
    kwargs['init'] = 'base'
    super().__init__(env, task, init_std=init_std, **kwargs)
    self.w_center = w_center
    self.w_tilt = w_tilt

  def _has_goals(self):  # no goal list: keep going until all are stacked
    return True

  def _goal_symmetry(self, obj_id):  # cubes: 90-degree symmetry
    del obj_id
    return np.pi / 2

  def _next_object(self):
    """Unstacked block closest to the stand."""
    stand = self.env.obj_ids['fixed'][0]
    s_pos = np.float64(p.getBasePositionAndOrientation(stand)[0])
    world_to_stand = utils.invert(p.getBasePositionAndOrientation(stand))
    todo = []
    for oid in self.env.obj_ids['rigid']:
      pos, rot = p.getBasePositionAndOrientation(oid)
      local = utils.multiply(world_to_stand, (pos, rot))[0]
      if not (abs(local[0]) <= STAND_HALF_X + 0.005 and
              abs(local[1]) <= STAND_HALF_Y + 0.005 and pos[2] > s_pos[2]):
        todo.append((np.linalg.norm(np.float64(pos[:2]) - s_pos[:2]), oid))
    return min(todo)[1] if todo else None

  def _score(self, obj_id, before):
    """Outcome-only score: no goal positions anywhere."""
    n_stacked, _, _, tilts = stack_metrics(self.env)

    disturb = sum(
        np.linalg.norm(np.float64(p.getBasePositionAndOrientation(oid)[0]) -
                       before[oid])
        for oid in self.env.obj_ids['rigid'] if oid != obj_id)

    # Keep the structure compact: distance of the placed block from the
    # stand's centre line (in the stand's frame).
    stand = self.env.obj_ids['fixed'][0]
    world_to_stand = utils.invert(p.getBasePositionAndOrientation(stand))
    local = utils.multiply(world_to_stand,
                           p.getBasePositionAndOrientation(obj_id))[0]
    off_center = abs(local[0])

    return (1.0 * n_stacked - self.w_disturb * disturb -
            self.w_tilt * tilts[obj_id] - self.w_center * off_center)
