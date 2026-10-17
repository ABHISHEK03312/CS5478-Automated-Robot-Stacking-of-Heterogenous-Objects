"""Beam search planner for pick-and-place stacking.

No training. At every real step the agent looks several moves ahead:

  depth 1: for each of a few candidate blocks, try a few candidate placements
           (in a practice copy of the scene) -> many possible "first moves"
  keep:    only the K best resulting scenes (the "beam")
  depth 2: from each of those K scenes, try every block x placement again
  keep:    the K best two-move sequences
  ...      repeat up to `horizon` moves ahead

It then executes only the FIRST move of the best sequence for real, and plans
again from the new scene next step ("receding horizon"). This handles both
WHICH block to move (the order) and WHERE to put it, and it can prefer a first
move that sets up good later moves, which CEM (one move ahead) cannot.

Two modes, same as CEM:
  objective='target': match the given pyramid goal spots.
  objective='free':   no goal spots; build a stable stack on the stand.
"""

import copy

import numpy as np
import pybullet as p

from ravens.utils import utils

# Stand footprint in its own frame (3 plates of 5 x 5 cm in a row).
STAND_HALF_X = 0.025
STAND_HALF_Y = 0.075
UPRIGHT_TOL = np.deg2rad(10)


def stack_metrics(env, tol=0.005):
  """Grade the scene WITHOUT goal positions (used by 'free' mode).

  A block is "stacked" if its centre is over the stand's footprint (on the
  stand or on other blocks, not the table) and it is upright.
  Returns (n_stacked, n_total, height_of_top_block, {block: tilt_radians}).
  """
  stand = env.obj_ids['fixed'][0]
  s_pos, s_rot = p.getBasePositionAndOrientation(stand)
  world_to_stand = utils.invert((s_pos, s_rot))
  n_stacked, height, tilts = 0, 0.0, {}
  for oid in env.obj_ids['rigid']:
    pos, rot = p.getBasePositionAndOrientation(oid)
    local = utils.multiply(world_to_stand, (pos, rot))[0]
    over = (abs(local[0]) <= STAND_HALF_X + tol and
            abs(local[1]) <= STAND_HALF_Y + tol and pos[2] > s_pos[2])
    r = np.float64(p.getMatrixFromQuaternion(rot)).reshape(3, 3)
    tilt = np.arccos(np.clip(np.max(np.abs(r[2, :])), -1, 1))
    tilts[oid] = tilt
    if over and tilt < UPRIGHT_TOL:
      n_stacked += 1
      height = max(height, p.getAABB(oid)[1][2])
  return n_stacked, len(env.obj_ids['rigid']), height, tilts


class BeamSearchAgent:
  """Beam search over sequences of (block, placement) moves."""

  def __init__(self, env, task,
               objective='target',  # 'target' or 'free'
               beam_width=3,        # K: sequences kept at each depth
               horizon=2,           # how many moves to look ahead
               n_blocks=3,          # candidate blocks tried per expansion
               n_place=6,           # candidate placements per block
               verify_k=3,          # re-check top first moves with the arm
               place_std=(0.004, 0.004, np.pi / 16),  # spread around spots
               rollout='teleport',  # 'teleport' (fast) or 'full' (arm)
               w_target=10.0,       # weight on distance to goal spots
               seed=0, verbose=True):
    self.env = env
    self.task = task
    self.objective = objective
    self.verify_k = verify_k
    self.rollout = rollout
    self.w_target = w_target
    self.rng = np.random.RandomState(seed)
    self.verbose = verbose
    self.n_rollouts = 0  # total simulated tries, for reporting cost
    self.beam_width = beam_width
    self.horizon = horizon
    self.n_blocks = n_blocks
    self.n_place = n_place
    self.place_std = np.float64(place_std)

  # -------------------------------------------------------------------------
  # Main loop
  # -------------------------------------------------------------------------

  def act(self, obs, info, goal=None):  # pylint: disable=unused-argument
    if not self._candidate_blocks():
      return None

    root = self._snapshot()
    # Each beam entry: (value, [(block, placement), ...], snapshot)
    beams = [(self._value(), [], root)]

    for depth in range(self.horizon):
      children = []
      for value, seq, snap in beams:
        self._restore(snap, keep=True)
        blocks = self._candidate_blocks()
        if not blocks:  # nothing left to place on this branch
          children.append((value, seq, snap))
          continue
        for oid in blocks:
          for a in self._candidate_places(oid):
            self._restore(snap, keep=True)
            self._teleport(oid, self._pick_pose(oid), self._place_pose(oid, a))
            self._settle()
            self.n_rollouts += 1
            children.append((self._value(), seq + [(oid, a)],
                             self._snapshot()))

      # Keep the K best; free the memory of everything else.
      children.sort(key=lambda c: c[0], reverse=True)
      kept = children[:self.beam_width]
      kept_snaps = {id(c[2]) for c in kept}
      for c in children[self.beam_width:]:
        if id(c[2]) not in kept_snaps and c[2] is not root:
          p.removeState(c[2]['bullet'])
      for _, _, snap in beams:
        if snap is not root and id(snap) not in kept_snaps:
          p.removeState(snap['bullet'])
      beams = kept
      if self.verbose:
        print(f'  beam depth {depth + 1}/{self.horizon}: '
              f'{len(children)} options, best={beams[0][0]:.3f}')

    for _, _, snap in beams:
      if snap is not root:
        p.removeState(snap['bullet'])
    self._restore(root, keep=True)

    # Distinct first moves of the best sequences, best first.
    firsts = []
    for _, seq, _ in beams:
      if seq and all(not self._same(seq[0], f) for f in firsts):
        firsts.append(seq[0])
    oid, a = firsts[0]

    # Re-check the top first moves with the real arm motion.
    if self.verify_k > 0 and self.rollout == 'teleport':
      checks = []
      for f_oid, f_a in firsts[:self.verify_k]:
        self._restore(root, keep=True)
        timeout = self.task.primitive(
            self.env.movej, self.env.movep, self.env.ee,
            self._pick_pose(f_oid), self._place_pose(f_oid, f_a))
        self._settle()
        self.n_rollouts += 1
        checks.append(-1e3 if timeout else self._value())
        self._restore(root, keep=True)
      oid, a = firsts[int(np.argmax(checks))]
      if self.verbose:
        print(f'  arm check of top {len(checks)}: {np.round(checks, 3)}')

    self._restore(root)  # rewinds and frees the root snapshot
    return {'pose0': self._pick_pose(oid), 'pose1': self._place_pose(oid, a)}

  @staticmethod
  def _same(m1, m2):
    return m1[0] == m2[0] and np.allclose(m1[1], m2[1])

  # -------------------------------------------------------------------------
  # Candidate moves
  # -------------------------------------------------------------------------

  def _stand(self):
    return p.getBasePositionAndOrientation(self.env.obj_ids['fixed'][0])

  def _dist_to_stand(self, oid):
    s_pos = np.float64(self._stand()[0])
    pos = np.float64(p.getBasePositionAndOrientation(oid)[0])
    return np.linalg.norm(pos[:2] - s_pos[:2])

  def _candidate_blocks(self):
    """Which blocks to consider moving next (up to n_blocks, nearest first)."""
    if self.objective == 'target':
      layer = self._current_layer()
      if layer is None:
        return []
      objs, _, targs = layer[0], layer[1], layer[2]
      todo = [o for o, (s, _) in objs if not any(
          self.task.is_match(p.getBasePositionAndOrientation(o), t, s)
          for t in targs)]
    else:
      todo = [o for o in self.env.obj_ids['rigid'] if not self._on_stand(o)]
    todo.sort(key=self._dist_to_stand)
    return todo[:self.n_blocks]

  def _candidate_places(self, oid):
    """A small set of (x, y, yaw) placements to try for this block.

    target: around each free goal spot of the current layer.
    free:   around natural spots: the 3 stand plates, and the top of every
            block already on the stand.
    A little random spread is added around each spot.
    """
    obj_yaw = utils.quatXYZW_to_eulerXYZ(
        p.getBasePositionAndOrientation(oid)[1])[2]
    spots = []  # list of (x, y, goal_yaw)
    if self.objective == 'target':
      objs, _, targs = self._current_layer()[:3]
      for t in targs:
        if not any(self.task.is_match(p.getBasePositionAndOrientation(o), t, s)
                   for o, (s, _) in objs):
          spots.append((t[0][0], t[0][1],
                        utils.quatXYZW_to_eulerXYZ(t[1])[2]))
    else:
      s_pos, s_rot = self._stand()
      s_yaw = utils.quatXYZW_to_eulerXYZ(s_rot)[2]
      for dy in (-0.05, 0.0, 0.05):
        pt = utils.apply((s_pos, s_rot), (0, dy, 0))
        spots.append((pt[0], pt[1], s_yaw))
      for o in self.env.obj_ids['rigid']:
        if o != oid and self._on_stand(o):
          pos = p.getBasePositionAndOrientation(o)[0]
          spots.append((pos[0], pos[1], s_yaw))

    if not spots:
      return []
    per_spot = max(1, self.n_place // len(spots))
    places = []
    for x, y, g_yaw in spots:
      mean = np.float64([x, y, g_yaw - obj_yaw])
      places.append(mean)  # always include the exact spot
      for _ in range(per_spot - 1):
        places.append(mean + self.place_std * self.rng.randn(3))
    return places

  # -------------------------------------------------------------------------
  # How good is a scene? (used to rank beams)
  # -------------------------------------------------------------------------

  def _value(self):
    if self.objective == 'target':
      return self._value_target()
    return self._value_free()

  def _current_layer(self):
    """First goal layer that still has an unfilled spot (in this scene)."""
    for g in self.task.goals:
      objs, _, targs = g[0], g[1], g[2]
      for t in targs:
        if not any(self.task.is_match(p.getBasePositionAndOrientation(o), t, s)
                   for o, (s, _) in objs):
          return g
    return None

  def _value_target(self):
    """Blocks matched on goal spots, minus distance of each spot to a block."""
    value = 0.0
    for objs, matches, targs, *_ in self.task.goals:
      ids = [o for o, _ in objs]
      for j, t in enumerate(targs):
        cands = [ids[i] for i in range(len(ids)) if matches[i, j]]
        d = min(np.linalg.norm(
            np.float64(p.getBasePositionAndOrientation(o)[0]) -
            np.float64(t[0])) for o in cands)
        value -= self.w_target * d
      value += sum(any(self.task.is_match(
          p.getBasePositionAndOrientation(o), t, s) for t in targs)
                   for o, (s, _) in objs)
    return value

  def _on_stand(self, oid, tol=0.005):
    s_pos, s_rot = self._stand()
    pos, rot = p.getBasePositionAndOrientation(oid)
    local = utils.multiply(utils.invert((s_pos, s_rot)), (pos, rot))[0]
    return (abs(local[0]) <= STAND_HALF_X + tol and
            abs(local[1]) <= STAND_HALF_Y + tol and pos[2] > s_pos[2])

  def _value_free(self):
    """Blocks stacked, minus tilt and spread (no goal spots anywhere)."""
    n_stacked, _, _, tilts = stack_metrics(self.env)
    s_pos, s_rot = self._stand()
    world_to_stand = utils.invert((s_pos, s_rot))
    off = 0.0
    for o in self.env.obj_ids['rigid']:
      if self._on_stand(o):
        local = utils.multiply(world_to_stand,
                               p.getBasePositionAndOrientation(o))[0]
        off += abs(local[0])
    return n_stacked - 2.0 * sum(tilts.values()) - 2.0 * off

  # -------------------------------------------------------------------------
  # Simulator helpers: grasp/place poses, practice tries, save/rewind
  # -------------------------------------------------------------------------

  def _pick_pose(self, obj_id):
    """Grasp at the object's centre, at the height of its top surface."""
    pos, _ = p.getBasePositionAndOrientation(obj_id)
    top_z = p.getAABB(obj_id)[1][2]
    pick_pose = (np.float64([pos[0], pos[1], top_z]), np.float64([0, 0, 0, 1]))
    return self.task.get_pick_pose(obj_id, pick_pose)

  def _place_pose(self, obj_id, a):
    """(x, y, yaw) -> gripper place pose; height = top of what is below."""
    x, y, yaw = a
    lo, hi = p.getAABB(obj_id)
    z = self._support_height(x, y, exclude=obj_id) + (hi[2] - lo[2]) + 0.002
    quat = utils.eulerXYZ_to_quatXYZW((0, 0, yaw))
    return (np.float64([x, y, z]), np.float64(quat))

  def _support_height(self, x, y, exclude, margin=0.01):
    """Highest top surface of anything under (x, y); 0 = table."""
    top = 0.0
    for oid in self.env.obj_ids['fixed'] + self.env.obj_ids['rigid']:
      if oid == exclude:
        continue
      for link in range(-1, p.getNumJoints(oid)):  # the stand has 3 plates
        lo, hi = p.getAABB(oid, link)
        if (lo[0] - margin <= x <= hi[0] + margin and
            lo[1] - margin <= y <= hi[1] + margin):
          top = max(top, hi[2])
    return top

  @staticmethod
  def _teleport(obj_id, pick_pose, place_pose):
    """Move the object to where the gripper would release it (fast try)."""
    obj_pose = p.getBasePositionAndOrientation(obj_id)
    obj_in_gripper = utils.multiply(utils.invert(pick_pose), obj_pose)
    new_pose = utils.multiply(place_pose, obj_in_gripper)
    p.resetBasePositionAndOrientation(obj_id, new_pose[0], new_pose[1])
    p.resetBaseVelocity(obj_id, [0, 0, 0], [0, 0, 0])

  def _settle(self, max_steps=960):
    """Let physics run until nothing moves (capped)."""
    for _ in range(max_steps):
      p.stepSimulation()
      if self.env.is_static:
        break

  def _snapshot(self):
    """PyBullet state AND the task's Python bookkeeping."""
    return {
        'bullet': p.saveState(),
        'goals': copy.deepcopy(self.task.goals),
        'progress': self.task.progress,
        'rewards': self.task._rewards,  # pylint: disable=protected-access
        'fingers': dict(getattr(self.env.ee, 'finger_positions', {})),
    }

  def _restore(self, snap, keep=False):
    """Rewind to a snapshot. keep=True leaves it usable for more rewinds."""
    ee = self.env.ee
    if getattr(ee, 'contact_constraint', None) is not None:
      p.removeConstraint(ee.contact_constraint)  # leftover grasp from a try
      ee.contact_constraint = None
      ee.activated = False
    p.restoreState(snap['bullet'])
    if not keep:
      p.removeState(snap['bullet'])
    self.task.goals = copy.deepcopy(snap['goals'])
    self.task.progress = snap['progress']
    self.task._rewards = snap['rewards']  # pylint: disable=protected-access
    if hasattr(ee, 'finger_positions'):
      ee.finger_positions = dict(snap['fingers'])
