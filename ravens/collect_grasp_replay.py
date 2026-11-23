"""Collect sequential grasp transitions for compact QT-Opt-style training."""

import os

from absl import app
from absl import flags
import numpy as np
import pybullet as p

from ravens import tasks
from ravens.dataset import BOUNDS
from ravens.dataset import PIXEL_SIZE
from ravens.environments.environment import Environment
from ravens.grasping.observations import fused_heightmap
from ravens.grasping.replay import GraspReplayDataset
from ravens.grasping.replay import GraspTransition
from ravens.grasping.selection import component_grasp_candidates
from ravens.tasks.grippers import ParallelJaw
from ravens.utils import utils


flags.DEFINE_string('assets_root', './ravens/environments/assets/', '')
flags.DEFINE_string('output_dir', './data/rgbd-grasp-replay', '')
flags.DEFINE_string('task', 'stack-block-pyramid-parallel-jaw', '')
flags.DEFINE_integer('episodes', 100, '')
flags.DEFINE_integer('max_steps', 12, '')
flags.DEFINE_integer('seed', 40000, '')
flags.DEFINE_float('random_fraction', 0.5, '')
flags.DEFINE_float('random_std_pixels', 12.0, '')
flags.DEFINE_float('min_surface_height', 0.015, '')
flags.DEFINE_bool('disp', False, '')
FLAGS = flags.FLAGS


def choose_candidate(image, rng):
  mask = image[..., 3] >= FLAGS.min_surface_height
  candidates = component_grasp_candidates(mask, PIXEL_SIZE)
  if not candidates:
    return None
  candidate = dict(candidates[int(rng.integers(len(candidates)))])
  if rng.random() < FLAGS.random_fraction:
    offset = np.rint(rng.normal(0, FLAGS.random_std_pixels, size=2)).astype(int)
    pixel = np.asarray(candidate['pixel']) + offset
    pixel[0] = np.clip(pixel[0], 0, image.shape[0] - 1)
    pixel[1] = np.clip(pixel[1], 0, image.shape[1] - 1)
    candidate['pixel'] = tuple(map(int, pixel))
    candidate['yaw'] = float(rng.uniform(0, np.pi))
  return candidate


def main(unused_argv):
  del unused_argv
  dataset = GraspReplayDataset(FLAGS.output_dir)
  rng = np.random.default_rng(FLAGS.seed)
  env = Environment(FLAGS.assets_root, disp=FLAGS.disp, hz=480)
  task = tasks.names[FLAGS.task]()
  if task.ee is not ParallelJaw:
    raise ValueError('grasp replay collection requires a ParallelJaw task')
  task.mode = 'train'
  try:
    for episode in range(FLAGS.episodes):
      scene_seed = FLAGS.seed + episode
      np.random.seed(scene_seed)
      env.seed(scene_seed)
      env.set_task(task)
      obs = env.reset()
      initial_count = len(env.obj_ids['rigid'])
      cleared = set()
      for step in range(FLAGS.max_steps):
        image = fused_heightmap(obs)
        candidate = choose_candidate(image, rng)
        if candidate is None:
          break
        pixel = np.asarray(candidate['pixel'], dtype=np.int32)
        position = utils.pix_to_xyz(pixel, image[..., 3], BOUNDS, PIXEL_SIZE)
        rotation = utils.eulerXYZ_to_quatXYZW((0, 0, candidate['yaw']))
        pose = (position, rotation)
        obs_after, _, _, _ = env.step({
            'pose0': pose, 'pose1': pose, 'lift_only': True})
        object_id = task.primitive.last_grasp_object_id
        success = bool(task.primitive.last_lift_success)
        newly_cleared = success and object_id not in cleared
        if newly_cleared:
          cleared.add(object_id)
          p.resetBasePositionAndOrientation(
              object_id, (0.1, -0.45 + 0.02 * len(cleared), 0.02),
              (0, 0, 0, 1))
          obs_after = env._get_obs()  # pylint: disable=protected-access
        done = (len(cleared) == initial_count or
                step + 1 == FLAGS.max_steps)
        reward = float(newly_cleared) / max(initial_count, 1)
        next_image = fused_heightmap(obs_after)
        path = dataset.add(GraspTransition(
            image=image, pixel=pixel, yaw=float(candidate['yaw']),
            width=float(candidate['width']), reward=reward,
            next_image=next_image, done=done,
            metadata={'episode': episode, 'step': step, 'seed': scene_seed,
                      'success': success, 'object_id': object_id,
                      'cleared': len(cleared)}))
        print(f'episode={episode + 1}/{FLAGS.episodes} step={step + 1} '
              f'reward={reward:.3f} cleared={len(cleared)}/{initial_count} '
              f'file={os.path.basename(path)}', flush=True)
        obs = obs_after
        if done:
          break
  finally:
    env.close()
  print(f'Saved {len(dataset)} transitions to {dataset.path}')


if __name__ == '__main__':
  app.run(main)
