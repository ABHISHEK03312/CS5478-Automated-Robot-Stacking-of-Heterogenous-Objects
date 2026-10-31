"""Collect oracle and perturbed pick-and-place transitions with task reward."""

import os

from absl import app
from absl import flags
import numpy as np

from ravens import tasks
from ravens.dataset import BOUNDS
from ravens.dataset import PIXEL_SIZE
from ravens.environments.environment import Environment
from ravens.grasping.observations import fused_heightmap
from ravens.grasping.stack_replay import StackActionReplayDataset
from ravens.grasping.stack_replay import StackActionTransition
from ravens.tasks.grippers import ParallelJaw
from ravens.utils import utils


flags.DEFINE_string('assets_root', './ravens/environments/assets/', '')
flags.DEFINE_string('output_dir', './data/stack-action-replay', '')
flags.DEFINE_string('task', 'stack-block-pyramid-parallel-jaw', '')
flags.DEFINE_integer('episodes', 100, '')
flags.DEFINE_integer('seed', 50000, '')
flags.DEFINE_float('perturb_fraction', 0.5, '')
flags.DEFINE_float('place_std_m', 0.035, '')
flags.DEFINE_float('yaw_std', 0.35, '')
flags.DEFINE_bool('disp', False, '')
FLAGS = flags.FLAGS


def perturb_action(action, rng):
  action = {'pose0': (np.array(action['pose0'][0], copy=True),
                      np.array(action['pose0'][1], copy=True)),
            'pose1': (np.array(action['pose1'][0], copy=True),
                      np.array(action['pose1'][1], copy=True))}
  if rng.random() >= FLAGS.perturb_fraction:
    return action, False
  position = action['pose1'][0]
  position[:2] += rng.normal(0, FLAGS.place_std_m, size=2)
  position[0] = np.clip(position[0], BOUNDS[0, 0], BOUNDS[0, 1])
  position[1] = np.clip(position[1], BOUNDS[1, 0], BOUNDS[1, 1])
  yaw = utils.quatXYZW_to_eulerXYZ(action['pose1'][1])[2]
  yaw += rng.normal(0, FLAGS.yaw_std)
  action['pose1'] = (position, utils.eulerXYZ_to_quatXYZW((0, 0, yaw)))
  return action, True


def main(unused_argv):
  del unused_argv
  dataset = StackActionReplayDataset(FLAGS.output_dir)
  rng = np.random.default_rng(FLAGS.seed)
  env = Environment(FLAGS.assets_root, disp=FLAGS.disp, hz=480)
  task = tasks.names[FLAGS.task]()
  if task.ee is not ParallelJaw:
    raise ValueError('stack replay requires a ParallelJaw task')
  task.mode = 'train'
  oracle = task.oracle(env)
  try:
    for episode in range(FLAGS.episodes):
      scene_seed = FLAGS.seed + episode
      np.random.seed(scene_seed)
      env.seed(scene_seed)
      env.set_task(task)
      obs, info = env.reset(), None
      for step in range(task.max_steps):
        oracle_action = oracle.act(obs, info)
        if oracle_action is None:
          break
        action, perturbed = perturb_action(oracle_action, rng)
        image = fused_heightmap(obs)
        pick_pixel = np.asarray(utils.xyz_to_pix(
            action['pose0'][0], BOUNDS, PIXEL_SIZE), dtype=np.int32)
        if not (0 <= pick_pixel[0] < image.shape[0] and
                0 <= pick_pixel[1] < image.shape[1]):
          print(f'episode={episode + 1}/{FLAGS.episodes} step={step + 1} '
                f'skipped: pick pixel {pick_pixel.tolist()} is outside '
                f'{image.shape[:2]}', flush=True)
          break
        pick_yaw = float(utils.quatXYZW_to_eulerXYZ(action['pose0'][1])[2])
        place_yaw = float(utils.quatXYZW_to_eulerXYZ(action['pose1'][1])[2])
        obs, reward, done, info = env.step(action)
        transition = StackActionTransition(
            image=image, pick_pixel=pick_pixel, pick_yaw=pick_yaw,
            width=0.048, place_position=np.asarray(action['pose1'][0]),
            place_yaw=place_yaw, reward=float(reward),
            next_image=fused_heightmap(obs), done=bool(done),
            metadata={'episode': episode, 'step': step, 'seed': scene_seed,
                      'perturbed': perturbed})
        path = dataset.add(transition)
        print(f'episode={episode + 1}/{FLAGS.episodes} step={step + 1} '
              f'reward={reward:.3f} perturbed={perturbed} '
              f'file={os.path.basename(path)}', flush=True)
        if done:
          break
  finally:
    env.close()
  print(f'Saved {len(dataset)} stack-action transitions to {dataset.path}')


if __name__ == '__main__':
  app.run(main)
