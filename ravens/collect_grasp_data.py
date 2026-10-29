"""Collect balanced RGB-D grasp attempts in the Ravens simulator.

The collector uses simulator object poses only to sample useful positive and
negative candidates. The saved model input is the fused RGB-D heightmap; the
label is the grasp outcome observed after executing the parallel-jaw primitive.
"""

import os

from absl import app
from absl import flags
import numpy as np
import pybullet as p
from ravens import tasks
from ravens.dataset import BOUNDS
from ravens.dataset import CAMERA_CONFIG
from ravens.dataset import PIXEL_SIZE
from ravens.environments.environment import Environment
from ravens.grasping.dataset import GraspDataset
from ravens.grasping.dataset import GraspSample
from ravens.tasks.grippers import ParallelJaw
from ravens.utils import utils


flags.DEFINE_string('assets_root', './ravens/environments/assets/', '')
flags.DEFINE_string('output_dir', './data/rgbd-grasp-attempts', '')
flags.DEFINE_string('task', 'stack-block-pyramid-parallel-jaw', '')
flags.DEFINE_integer('attempts', 1000, '')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_float('positive_fraction', 0.5, '')
flags.DEFINE_float('positive_std_m', 0.008, '')
flags.DEFINE_float('negative_min_m', 0.045, '')
flags.DEFINE_float('negative_max_m', 0.10, '')
flags.DEFINE_bool('disp', False, '')
FLAGS = flags.FLAGS


def fused_heightmap(obs):
  color, height = utils.get_fused_heightmap(
      obs, CAMERA_CONFIG, BOUNDS, PIXEL_SIZE)
  return np.concatenate((color,
                         height[..., None],
                         height[..., None],
                         height[..., None]), axis=2).astype(np.float32)


def sample_offset(rng, near):
  if near:
    return rng.normal(0, FLAGS.positive_std_m, size=2)
  radius = rng.uniform(FLAGS.negative_min_m, FLAGS.negative_max_m)
  angle = rng.uniform(0, 2 * np.pi)
  return radius * np.array([np.cos(angle), np.sin(angle)])


def candidate(env, rng):
  object_id = int(rng.choice(env.obj_ids['rigid']))
  lo, hi = p.getAABB(object_id)
  center = np.mean(np.asarray([lo, hi]), axis=0)
  proposed_positive = bool(rng.random() < FLAGS.positive_fraction)
  xy = center[:2] + sample_offset(rng, proposed_positive)
  xy[0] = np.clip(xy[0], BOUNDS[0, 0] + 0.01, BOUNDS[0, 1] - 0.01)
  xy[1] = np.clip(xy[1], BOUNDS[1, 0] + 0.01, BOUNDS[1, 1] - 0.01)
  yaw = float(rng.uniform(0, np.pi))
  dimensions = np.asarray(hi) - np.asarray(lo)
  width = float(np.clip(min(dimensions[:2]) + 0.008,
                        0.015, ParallelJaw.OPEN_INNER_WIDTH))
  pick_position = (float(xy[0]), float(xy[1]), float(hi[2]))
  rotation = utils.eulerXYZ_to_quatXYZW((0, 0, yaw))
  pose0 = (pick_position, rotation)

  # Put successful objects back near their original position. Every attempt
  # starts from a fresh reset, so placement reward is irrelevant here.
  place_position = (float(center[0]), float(center[1]), float(hi[2]))
  pose1 = (place_position, rotation)
  return object_id, proposed_positive, width, pose0, pose1


def main(unused_argv):
  del unused_argv
  if FLAGS.task not in tasks.names:
    raise ValueError(f'unknown task: {FLAGS.task}')
  dataset = GraspDataset(FLAGS.output_dir)
  rng = np.random.default_rng(FLAGS.seed)
  env = Environment(FLAGS.assets_root, disp=FLAGS.disp, hz=480)
  task = tasks.names[FLAGS.task]()
  if task.ee is not ParallelJaw:
    raise ValueError('grasp collection currently requires a ParallelJaw task')
  task.mode = 'train'

  try:
    for attempt in range(FLAGS.attempts):
      seed = FLAGS.seed + attempt
      np.random.seed(seed)
      env.seed(seed)
      env.set_task(task)
      obs = env.reset()
      object_id, proposed_positive, width, pose0, pose1 = candidate(env, rng)
      image = fused_heightmap(obs)
      pixel = np.asarray(utils.xyz_to_pix(
          pose0[0], BOUNDS, PIXEL_SIZE), dtype=np.int32)
      _, _, _, info = env.step({'pose0': pose0, 'pose1': pose1})
      success = bool(task.primitive.last_grasp)
      metadata = {
          'attempt': attempt,
          'seed': seed,
          'task': FLAGS.task,
          'object_id': object_id,
          'proposed_positive': proposed_positive,
          'timeout': bool(task.primitive.last_timeout),
          'reward_info_keys': sorted(str(key) for key in info),
      }
      path = dataset.add(GraspSample(
          image=image,
          pixel=pixel,
          yaw=float(utils.quatXYZW_to_eulerXYZ(pose0[1])[2]),
          width=width,
          success=success,
          metadata=metadata))
      print(f'{attempt + 1}/{FLAGS.attempts}: success={success} '
            f'pixel={tuple(pixel)} file={os.path.basename(path)}', flush=True)
  finally:
    env.close()

  print(f'Saved {len(dataset)} attempts to {dataset.path}')
  print(f'Observed success rate: {dataset.success_rate():.3f}')


if __name__ == '__main__':
  app.run(main)
