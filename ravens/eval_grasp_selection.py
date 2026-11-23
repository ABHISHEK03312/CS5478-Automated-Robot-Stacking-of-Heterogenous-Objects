"""Compare random, geometry, and learned RGB-D grasp selection."""

import json
import os

from absl import app
from absl import flags
import numpy as np
import pybullet as p
from ravens import tasks
from ravens.dataset import BOUNDS
from ravens.dataset import PIXEL_SIZE
from ravens.environments.environment import Environment
from ravens.grasping.dataset import GraspDataset
from ravens.grasping.encoding import GraspActionSpace
from ravens.grasping.observations import fused_heightmap
from ravens.grasping.selection import component_grasp_candidates
from ravens.grasping.selection import filter_supported_candidates
from ravens.grasping.selection import rank_grasp_candidates
from ravens.models.grasp_q import GraspQModel
from ravens.tasks.grippers import ParallelJaw
from ravens.utils import utils
import tensorflow as tf


flags.DEFINE_string('assets_root', './ravens/environments/assets/', '')
flags.DEFINE_string('checkpoint', None, 'Grasp-Q .h5 checkpoint.')
flags.DEFINE_string('coverage_data_dir', './data/rgbd-grasp-attempts-train',
                    'Training attempts used to mask unsupported action bins.')
flags.DEFINE_string('output', './results/grasp-selection.json', '')
flags.DEFINE_string('task', 'stack-block-pyramid-parallel-jaw', '')
flags.DEFINE_integer('episodes', 20, '')
flags.DEFINE_integer('seed', 30000, '')
flags.DEFINE_float('min_surface_height', 0.015, '')
flags.DEFINE_integer('border', 3, '')
flags.DEFINE_integer('n_rotations', 16, '')
flags.DEFINE_integer('n_widths', 5, '')
flags.DEFINE_integer('min_action_samples', 3, '')
flags.DEFINE_float('min_width', 0.015, '')
flags.DEFINE_float('max_width', 0.12, '')
flags.DEFINE_integer('gpu', 0, '')
flags.DEFINE_integer('gpu_limit', None, '')
FLAGS = flags.FLAGS


def configure_tensorflow():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if not gpus:
    return
  if not 0 <= FLAGS.gpu < len(gpus):
    raise ValueError(f'GPU index {FLAGS.gpu} is unavailable')
  gpu = gpus[FLAGS.gpu]
  tf.config.experimental.set_visible_devices(gpu, 'GPU')
  if FLAGS.gpu_limit is not None:
    tf.config.experimental.set_virtual_device_configuration(
        gpu, [tf.config.experimental.VirtualDeviceConfiguration(
            memory_limit=1024 * FLAGS.gpu_limit)])


def valid_surface_mask(image):
  mask = image[..., 3] >= FLAGS.min_surface_height
  if FLAGS.border:
    mask[:FLAGS.border, :] = False
    mask[-FLAGS.border:, :] = False
    mask[:, :FLAGS.border] = False
    mask[:, -FLAGS.border:] = False
  return mask


def action_coverage(dataset, action_space):
  counts = np.zeros((action_space.n_rotations, action_space.n_widths),
                    dtype=np.int32)
  for index in range(len(dataset)):
    sample = dataset.load(index)
    counts[action_space.rotation_bin(sample.yaw),
           action_space.width_bin(sample.width)] += 1
  return counts, counts >= FLAGS.min_action_samples


def geometry_grasp(env, rng):
  object_id = int(rng.choice(env.obj_ids['rigid']))
  lo, hi = p.getAABB(object_id)
  center = (np.asarray(lo) + np.asarray(hi)) / 2
  dimensions = np.asarray(hi) - np.asarray(lo)
  position = (float(center[0]), float(center[1]), float(hi[2]))
  pixel = tuple(map(int, utils.xyz_to_pix(position, BOUNDS, PIXEL_SIZE)))
  return {
      'pixel': pixel,
      'yaw': float(p.getEulerFromQuaternion(
          p.getBasePositionAndOrientation(object_id)[1])[2] % np.pi),
      'width': float(np.clip(min(dimensions[:2]) + 0.008,
                             0.015, ParallelJaw.OPEN_INNER_WIDTH)),
      'quality': None,
      'object_id': object_id,
      'position': position,
  }


def execute(env, task, image, grasp):
  if 'position' in grasp:
    position = grasp['position']
  else:
    position = utils.pix_to_xyz(
        grasp['pixel'], image[..., 3], BOUNDS, PIXEL_SIZE)
  rotation = utils.eulerXYZ_to_quatXYZW((0, 0, grasp['yaw']))
  pose = (position, rotation)
  env.step({'pose0': pose, 'pose1': pose, 'lift_only': True})
  return {
      'position': [float(value) for value in position],
      'surface_height_m': float(image[grasp['pixel'][0], grasp['pixel'][1], 3]),
      'success': bool(task.primitive.last_lift_success),
      'grasp_attached': bool(task.primitive.last_grasp),
      'lift_height_m': float(task.primitive.last_lift_height),
      'grasped_object_id': task.primitive.last_grasp_object_id,
      'transport_retained': bool(task.primitive.last_transport_retained),
      'timeout': bool(task.primitive.last_timeout),
  }


def scene_signature(env):
  """Return an ID-independent, rounded rigid-object state snapshot."""
  objects = []
  for object_id in env.obj_ids['rigid']:
    position, rotation = p.getBasePositionAndOrientation(object_id)
    dimensions = p.getVisualShapeData(object_id)[0][3]
    objects.append({
        'position': [round(float(value), 6) for value in position],
        'rotation': [round(float(value), 6) for value in rotation],
        'dimensions': [round(float(value), 6) for value in dimensions],
    })
  return sorted(objects, key=lambda item: tuple(item['position']))


def main(unused_argv):
  del unused_argv
  if not FLAGS.checkpoint:
    raise ValueError('--checkpoint is required')
  configure_tensorflow()
  action_space = GraspActionSpace(
      n_rotations=FLAGS.n_rotations,
      n_widths=FLAGS.n_widths,
      min_width=FLAGS.min_width,
      max_width=FLAGS.max_width)
  model = GraspQModel((320, 160, 6), action_space)
  model.load(FLAGS.checkpoint)
  coverage_dataset = GraspDataset(FLAGS.coverage_data_dir)
  if not len(coverage_dataset):
    raise ValueError('coverage dataset is empty')
  action_counts, valid_actions = action_coverage(
      coverage_dataset, action_space)
  if not np.any(valid_actions):
    raise ValueError('no action bin meets --min_action_samples')
  env = Environment(FLAGS.assets_root, disp=False, hz=480)
  task = tasks.names[FLAGS.task]()
  if task.ee is not ParallelJaw:
    raise ValueError('selection evaluation requires a ParallelJaw task')
  task.mode = 'test'
  rng = np.random.default_rng(FLAGS.seed)
  rows = []

  try:
    for episode in range(FLAGS.episodes):
      scene_seed = FLAGS.seed + episode
      for method in ('random', 'geometry', 'grasp_q'):
        np.random.seed(scene_seed)
        env.seed(scene_seed)
        env.set_task(task)
        obs = env.reset()
        image = fused_heightmap(obs)
        state_before_action = scene_signature(env)
        candidates = component_grasp_candidates(
            valid_surface_mask(image), PIXEL_SIZE)
        supported = filter_supported_candidates(
            candidates, action_space, valid_actions)
        if not supported:
          raise ValueError(f'no supported candidates for seed {scene_seed}')
        if method == 'random':
          grasp = dict(supported[int(rng.integers(len(supported)))])
        elif method == 'geometry':
          grasp = geometry_grasp(env, rng)
        else:
          quality = model.forward(image)
          grasp = rank_grasp_candidates(
              quality, candidates, action_space, valid_actions)
        outcome = execute(env, task, image, grasp)
        row = {
            'episode': episode,
            'seed': scene_seed,
            'method': method,
            'pixel': list(grasp['pixel']),
            'yaw': float(grasp['yaw']),
            'width': float(grasp['width']),
            'quality': grasp['quality'],
            'scene_signature': state_before_action,
            **outcome,
        }
        rows.append(row)
        print(f'episode={episode + 1}/{FLAGS.episodes} method={method} '
              f'success={outcome["success"]}', flush=True)
  finally:
    env.close()

  summary = {}
  for method in ('random', 'geometry', 'grasp_q'):
    selected = [row for row in rows if row['method'] == method]
    successes = sum(row['success'] for row in selected)
    summary[method] = {
        'episodes': len(selected),
        'successes': int(successes),
        'success_rate': float(successes / len(selected)),
        'timeouts': int(sum(row['timeout'] for row in selected)),
    }
  report = {
      'checkpoint': os.path.abspath(FLAGS.checkpoint),
      'task': FLAGS.task,
      'seed_start': FLAGS.seed,
      'min_action_samples': FLAGS.min_action_samples,
      'action_counts': action_counts.tolist(),
      'valid_actions': valid_actions.tolist(),
      'summary': summary,
      'episodes': rows,
  }
  print(json.dumps(summary, indent=2, sort_keys=True))
  output = os.path.abspath(FLAGS.output)
  output_dir = os.path.dirname(output)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(output, 'w', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
    stream.write('\n')
  print(f'Saved {output}')


if __name__ == '__main__':
  app.run(main)
