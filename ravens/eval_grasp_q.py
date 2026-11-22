"""Evaluate a grasp-Q checkpoint on held-out executed attempts."""

import json
import os

from absl import app
from absl import flags
import numpy as np
from ravens.grasping.dataset import GraspDataset
from ravens.grasping.encoding import GraspActionSpace
from ravens.grasping.metrics import binary_grasp_metrics
from ravens.models.grasp_q import GraspQModel
import tensorflow as tf


flags.DEFINE_string('data_dir', './data/rgbd-grasp-attempts-validation', '')
flags.DEFINE_string('checkpoint', None, 'Grasp-Q .h5 checkpoint to evaluate.')
flags.DEFINE_string('output', '', 'Optional JSON metrics path.')
flags.DEFINE_integer('max_samples', 0, '0 evaluates all samples.')
flags.DEFINE_integer('n_rotations', 16, '')
flags.DEFINE_integer('n_widths', 5, '')
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


def main(unused_argv):
  del unused_argv
  if not FLAGS.checkpoint:
    raise ValueError('--checkpoint is required')
  configure_tensorflow()
  dataset = GraspDataset(FLAGS.data_dir)
  if not len(dataset):
    raise ValueError(f'no .npz grasp samples found in {dataset.path}')
  count = min(FLAGS.max_samples, len(dataset)) if FLAGS.max_samples else len(dataset)
  action_space = GraspActionSpace(
      n_rotations=FLAGS.n_rotations,
      n_widths=FLAGS.n_widths,
      min_width=FLAGS.min_width,
      max_width=FLAGS.max_width)
  model = GraspQModel(dataset.load(0).image.shape, action_space)
  model.load(FLAGS.checkpoint)
  labels = []
  probabilities = []
  for index in range(count):
    sample = dataset.load(index)
    probability = model.score(
        sample.image, sample.pixel, sample.yaw, sample.width)
    labels.append(int(sample.success))
    probabilities.append(probability)
    print(f'{index + 1}/{count}: label={int(sample.success)} '
          f'probability={probability:.4f}', flush=True)

  metrics = binary_grasp_metrics(labels, probabilities)
  metrics['checkpoint'] = os.path.abspath(FLAGS.checkpoint)
  metrics['data_dir'] = dataset.path
  print(json.dumps(metrics, indent=2, sort_keys=True))
  if FLAGS.output:
    output = os.path.abspath(FLAGS.output)
    output_dir = os.path.dirname(output)
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as stream:
      json.dump(metrics, stream, indent=2, sort_keys=True)
      stream.write('\n')
    print(f'Saved {output}')


if __name__ == '__main__':
  app.run(main)
