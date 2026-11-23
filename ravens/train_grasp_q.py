"""Train a dense RGB-D grasp Q model from executed grasp attempts."""

import os

from absl import app
from absl import flags
import numpy as np
from ravens.grasping.dataset import GraspDataset
from ravens.grasping.encoding import GraspActionSpace
from ravens.models.grasp_q import GraspQModel
import tensorflow as tf


flags.DEFINE_string('data_dir', './data/rgbd-grasp-attempts', '')
flags.DEFINE_string('root_dir', '.', '')
flags.DEFINE_integer('steps', 40000, '')
flags.DEFINE_integer('save_interval', 1000, '')
flags.DEFINE_integer('n_rotations', 16, '')
flags.DEFINE_integer('n_widths', 5, '')
flags.DEFINE_float('min_width', 0.015, '')
flags.DEFINE_float('max_width', 0.12, '')
flags.DEFINE_float('learning_rate', 1e-4, '')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('gpu', 0, '')
flags.DEFINE_integer('gpu_limit', None, '')
FLAGS = flags.FLAGS


def configure_tensorflow():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if not gpus:
    print('No TensorFlow GPU detected; training on CPU.', flush=True)
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
  configure_tensorflow()
  dataset = GraspDataset(FLAGS.data_dir)
  if not len(dataset):
    raise ValueError(f'no .npz grasp samples found in {dataset.path}')
  first = dataset.load(0)
  action_space = GraspActionSpace(
      n_rotations=FLAGS.n_rotations,
      n_widths=FLAGS.n_widths,
      min_width=FLAGS.min_width,
      max_width=FLAGS.max_width)
  model = GraspQModel(first.image.shape, action_space, FLAGS.learning_rate)
  rng = np.random.default_rng(FLAGS.seed)
  checkpoint_dir = os.path.join(FLAGS.root_dir, 'checkpoints', 'grasp-q')
  log_dir = os.path.join(FLAGS.root_dir, 'logs', 'grasp-q')
  writer = tf.summary.create_file_writer(log_dir)

  print(f'Training on {len(dataset)} attempts; '
        f'success rate={dataset.success_rate():.3f}', flush=True)
  for step in range(1, FLAGS.steps + 1):
    sample = dataset.sample(rng)
    loss = model.train(sample)
    with writer.as_default():
      tf.summary.scalar('train/loss', loss, step=step)
    if step == 1 or step % 100 == 0:
      print(f'step={step} loss={loss:.5f}', flush=True)
    if step % FLAGS.save_interval == 0 or step == FLAGS.steps:
      path = os.path.join(checkpoint_dir, f'grasp-q-ckpt-{step}.h5')
      model.save(path)
      print(f'Saved {path}', flush=True)


if __name__ == '__main__':
  app.run(main)
