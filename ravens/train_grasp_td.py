"""Train a compact QT-Opt-style critic from sequential grasp replay."""

import os

from absl import app
from absl import flags
import numpy as np

from ravens.grasping.replay import GraspReplayDataset
from ravens.grasping.selection import component_grasp_candidates
from ravens.models.grasp_candidate_q import GraspCandidateQModel
from ravens.dataset import PIXEL_SIZE
import tensorflow as tf


flags.DEFINE_string('replay_dir', './data/rgbd-grasp-replay', '')
flags.DEFINE_string('root_dir', './runs/grasp-qt-opt', '')
flags.DEFINE_integer('steps', 1500, '')
flags.DEFINE_integer('batch_size', 16, '')
flags.DEFINE_integer('save_interval', 250, '')
flags.DEFINE_integer('target_update_interval', 100, '')
flags.DEFINE_integer('crop_size', 48, '')
flags.DEFINE_float('learning_rate', 1e-4, '')
flags.DEFINE_float('gamma', 0.9, '')
flags.DEFINE_float('min_surface_height', 0.015, '')
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


def next_value(model, transition):
  if transition.done:
    return 0.0
  mask = transition.next_image[..., 3] >= FLAGS.min_surface_height
  candidates = component_grasp_candidates(mask, PIXEL_SIZE)
  if not candidates:
    return 0.0
  ranked = model.score_candidates(
      transition.next_image, candidates, mc_samples=1,
      uncertainty_weight=0.0)
  return max((candidate['probability'] for candidate in ranked), default=0.0)


def main(unused_argv):
  del unused_argv
  if not 0 <= FLAGS.gamma <= 1:
    raise ValueError('--gamma must be in [0, 1]')
  configure_tensorflow()
  dataset = GraspReplayDataset(FLAGS.replay_dir)
  if not len(dataset):
    raise ValueError(f'grasp replay dataset is empty: {dataset.path}')
  transitions = [dataset.load(index) for index in range(len(dataset))]
  model = GraspCandidateQModel(
      image_channels=transitions[0].image.shape[2],
      crop_size=FLAGS.crop_size, learning_rate=FLAGS.learning_rate)
  target = GraspCandidateQModel(
      image_channels=transitions[0].image.shape[2],
      crop_size=FLAGS.crop_size, learning_rate=FLAGS.learning_rate)
  target.model.set_weights(model.model.get_weights())
  rng = np.random.default_rng(FLAGS.seed)
  writer = tf.summary.create_file_writer(os.path.join(FLAGS.root_dir, 'logs'))
  print(f'Training on {len(transitions)} sequential transitions.', flush=True)
  for step in range(1, FLAGS.steps + 1):
    indices = rng.choice(
        len(transitions), FLAGS.batch_size,
        replace=len(transitions) < FLAGS.batch_size)
    batch = [transitions[int(index)] for index in indices]
    bootstrapped = np.asarray(
        [next_value(target, transition) for transition in batch],
        dtype=np.float32)
    targets = np.asarray([
        transition.reward + FLAGS.gamma * value
        for transition, value in zip(batch, bootstrapped)
    ], dtype=np.float32)
    targets = np.clip(targets, 0, 1)
    loss = model.train_targets(batch, targets)
    with writer.as_default():
      tf.summary.scalar('train/loss', loss, step=step)
      tf.summary.scalar('train/target_mean', float(np.mean(targets)), step=step)
    if step == 1 or step % 25 == 0:
      print(f'step={step} loss={loss:.5f} '
            f'target={np.mean(targets):.4f}', flush=True)
    if step % FLAGS.target_update_interval == 0:
      target.model.set_weights(model.model.get_weights())
    if step % FLAGS.save_interval == 0 or step == FLAGS.steps:
      path = os.path.join(FLAGS.root_dir, 'checkpoints',
                          f'grasp-qt-opt-{step}.h5')
      model.save(path)
      print(f'Saved {path}', flush=True)


if __name__ == '__main__':
  app.run(main)
