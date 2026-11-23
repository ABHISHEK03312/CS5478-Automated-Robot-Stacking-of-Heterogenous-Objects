"""Train candidate-conditioned grasp Q with class-balanced mini-batches."""

import os

from absl import app, flags
import numpy as np
from ravens.grasping.dataset import GraspDataset
from ravens.models.grasp_candidate_q import GraspCandidateQModel
from ravens.train_grasp_q import configure_tensorflow
import tensorflow as tf


flags.DEFINE_string('candidate_data_dir', './data/rgbd-grasp-attempts-train', '')
flags.DEFINE_string('candidate_root_dir', './runs/grasp-candidate-q', '')
flags.DEFINE_integer('candidate_steps', 1500, '')
flags.DEFINE_integer('candidate_batch_size', 16, '')
flags.DEFINE_integer('candidate_save_interval', 250, '')
flags.DEFINE_integer('candidate_crop_size', 48, '')
flags.DEFINE_float('candidate_learning_rate', 1e-4, '')
flags.DEFINE_integer('candidate_seed', 0, '')
FLAGS = flags.FLAGS


def balanced_indices(labels, batch_size, rng):
  """Sample an approximately equal number of successes and failures."""
  labels = np.asarray(labels, dtype=bool)
  positive, negative = np.flatnonzero(labels), np.flatnonzero(~labels)
  if not len(positive) or not len(negative):
    raise ValueError('balanced training requires both successes and failures')
  n_positive = batch_size // 2
  indices = np.concatenate((
      rng.choice(positive, n_positive, replace=True),
      rng.choice(negative, batch_size - n_positive, replace=True)))
  rng.shuffle(indices)
  return indices


def main(unused_argv):
  del unused_argv
  configure_tensorflow()
  dataset = GraspDataset(FLAGS.candidate_data_dir)
  if not len(dataset):
    raise ValueError('candidate grasp dataset is empty')
  samples = [dataset.load(i) for i in range(len(dataset))]
  labels = [sample.success for sample in samples]
  model = GraspCandidateQModel(
      image_channels=samples[0].image.shape[2],
      crop_size=FLAGS.candidate_crop_size,
      learning_rate=FLAGS.candidate_learning_rate)
  rng = np.random.default_rng(FLAGS.candidate_seed)
  writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.candidate_root_dir, 'logs'))
  print(f'Training on {len(samples)} attempts with balanced mini-batches.',
        flush=True)
  for step in range(1, FLAGS.candidate_steps + 1):
    batch = [samples[i] for i in balanced_indices(
        labels, FLAGS.candidate_batch_size, rng)]
    loss = model.train_batch(batch)
    with writer.as_default():
      tf.summary.scalar('train/loss', loss, step=step)
    if step == 1 or step % 100 == 0:
      print(f'step={step} loss={loss:.5f}', flush=True)
    if (step % FLAGS.candidate_save_interval == 0 or
        step == FLAGS.candidate_steps):
      path = os.path.join(FLAGS.candidate_root_dir, 'checkpoints',
                          f'grasp-candidate-q-{step}.h5')
      model.save(path)
      print(f'Saved {path}', flush=True)


if __name__ == '__main__':
  app.run(main)
