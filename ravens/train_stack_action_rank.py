"""Fine-tune joint-action Q to rank counterfactual CEM candidates."""

import os

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from ravens.grasping.stack_replay import StackActionReplayDataset
from ravens.models.stack_action_q import StackActionQModel


flags.DEFINE_multi_string('replay_shard', [], '')
flags.DEFINE_string('replay_dir', './data/cem-pair-replay', '')
flags.DEFINE_string('root_dir', './runs/stack-action-rank', '')
flags.DEFINE_string('init_checkpoint', '', '')
flags.DEFINE_integer('steps', 1000, '')
flags.DEFINE_integer('batch_size', 16, '')
flags.DEFINE_integer('save_interval', 250, '')
flags.DEFINE_float('learning_rate', 1e-5, '')
flags.DEFINE_float('min_reward_gap', 0.1, '')
flags.DEFINE_float('validation_fraction', 0.2, '')
flags.DEFINE_integer('seed', 0, '')
FLAGS = flags.FLAGS


def preference_pairs(transitions, min_reward_gap):
  groups = {}
  for transition in transitions:
    key = transition.metadata['candidate_group']
    groups.setdefault(key, []).append(transition)
  pairs = []
  for candidates in groups.values():
    for first in candidates:
      for second in candidates:
        if first.reward >= second.reward + min_reward_gap:
          pairs.append((first, second))
  return pairs


def pair_accuracy(model, pairs):
  if not pairs:
    return float('nan')
  preferred = model.predict([pair[0] for pair in pairs])
  rejected = model.predict([pair[1] for pair in pairs])
  return float(np.mean(preferred > rejected))


def main(unused_argv):
  del unused_argv
  paths = list(FLAGS.replay_shard) or [FLAGS.replay_dir]
  datasets = [StackActionReplayDataset(path) for path in paths]
  transitions = [dataset.load(index) for dataset in datasets
                 for index in range(len(dataset))]
  groups = sorted({item.metadata['candidate_group'] for item in transitions})
  split_rng = np.random.default_rng(FLAGS.seed)
  split_rng.shuffle(groups)
  validation_count = max(1, int(len(groups) * FLAGS.validation_fraction))
  validation_groups = set(groups[:validation_count])
  training_transitions = [
      item for item in transitions
      if item.metadata['candidate_group'] not in validation_groups]
  validation_transitions = [
      item for item in transitions
      if item.metadata['candidate_group'] in validation_groups]
  pairs = preference_pairs(training_transitions, FLAGS.min_reward_gap)
  validation_pairs = preference_pairs(
      validation_transitions, FLAGS.min_reward_gap)
  if not pairs or not validation_pairs:
    raise ValueError('no preference pairs satisfy --min_reward_gap')
  tf.keras.utils.set_random_seed(FLAGS.seed)
  model = StackActionQModel(
      image_channels=transitions[0].image.shape[2],
      learning_rate=FLAGS.learning_rate)
  if FLAGS.init_checkpoint:
    model.load(FLAGS.init_checkpoint)
  rng = np.random.default_rng(FLAGS.seed)
  print(f'Training on {len(pairs)} pairs; validating on '
        f'{len(validation_pairs)} held-out decision-group pairs.', flush=True)
  print(f'initial_validation_accuracy='
        f'{pair_accuracy(model, validation_pairs):.4f}', flush=True)
  for step in range(1, FLAGS.steps + 1):
    indices = rng.choice(len(pairs), FLAGS.batch_size,
                         replace=len(pairs) < FLAGS.batch_size)
    preferred = [pairs[int(index)][0] for index in indices]
    rejected = [pairs[int(index)][1] for index in indices]
    loss = model.train_pairs(preferred, rejected)
    if step == 1 or step % 25 == 0:
      print(f'step={step} pair_loss={loss:.5f}', flush=True)
    if step % FLAGS.save_interval == 0 or step == FLAGS.steps:
      path = os.path.join(FLAGS.root_dir, 'checkpoints',
                          f'stack-action-rank-{step}.h5')
      model.save(path)
      print(f'Saved {path}', flush=True)
  print(f'final_training_accuracy={pair_accuracy(model, pairs):.4f}',
        flush=True)
  print(f'final_validation_accuracy='
        f'{pair_accuracy(model, validation_pairs):.4f}', flush=True)


if __name__ == '__main__':
  app.run(main)
