"""Train a candidate-relative ranker on counterfactual CEM outcomes."""

import os

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from ravens.grasping.stack_replay import StackActionReplayDataset
from ravens.models.stack_pair_rank import StackActionPairRankModel


flags.DEFINE_multi_string('pair_replay_shard', [], '')
flags.DEFINE_string('pair_replay_dir', './data/cem-pair-replay', '')
flags.DEFINE_string('root_dir', './runs/stack-pair-rank', '')
flags.DEFINE_string('pair_init_checkpoint', '',
                    'Optional scalar-Q checkpoint for RGB-D initialization.')
flags.DEFINE_integer('steps', 1000, '')
flags.DEFINE_integer('batch_size', 16, '')
flags.DEFINE_integer('save_interval', 250, '')
flags.DEFINE_float('learning_rate', 1e-4, '')
flags.DEFINE_float('min_reward_gap', 0.1, '')
flags.DEFINE_float('validation_fraction', 0.2, '')
flags.DEFINE_integer('pair_seed', 0, '')
flags.DEFINE_integer(
    'pair_max_episode', -1,
    'If non-negative, only load episode indices below this value per shard.')
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
  preferred = [pair[0] for pair in pairs]
  rejected = [pair[1] for pair in pairs]
  preferred_scores = [item.metadata['physics_score'] for item in preferred]
  rejected_scores = [item.metadata['physics_score'] for item in rejected]
  probabilities = model.predict_pairs(
      preferred, rejected, preferred_scores, rejected_scores)
  return float(np.mean(probabilities > 0.5))


def main(unused_argv):
  del unused_argv
  paths = list(FLAGS.pair_replay_shard) or [FLAGS.pair_replay_dir]
  datasets = [StackActionReplayDataset(path) for path in paths]
  transitions = [dataset.load(index) for dataset in datasets
                 for index in range(len(dataset))]
  if FLAGS.pair_max_episode >= 0:
    transitions = [item for item in transitions
                   if item.metadata['episode'] < FLAGS.pair_max_episode]
  if not transitions:
    raise ValueError('no transitions after replay filtering')
  scenes = sorted({item.metadata['seed'] for item in transitions})
  rng = np.random.default_rng(FLAGS.pair_seed)
  rng.shuffle(scenes)
  validation_count = max(1, int(len(scenes) * FLAGS.validation_fraction))
  validation_scenes = set(scenes[:validation_count])
  training = [item for item in transitions
              if item.metadata['seed'] not in validation_scenes]
  validation = [item for item in transitions
                if item.metadata['seed'] in validation_scenes]
  train_pairs = preference_pairs(training, FLAGS.min_reward_gap)
  validation_pairs = preference_pairs(validation, FLAGS.min_reward_gap)
  if not train_pairs or not validation_pairs:
    raise ValueError('no train or validation preference pairs')
  tf.keras.utils.set_random_seed(FLAGS.pair_seed)
  model = StackActionPairRankModel(
      image_channels=transitions[0].image.shape[2],
      learning_rate=FLAGS.learning_rate)
  if FLAGS.pair_init_checkpoint:
    model.load_backbone(FLAGS.pair_init_checkpoint)
  print(f'Training on {len(train_pairs)} pairs; validating on '
        f'{len(validation_pairs)} pairs from {len(validation_scenes)} '
        f'held-out scenes.', flush=True)
  print(f'initial_validation_accuracy='
        f'{pair_accuracy(model, validation_pairs):.4f}', flush=True)
  for step in range(1, FLAGS.steps + 1):
    indices = rng.choice(len(train_pairs), FLAGS.batch_size,
                         replace=len(train_pairs) < FLAGS.batch_size)
    preferred = [train_pairs[int(index)][0] for index in indices]
    rejected = [train_pairs[int(index)][1] for index in indices]
    loss = model.train_pairs(preferred, rejected)
    if step == 1 or step % 25 == 0:
      print(f'step={step} pair_loss={loss:.5f}', flush=True)
    if step % FLAGS.save_interval == 0 or step == FLAGS.steps:
      path = os.path.join(FLAGS.root_dir, 'checkpoints',
                          f'stack-pair-rank-{step}.h5')
      model.save(path)
      print(f'Saved {path}', flush=True)
  print(f'final_training_accuracy={pair_accuracy(model, train_pairs):.4f}',
        flush=True)
  print(f'final_validation_accuracy='
        f'{pair_accuracy(model, validation_pairs):.4f}', flush=True)


if __name__ == '__main__':
  app.run(main)
