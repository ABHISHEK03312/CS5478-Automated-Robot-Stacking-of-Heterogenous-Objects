"""Train a joint pick-and-place critic on discounted stacking returns."""

import os

from absl import app
from absl import flags
import numpy as np

from ravens.grasping.stack_replay import StackActionReplayDataset
from ravens.models.stack_action_q import StackActionQModel


flags.DEFINE_string('replay_dir', './data/stack-action-replay', '')
flags.DEFINE_multi_string(
    'replay_shard', [],
    'Replay directory; repeat this flag to train from multiple shards.')
flags.DEFINE_string('root_dir', './runs/stack-action-q', '')
flags.DEFINE_integer('steps', 1500, '')
flags.DEFINE_integer('batch_size', 16, '')
flags.DEFINE_integer('save_interval', 250, '')
flags.DEFINE_float('gamma', 0.95, '')
flags.DEFINE_float('learning_rate', 1e-4, '')
flags.DEFINE_integer('seed', 0, '')
FLAGS = flags.FLAGS


def discounted_targets(transitions, gamma):
  groups = {}
  for index, transition in enumerate(transitions):
    key = (transition.metadata['seed'], transition.metadata['episode'])
    groups.setdefault(key, []).append((index, transition))
  targets = np.zeros(len(transitions), dtype=np.float32)
  for episode in groups.values():
    future = 0.0
    for index, transition in sorted(
        episode, key=lambda item: item[1].metadata['step'], reverse=True):
      future = float(transition.reward) + gamma * future
      targets[index] = np.clip(future, 0, 1)
  return targets


def main(unused_argv):
  del unused_argv
  replay_paths = list(FLAGS.replay_shard) or [FLAGS.replay_dir]
  datasets = [StackActionReplayDataset(path) for path in replay_paths]
  transitions = [dataset.load(index) for dataset in datasets
                 for index in range(len(dataset))]
  if not transitions:
    raise ValueError(
        f'stack action replay is empty: {", ".join(replay_paths)}')
  targets = discounted_targets(transitions, FLAGS.gamma)
  model = StackActionQModel(
      image_channels=transitions[0].image.shape[2],
      learning_rate=FLAGS.learning_rate)
  rng = np.random.default_rng(FLAGS.seed)
  print(f'Training on {len(transitions)} stack-action transitions.', flush=True)
  for step in range(1, FLAGS.steps + 1):
    indices = rng.choice(len(transitions), FLAGS.batch_size,
                         replace=len(transitions) < FLAGS.batch_size)
    batch = [transitions[int(index)] for index in indices]
    loss = model.train_targets(batch, targets[indices])
    if step == 1 or step % 25 == 0:
      print(f'step={step} loss={loss:.5f} '
            f'target={np.mean(targets[indices]):.4f}', flush=True)
    if step % FLAGS.save_interval == 0 or step == FLAGS.steps:
      path = os.path.join(FLAGS.root_dir, 'checkpoints',
                          f'stack-action-q-{step}.h5')
      model.save(path)
      print(f'Saved {path}', flush=True)


if __name__ == '__main__':
  app.run(main)
