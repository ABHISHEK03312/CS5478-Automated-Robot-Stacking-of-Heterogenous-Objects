"""Evaluate the beam search planner. Same scenes and grading as cem_test.py.

Examples:
  python -m ravens.beam_test --objective=target --split=test --n=100
  python -m ravens.beam_test --objective=free --split=test --n=100

Scenes come from the {task}-train / {task}-test folders if they exist (same
seeds as the Transporter baseline), otherwise seeds 2,4,6... / 1,3,5...
"""

import os
import pickle
import time

from absl import app
from absl import flags
import numpy as np

from ravens import tasks
from ravens.agents.beam import BeamSearchAgent
from ravens.agents.beam import stack_metrics
from ravens.dataset import Dataset
from ravens.environments.environment import Environment

flags.DEFINE_string('assets_root', './ravens/environments/assets/', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_string('root_dir', '.', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_string('task', 'stack-block-pyramid-parallel-jaw', '')
flags.DEFINE_enum('split', 'test', ['train', 'test'],
                  'Which scenes: train (for tuning) or test (final).')
flags.DEFINE_integer('n', 10, 'Number of episodes.')
flags.DEFINE_enum('objective', 'target', ['target', 'free'],
                  'target: match the pyramid goal spots. '
                  'free: no goal spots, build a stable stack on the stand.')
flags.DEFINE_integer('beam_width', 3, 'K: sequences kept at each depth.')
flags.DEFINE_integer('horizon', 2, 'How many moves to look ahead.')
flags.DEFINE_integer('n_blocks', 3, 'Candidate blocks tried per expansion.')
flags.DEFINE_integer('n_place', 6, 'Candidate placements per block.')
flags.DEFINE_integer('verify_k', 3,
                     'Re-check this many top first moves with the real arm.')
flags.DEFINE_integer('seed', 0, 'Random seed for placement sampling.')

FLAGS = flags.FLAGS


def get_seeds():
  path = os.path.join(FLAGS.data_dir, f'{FLAGS.task}-{FLAGS.split}')
  if os.path.isdir(path):
    ds = Dataset(path)
    if ds.n_episodes > 0:
      n = min(FLAGS.n, ds.n_episodes)
      print(f'Using {n} seeds from {path}')
      return [ds.load(i, images=False)[1] for i in range(n)]
  first = 2 if FLAGS.split == 'train' else 1
  print(f'No {FLAGS.split} dataset found; using seeds {first}, {first + 2}, ...')
  return [first + 2 * i for i in range(FLAGS.n)]


def main(unused_argv):
  env = Environment(FLAGS.assets_root, disp=FLAGS.disp, hz=480)
  task = tasks.names[FLAGS.task]()
  task.mode = FLAGS.split

  name = (f'{FLAGS.task}-beam-{FLAGS.objective}-{FLAGS.split}-'
          f'K{FLAGS.beam_width}-H{FLAGS.horizon}-v{FLAGS.verify_k}')
  results = []
  for k, seed in enumerate(get_seeds()):
    print(f'Test: {k + 1} (seed {seed})')
    np.random.seed(seed)
    env.seed(seed)
    env.set_task(task)
    obs = env.reset()

    agent = BeamSearchAgent(
        env, task, objective=FLAGS.objective, beam_width=FLAGS.beam_width,
        horizon=FLAGS.horizon, n_blocks=FLAGS.n_blocks,
        n_place=FLAGS.n_place, verify_k=FLAGS.verify_k,
        seed=FLAGS.seed + seed)

    info, total_reward, start = None, 0.0, time.time()
    for _ in range(task.max_steps):
      act = agent.act(obs, info)
      if act is None:
        break
      obs, reward, done, info = env.step(act)
      if FLAGS.objective == 'free':
        n_stacked, n_total, height, _ = stack_metrics(env)
        total_reward = n_stacked / n_total
        print(f'Stacked: {n_stacked}/{n_total}  height: {height * 100:.1f} cm')
        if n_stacked == n_total:
          break
      else:
        total_reward += reward
        print(f'Total Reward: {total_reward:.3f} Done: {done}')
        if done:
          break
    secs = time.time() - start
    print(f'  episode time {secs:.0f}s, rollouts {agent.n_rollouts}')
    if FLAGS.objective == 'free':
      info = {'height': stack_metrics(env)[2]}
    results.append((total_reward, info))

    with open(os.path.join(FLAGS.root_dir, f'{name}.pkl'), 'wb') as f:
      pickle.dump(results, f)

  rewards = np.float32([r for r, _ in results])
  print(f'\n{name}: success rate {np.mean(rewards > 0.99):.0%}, '
        f'mean reward {rewards.mean():.3f} over {len(rewards)} episodes')
  if FLAGS.objective == 'free':
    heights = np.float32([i['height'] for _, i in results])
    print(f'mean stack height {heights.mean() * 100:.1f} cm')


if __name__ == '__main__':
  app.run(main)
