"""Evaluate the CEM planner. Mirrors ravens/test.py, but needs no training.

Example:
  python -m ravens.cem_test --assets_root=./ravens/environments/assets/ \
      --task=stack-block-pyramid-parallel-jaw --n=10

If a test dataset folder exists (e.g. stack-block-pyramid-parallel-jaw-test/),
its seeds are used, so the scenes are exactly the ones the Transporter
baseline is tested on. Otherwise the same odd seeds demos.py uses (1, 3, 5, ...).
"""

import os
import pickle
import time

from absl import app
from absl import flags
import numpy as np

from ravens import tasks
from ravens.agents.cem import CEMAgent
from ravens.agents.cem import FreeStackCEMAgent
from ravens.agents.cem import stack_metrics
from ravens.dataset import Dataset
from ravens.environments.environment import Environment

flags.DEFINE_string('assets_root', './ravens/environments/assets/', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_string('root_dir', '.', '')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_string('task', 'stack-block-pyramid-parallel-jaw', '')
flags.DEFINE_enum('split', 'test', ['train', 'test'],
                  'Which scenes to run on: train (for tuning) or test (final).')
flags.DEFINE_integer('n', 10, 'Number of episodes.')
flags.DEFINE_integer('n_samples', 24, 'CEM candidates per iteration.')
flags.DEFINE_integer('n_elites', 5, 'CEM elites per iteration.')
flags.DEFINE_integer('n_iters', 3, 'CEM iterations per action.')
flags.DEFINE_enum('rollout', 'teleport', ['teleport', 'full'],
                  'How candidates are simulated.')
flags.DEFINE_enum('objective', 'target', ['target', 'free'],
                  'target: place blocks on the given goal spots (pyramid). '
                  'free: no goal spots, CEM works out how to stack.')
flags.DEFINE_enum('init', 'target', ['target', 'base'],
                  'Where CEM starts searching: goal spot (warm) or base (cold).')
flags.DEFINE_integer('verify_k', 3,
                     'Re-check this many top candidates with the real arm.')
flags.DEFINE_integer('seed', 0, 'Random seed for CEM sampling.')

FLAGS = flags.FLAGS


def get_seeds():
  path = os.path.join(FLAGS.data_dir, f'{FLAGS.task}-{FLAGS.split}')
  if os.path.isdir(path):
    ds = Dataset(path)
    if ds.n_episodes > 0:
      n = min(FLAGS.n, ds.n_episodes)
      print(f'Using {n} seeds from {path}')
      return [ds.load(i, images=False)[1] for i in range(n)]
  # Same convention as demos.py: train seeds are even, test seeds are odd.
  first = 2 if FLAGS.split == 'train' else 1
  print(f'No {FLAGS.split} dataset found; using seeds {first}, {first + 2}, ...')
  return [first + 2 * i for i in range(FLAGS.n)]


def main(unused_argv):
  env = Environment(FLAGS.assets_root, disp=FLAGS.disp, hz=480)
  task = tasks.names[FLAGS.task]()
  task.mode = FLAGS.split

  init = 'base' if FLAGS.objective == 'free' else FLAGS.init
  name = (f'{FLAGS.task}-cem-{FLAGS.objective}-{FLAGS.split}-{FLAGS.rollout}-'
          f'{init}-v{FLAGS.verify_k}-{FLAGS.n_samples}x{FLAGS.n_iters}')
  results = []
  for k, seed in enumerate(get_seeds()):
    print(f'Test: {k + 1} (seed {seed})')
    np.random.seed(seed)
    env.seed(seed)
    env.set_task(task)
    obs = env.reset()

    kwargs = dict(n_samples=FLAGS.n_samples, n_elites=FLAGS.n_elites,
                  n_iters=FLAGS.n_iters, rollout=FLAGS.rollout,
                  verify_k=FLAGS.verify_k, seed=FLAGS.seed + seed)
    if FLAGS.objective == 'free':
      agent = FreeStackCEMAgent(env, task, **kwargs)
    else:
      agent = CEMAgent(env, task, init=FLAGS.init, **kwargs)

    info, total_reward, start = None, 0.0, time.time()
    for _ in range(task.max_steps):
      act = agent.act(obs, info)
      if act is None:
        break
      obs, reward, done, info = env.step(act)
      if FLAGS.objective == 'free':
        # Graded by outcome only: fraction of blocks stacked on the stand.
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
