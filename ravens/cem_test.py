"""Evaluate Ravens CEM and save CSV summaries only (no PKL result files).

Save as ravens/cem_test.py alongside ravens/agents/cem.py.
"""
import csv
import math
import os
import time
import traceback

from absl import app, flags
import numpy as np

from ravens import tasks
from ravens.agents.cem import CEMAgent, FreeStackCEMAgent, stack_metrics
from ravens.dataset import Dataset
from ravens.environments.environment import Environment

flags.DEFINE_string('assets_root', './ravens/environments/assets/', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_string('root_dir', 'results', 'Directory for CSV evaluation outputs.')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_string('task', 'stack-block-pyramid-parallel-jaw', '')
flags.DEFINE_enum('split', 'test', ['train', 'test'], '')
flags.DEFINE_integer('n', 10, 'Number of episodes.')
flags.DEFINE_integer('n_samples', 24, 'CEM candidates per iteration.')
flags.DEFINE_integer('n_elites', 5, 'CEM elites per iteration.')
flags.DEFINE_integer('n_iters', 3, 'CEM iterations per action.')
flags.DEFINE_enum('rollout', 'teleport', ['teleport', 'full'], '')
flags.DEFINE_enum('objective', 'target', ['target', 'free'], '')
flags.DEFINE_enum('init', 'target', ['target', 'base'], '')
flags.DEFINE_integer('verify_k', 3, 'Top candidates rechecked using arm motion.')
flags.DEFINE_integer('seed', 0, 'Sampling seed offset.')
FLAGS = flags.FLAGS


def get_seeds():
  path = os.path.join(FLAGS.data_dir, f'{FLAGS.task}-{FLAGS.split}')
  if os.path.isdir(path):
    ds = Dataset(path)
    if ds.n_episodes > 0:
      count = min(FLAGS.n, ds.n_episodes)
      print(f'Using {count} seeds from {path}')
      return [ds.load(i, images=False)[1] for i in range(count)]
  first = 2 if FLAGS.split == 'train' else 1
  print(f'No {FLAGS.split} dataset found; using seeds {first}, {first + 2}, ...')
  return [first + 2 * i for i in range(FLAGS.n)]


def evaluate_summary(rows, name):
  count = len(rows)
  valid = [r for r in rows if not r['error']]
  rewards = np.asarray([r['reward'] for r in valid], dtype=float)
  mean = float(rewards.mean()) if len(rewards) else float('nan')
  if len(rewards) > 1:
    sem = float(rewards.std(ddof=1) / math.sqrt(len(rewards)))
    ci_lo, ci_hi = max(0., mean - 1.96 * sem), min(1., mean + 1.96 * sem)
  elif len(rewards) == 1:
    ci_lo = ci_hi = float('nan')  # CI cannot be estimated with one sample
  else:
    ci_lo = ci_hi = float('nan')
  full = sum(r['reward'] > 0.99 for r in valid)
  partial = sum(0 < r['reward'] <= 0.99 for r in valid)
  zero = sum(r['reward'] <= 0 for r in valid)
  errors = count - len(valid)
  avg_trials = sum(r['trials'] for r in rows) / count if count else 0
  avg_rollouts = sum(r['rollouts'] for r in rows) / count if count else 0
  summary = dict(method=f'CEM {FLAGS.objective}', task=FLAGS.task,
                 split=FLAGS.split, episodes=count, valid=len(valid),
                 avg_trials=round(avg_trials, 2), avg_rollouts=round(avg_rollouts, 2),
                 mean_reward=round(mean, 4), ci95_lower=round(ci_lo, 4),
                 ci95_upper=round(ci_hi, 4), full_stacks=full,
                 partial_stacks=partial, zero_reward=zero, errors=errors)
  ci_display = (f'[{ci_lo:.3f}, {ci_hi:.3f}]' if len(valid) >= 2 else 'N/A')
  print('\n' + '=' * 108)
  print(f'{"Method":<15} {"Episodes":>8} {"Avg.Trials":>10} {"Mean Reward":>12} '
        f'{"95% CI":>16} {"Full":>6} {"Partial":>8} {"Zero":>6} {"Errors":>7}')
  print(f'{summary["method"]:<15} {count:>8} {avg_trials:>10.2f} {mean:>12.3f} '
        f'{ci_display:>16} {full:>6} {partial:>8} {zero:>6} {errors:>7}')
  print('=' * 108)
  os.makedirs(FLAGS.root_dir, exist_ok=True)
  summary_path = os.path.join(FLAGS.root_dir, name + '_summary.csv')
  details_path = os.path.join(FLAGS.root_dir, name + '_episodes.csv')
  with open(summary_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary))
    writer.writeheader()
    writer.writerow(summary)
  with open(details_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else
                            ['episode', 'seed', 'reward', 'trials', 'rollouts', 'seconds', 'error'])
    writer.writeheader()
    writer.writerows(rows)
  print(f'Saved summary: {summary_path}\nSaved per-episode details: {details_path}')
  print('Trials = executed pick/place attempts; rollouts = internal CEM candidate evaluations.')
  print('Target reward = original Ravens task reward; free reward = fraction on stand.')
  print('95% CI uses normal approximation over valid episode rewards; errors are excluded from the mean.')


def main(_):
  if FLAGS.n < 1:
    raise ValueError('--n must be at least 1')
  if not (1 <= FLAGS.n_elites <= FLAGS.n_samples):
    raise ValueError('Require 1 <= n_elites <= n_samples')
  os.makedirs(FLAGS.root_dir, exist_ok=True)
  env = Environment(FLAGS.assets_root, disp=FLAGS.disp, hz=480)
  try:
    task = tasks.names[FLAGS.task]()
    task.mode = FLAGS.split
    init = 'base' if FLAGS.objective == 'free' else FLAGS.init
    name = (f'{FLAGS.task}-cem-{FLAGS.objective}-{FLAGS.split}-{FLAGS.rollout}-'
            f'{init}-v{FLAGS.verify_k}-{FLAGS.n_samples}x{FLAGS.n_iters}')
    rows = []
    for k, seed in enumerate(get_seeds(), start=1):
      print(f'\nTest {k} (seed {seed})')
      start = time.time()
      trials = rollouts = 0
      total_reward = 0.0
      info = None
      error = ''
      agent = None
      try:
        np.random.seed(seed)
        env.seed(seed)
        env.set_task(task)
        obs = env.reset()
        kwargs = dict(n_samples=FLAGS.n_samples, n_elites=FLAGS.n_elites,
                      n_iters=FLAGS.n_iters, rollout=FLAGS.rollout,
                      verify_k=FLAGS.verify_k, seed=FLAGS.seed + seed)
        agent = (FreeStackCEMAgent(env, task, **kwargs) if FLAGS.objective == 'free'
                 else CEMAgent(env, task, init=FLAGS.init, **kwargs))
        for _ in range(task.max_steps):
          act = agent.act(obs, info)
          if act is None:
            break
          trials += 1
          obs, reward, done, info = env.step(act)
          if FLAGS.objective == 'free':
            n_stacked, n_total, height, _ = stack_metrics(env)
            total_reward = float(n_stacked / n_total) if n_total else 0.0
            print(f'Stacked: {n_stacked}/{n_total}, height: {height * 100:.1f} cm')
            if n_total and n_stacked == n_total:
              break
          else:
            total_reward += float(reward)
            print(f'Total Reward: {total_reward:.3f} Done: {done}')
            if done:
              break
        if FLAGS.objective == 'free':
          info = {'height': float(stack_metrics(env)[2])}
      except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(f'ERROR in episode {k}: {error}')
        traceback.print_exc()
      finally:
        rollouts = int(agent.n_rollouts) if agent is not None else 0
      secs = time.time() - start
      rows.append(dict(episode=k, seed=seed, reward=round(total_reward, 6),
                       trials=trials, rollouts=rollouts, seconds=round(secs, 2),
                       error=error))
      print(f'Episode: reward={total_reward:.3f}, trials={trials}, rollouts={rollouts}, '
            f'time={secs:.1f}s' + (f', error={error}' if error else ''))
    evaluate_summary(rows, name)
  finally:
    env.close()


if __name__ == '__main__':
  app.run(main)
