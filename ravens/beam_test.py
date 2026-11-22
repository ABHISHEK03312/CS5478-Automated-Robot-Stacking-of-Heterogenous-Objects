"""Evaluate Ravens Beam Search on the original parallel-jaw pyramid task.

Print a CEM-compatible summary and save CSV results only (no PKL output).

Examples:
  python -m ravens.beam_test --objective=target --split=test --n=100
  python -m ravens.beam_test --objective=free --split=test --n=100

Uses existing Ravens dataset seeds when available; otherwise uses the standard
train-even / test-odd seeds. Requires ravens/agents/beam.py.
"""

import csv
import os
import time

from absl import app
from absl import flags
import numpy as np

from ravens import tasks
from ravens.agents.beam import BeamSearchAgent, stack_metrics
from ravens.dataset import Dataset
from ravens.environments.environment import Environment
from ravens.grasping.policies import create_grasp_planner
from ravens.grasping.policies import resolve_policy
from ravens.grasping.stack_action_planner import create_stack_action_scorer

flags.DEFINE_string('assets_root', './ravens/environments/assets/', '')
flags.DEFINE_string('data_dir', '.', '')
flags.DEFINE_string('root_dir', '.', 'Directory under which results/ is created.')
flags.DEFINE_bool('disp', False, '')
flags.DEFINE_string('task', 'stack-block-pyramid-parallel-jaw', '')
flags.DEFINE_enum('split', 'test', ['train', 'test'], 'Which scenes to run.')
flags.DEFINE_integer('n', 10, 'Number of episodes.')
flags.DEFINE_integer('start_index', 0,
                     'Zero-based held-out seed index for sharded evaluation.')
flags.DEFINE_enum('objective', 'target', ['target', 'free'],
                  'target: native Ravens reward; free: fraction on stand.')
flags.DEFINE_integer('beam_width', 3, 'Sequences kept per depth.')
flags.DEFINE_integer('horizon', 2, 'Moves to look ahead.')
flags.DEFINE_integer('n_blocks', 3, 'Candidate blocks per expansion.')
flags.DEFINE_integer('n_place', 6, 'Candidate placements per block.')
flags.DEFINE_integer('verify_k', 3, 'Top first moves verified by arm.')
flags.DEFINE_integer('seed', 0, 'Seed offset for placement sampling.')
flags.DEFINE_string('grasp_checkpoint', '',
                    'Optional candidate-Q checkpoint used to rank/refine picks.')
flags.DEFINE_enum('grasp_policy', 'geometry',
                  ['geometry', 'candidate_q', 'qt_opt', 'analytic', 'gpd',
                   'contact_graspnet'],
                  'Grasp candidate policy; checkpoint implies candidate_q.')
flags.DEFINE_multi_string(
    'grasp_backend_arg', [],
    'Repeated command arguments for an isolated point-cloud backend.')
flags.DEFINE_float('grasp_backend_timeout', 120, 'Backend timeout in seconds.')
flags.DEFINE_float('grasp_max_tilt_degrees', 20,
                   'Maximum external grasp approach tilt from top-down.')
flags.DEFINE_float('grasp_weight', 0.1,
                   'Small tie-break weight for conservative grasp Q.')
flags.DEFINE_integer('grasp_mc_samples', 8, '')
flags.DEFINE_bool('grasp_refine_pose', False,
                  'Experimental: replace planner center picks with Q proposals.')
flags.DEFINE_bool('grasp_gate', True,
                  'Only apply Q after probability/uncertainty/margin checks.')
flags.DEFINE_float('grasp_min_probability', 0.65, '')
flags.DEFINE_float('grasp_max_uncertainty', 0.08, '')
flags.DEFINE_float('grasp_min_margin', 0.15, '')
flags.DEFINE_string('stack_action_checkpoint', '',
                    'Optional placement-aware joint-action Q checkpoint.')
flags.DEFINE_float('stack_action_weight', 0.5,
                   'Weight for placement-aware Q during planning.')

FLAGS = flags.FLAGS


def get_seeds():
  if FLAGS.start_index < 0:
    raise ValueError('--start_index must be non-negative')
  path = os.path.join(FLAGS.data_dir, f'{FLAGS.task}-{FLAGS.split}')
  if os.path.isdir(path):
    ds = Dataset(path)
    if ds.n_episodes > 0:
      stop = min(FLAGS.start_index + FLAGS.n, ds.n_episodes)
      indices = range(FLAGS.start_index, stop)
      print(f'Using {len(indices)} seeds from {path}, starting at index '
            f'{FLAGS.start_index}')
      return [ds.load(i, images=False)[1] for i in indices]
  first = 2 if FLAGS.split == 'train' else 1
  first += 2 * FLAGS.start_index
  print(f'No {FLAGS.split} dataset found; using seeds {first}, {first + 2}, ...')
  return [first + 2 * i for i in range(FLAGS.n)]


def save_csv(path, fieldnames, rows):
  with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def print_and_save_summary(name, rows):
  valid = [r for r in rows if not r['error']]
  rewards = np.asarray([r['reward'] for r in valid], dtype=float)
  trials = [r['trials'] for r in valid]
  n = len(rows)
  full = int(np.sum(rewards > 0.99))
  partial = int(np.sum((rewards > 0) & (rewards <= 0.99)))
  zero = int(np.sum(rewards == 0))
  errors = n - len(valid)
  mean = float(np.mean(rewards)) if len(valid) else float('nan')
  ci = (1.96 * float(np.std(rewards, ddof=1)) / np.sqrt(len(rewards))
        if len(rewards) > 1 else float('nan'))
  avg_trials = float(np.mean(trials)) if trials else float('nan')
  method = f'Beam {FLAGS.objective} pyramid'
  summary = {
      'Method': method,
      'Episodes': n,
      'Avg. Trials': f'{avg_trials:.2f}' if trials else 'N/A',
      'Mean Reward': f'{mean:.3f}' if valid else 'N/A',
      '95% CI': f'±{ci:.3f}' if len(valid) > 1 else 'N/A',
      'Full': full,
      'Partial': partial,
      'Zero': zero,
      'Errors': errors,
  }
  outdir = os.path.join(FLAGS.root_dir, 'results')
  os.makedirs(outdir, exist_ok=True)
  summary_path = os.path.join(outdir, name + '_summary.csv')
  episodes_path = os.path.join(outdir, name + '_episodes.csv')
  save_csv(summary_path, list(summary), [summary])
  save_csv(episodes_path,
           ['episode', 'seed', 'reward', 'trials', 'rollouts', 'seconds',
            'height_m', 'grasp_gate_calls', 'grasp_gate_changes',
            'grasp_gate_abstentions', 'error'], rows)
  print('\n' + '=' * 110)
  print(f"{'Method':<24} {'Episodes':>8} {'Avg.Trials':>10} {'Mean Reward':>12} "
        f"{'95% CI':>12} {'Full':>6} {'Partial':>8} {'Zero':>6} {'Errors':>7}")
  print(f"{method:<24} {n:>8} {summary['Avg. Trials']:>10} "
        f"{summary['Mean Reward']:>12} {summary['95% CI']:>12} "
        f"{full:>6} {partial:>8} {zero:>6} {errors:>7}")
  print('=' * 110)
  print('Summary:', summary_path)
  print('Episodes:', episodes_path)
  print('Reward: native Ravens task reward (target), fraction stacked (free).')
  print('Trials: attempted UR5 pick/place actions; CI excludes errored episodes.')
  print('No PKL results are written.')


def main(unused_argv):
  grasp_policy = resolve_policy(FLAGS.grasp_policy, FLAGS.grasp_checkpoint)
  name = (f'{FLAGS.task}-beam-{FLAGS.objective}-{FLAGS.split}-'
          f'K{FLAGS.beam_width}-H{FLAGS.horizon}-v{FLAGS.verify_k}')
  if grasp_policy != 'geometry':
    name += f'-grasp-{grasp_policy}'
  if FLAGS.stack_action_checkpoint:
    name += '-stack-q'
  if FLAGS.start_index:
    name += f'-s{FLAGS.start_index}'
  env = Environment(FLAGS.assets_root, disp=FLAGS.disp, hz=480)
  rows = []
  grasp_scorer = create_grasp_planner(
      grasp_policy, checkpoint=FLAGS.grasp_checkpoint,
      mc_samples=FLAGS.grasp_mc_samples,
      refine_pose=FLAGS.grasp_refine_pose,
      min_probability=FLAGS.grasp_min_probability,
      max_uncertainty=FLAGS.grasp_max_uncertainty,
      min_margin=FLAGS.grasp_min_margin,
      backend_command=FLAGS.grasp_backend_arg,
      backend_timeout=FLAGS.grasp_backend_timeout,
      max_tilt_degrees=FLAGS.grasp_max_tilt_degrees)
  action_scorer = create_stack_action_scorer(FLAGS.stack_action_checkpoint)
  try:
    for k, seed in enumerate(get_seeds(), 1):
      print(f'Test: {k} (seed {seed})', flush=True)
      start = time.time()
      trials = 0
      total_reward = 0.0
      height = None
      agent = None
      error = ''
      gate_before = ((grasp_scorer.selection_calls,
                      grasp_scorer.selection_changes,
                      grasp_scorer.selection_abstentions)
                     if grasp_scorer is not None else (0, 0, 0))
      try:
        task = tasks.names[FLAGS.task]()
        task.mode = FLAGS.split
        np.random.seed(seed)
        env.seed(seed)
        env.set_task(task)
        obs = env.reset()
        agent = BeamSearchAgent(
            env, task, objective=FLAGS.objective, beam_width=FLAGS.beam_width,
            horizon=FLAGS.horizon, n_blocks=FLAGS.n_blocks,
            n_place=FLAGS.n_place, verify_k=FLAGS.verify_k,
            grasp_scorer=grasp_scorer, grasp_weight=FLAGS.grasp_weight,
            grasp_gate=FLAGS.grasp_gate,
            action_scorer=action_scorer,
            action_weight=FLAGS.stack_action_weight,
            seed=FLAGS.seed + seed)
        info = None
        for _ in range(task.max_steps):
          action = agent.act(obs, info)
          if action is None:
            break
          trials += 1
          obs, reward, done, info = env.step(action)
          if FLAGS.objective == 'free':
            n_stacked, n_total, height, _ = stack_metrics(env)
            total_reward = n_stacked / n_total if n_total else 0.0
            print(f'Stacked: {n_stacked}/{n_total}  height: {height * 100:.1f} cm')
            if n_stacked == n_total:
              break
          else:
            total_reward += float(reward)
            print(f'Total Reward: {total_reward:.3f} Done: {done}')
            if done:
              break
        if FLAGS.objective == 'free':
          _, _, height, _ = stack_metrics(env)
      except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(f'ERROR episode {k}: {error}', flush=True)
      elapsed = time.time() - start
      rollouts = agent.n_rollouts if agent is not None else 0
      gate_after = ((grasp_scorer.selection_calls,
                     grasp_scorer.selection_changes,
                     grasp_scorer.selection_abstentions)
                    if grasp_scorer is not None else gate_before)
      print(f'  episode time {elapsed:.1f}s, rollouts {rollouts}, trials {trials}')
      rows.append({
          'episode': k, 'seed': seed,
          'reward': round(total_reward, 8) if not error else '',
          'trials': trials, 'rollouts': rollouts,
          'seconds': round(elapsed, 3),
          'height_m': round(float(height), 6) if height is not None else '',
          'grasp_gate_calls': gate_after[0] - gate_before[0],
          'grasp_gate_changes': gate_after[1] - gate_before[1],
          'grasp_gate_abstentions': gate_after[2] - gate_before[2],
          'error': error,
      })
  finally:
    env.close()
    print_and_save_summary(name, rows)


if __name__ == '__main__':
  app.run(main)
