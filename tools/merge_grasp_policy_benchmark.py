"""Merge matched grasp-policy shards and report paired performance."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def reward(row):
  return float(row['stacked']) / float(row['total'])


def load_shards(paths):
  rows = []
  for path in paths:
    rows.extend(json.loads(path.read_text(encoding='utf-8')))
  rows.sort(key=lambda row: row['episode'])
  episodes = [row['episode'] for row in rows]
  if len(episodes) != len(set(episodes)):
    raise ValueError('duplicate episode IDs in input shards')
  return rows


def interval(values):
  half_width = (1.96 * values.std(ddof=1) / math.sqrt(len(values))
                if len(values) > 1 else 0.0)
  mean = float(values.mean())
  return max(0.0, mean - half_width), min(1.0, mean + half_width)


def summarize(name, rows):
  rewards = np.asarray([reward(row) for row in rows], dtype=float)
  lower, upper = interval(rewards)
  return {
      'policy': name,
      'episodes': len(rows),
      'avg_trials': round(float(np.mean([
          len(row.get('actions') or []) for row in rows])), 2),
      'mean_reward': round(float(rewards.mean()), 3),
      'ci95': f'[{lower:.3f}, {upper:.3f}]',
      'full_stacks': sum(reward(row) == 1 for row in rows),
      'partial_stacks': sum(0 < reward(row) < 1 for row in rows),
      'zero_reward': sum(reward(row) == 0 for row in rows),
      'errors': sum(bool(row.get('error')) for row in rows),
  }


def paired_stats(baseline, candidate, samples, seed):
  differences = candidate - baseline
  rng = np.random.default_rng(seed)
  bootstrapped = differences[rng.integers(
      0, len(differences), (samples, len(differences)))].mean(axis=1)
  lower, upper = np.quantile(bootstrapped, (0.025, 0.975))
  return {
      'wins_vs_geometry': int(np.sum(differences > 0)),
      'ties_vs_geometry': int(np.sum(differences == 0)),
      'losses_vs_geometry': int(np.sum(differences < 0)),
      'mean_delta': round(float(differences.mean()), 6),
      'paired_ci_lower': round(float(lower), 6),
      'paired_ci_upper': round(float(upper), 6),
  }


def write_csv(path, rows):
  with path.open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--geometry', type=Path, nargs='+', required=True)
  parser.add_argument('--analytic-refine', type=Path, nargs='+', required=True)
  parser.add_argument('--qt-opt-refine', type=Path, nargs='+', required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--bootstrap-samples', type=int, default=20000)
  parser.add_argument('--seed', type=int, default=5478)
  args = parser.parse_args()

  policies = {
      'geometry': load_shards(args.geometry),
      'analytic_refine': load_shards(args.analytic_refine),
      'qt_opt_refine': load_shards(args.qt_opt_refine),
  }
  episode_sets = {
      name: {row['episode'] for row in rows}
      for name, rows in policies.items()
  }
  if len({frozenset(value) for value in episode_sets.values()}) != 1:
    raise ValueError('policy episode IDs do not match')

  args.output_dir.mkdir(parents=True, exist_ok=True)
  summaries = []
  episode_rows = []
  episodes = sorted(episode_sets['geometry'])
  reward_arrays = {}
  for name, rows in policies.items():
    by_episode = {row['episode']: row for row in rows}
    ordered_rows = [by_episode[episode] for episode in episodes]
    reward_arrays[name] = np.asarray(
        [reward(row) for row in ordered_rows], dtype=float)
    (args.output_dir / f'{name}.json').write_text(
        json.dumps(ordered_rows, indent=2), encoding='utf-8')
    summary = summarize(name, ordered_rows)
    if name == 'geometry':
      summary.update({
          'wins_vs_geometry': 0,
          'ties_vs_geometry': len(episodes),
          'losses_vs_geometry': 0,
          'mean_delta': 0.0,
          'paired_ci_lower': 0.0,
          'paired_ci_upper': 0.0,
      })
    else:
      summary.update(paired_stats(
          reward_arrays['geometry'], reward_arrays[name],
          args.bootstrap_samples, args.seed))
    summaries.append(summary)

  for index, episode in enumerate(episodes):
    episode_rows.append({
        'episode': episode,
        'geometry_reward': round(float(reward_arrays['geometry'][index]), 6),
        'analytic_refine_reward': round(
            float(reward_arrays['analytic_refine'][index]), 6),
        'qt_opt_refine_reward': round(
            float(reward_arrays['qt_opt_refine'][index]), 6),
    })

  write_csv(args.output_dir / 'comparison.csv', summaries)
  write_csv(args.output_dir / 'episodes.csv', episode_rows)
  for row in summaries:
    print(row)


if __name__ == '__main__':
  main()
