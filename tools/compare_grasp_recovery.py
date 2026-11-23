"""Merge robust-grasp retry shards and compare them to no-retry results."""

import argparse
import json
from pathlib import Path

import numpy as np

from merge_grasp_policy_benchmark import load_shards
from merge_grasp_policy_benchmark import paired_stats
from merge_grasp_policy_benchmark import reward
from merge_grasp_policy_benchmark import summarize
from merge_grasp_policy_benchmark import write_csv


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--baseline', type=Path, nargs='+', required=True)
  parser.add_argument('--analytic-retry', type=Path, nargs='+', required=True)
  parser.add_argument('--qt-opt-retry', type=Path, nargs='+', required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--bootstrap-samples', type=int, default=20000)
  parser.add_argument('--seed', type=int, default=5478)
  args = parser.parse_args()

  policies = {
      'analytic_no_retry': load_shards(args.baseline),
      'analytic_retry': load_shards(args.analytic_retry),
      'qt_opt_retry': load_shards(args.qt_opt_retry),
  }
  episode_sets = {
      name: {row['episode'] for row in rows}
      for name, rows in policies.items()
  }
  if len({frozenset(value) for value in episode_sets.values()}) != 1:
    raise ValueError('policy episode IDs do not match')

  episodes = sorted(episode_sets['analytic_no_retry'])
  arrays = {}
  ordered = {}
  for name, rows in policies.items():
    by_episode = {row['episode']: row for row in rows}
    ordered[name] = [by_episode[episode] for episode in episodes]
    arrays[name] = np.asarray(
        [reward(row) for row in ordered[name]], dtype=float)

  summaries = []
  baseline = arrays['analytic_no_retry']
  for name in policies:
    summary = summarize(name, ordered[name])
    if name == 'analytic_no_retry':
      summary.update({
          'wins_vs_no_retry': 0,
          'ties_vs_no_retry': len(episodes),
          'losses_vs_no_retry': 0,
          'mean_delta': 0.0,
          'paired_ci_lower': 0.0,
          'paired_ci_upper': 0.0,
      })
    else:
      paired = paired_stats(
          baseline, arrays[name], args.bootstrap_samples, args.seed)
      summary.update({
          'wins_vs_no_retry': paired['wins_vs_geometry'],
          'ties_vs_no_retry': paired['ties_vs_geometry'],
          'losses_vs_no_retry': paired['losses_vs_geometry'],
          'mean_delta': paired['mean_delta'],
          'paired_ci_lower': paired['paired_ci_lower'],
          'paired_ci_upper': paired['paired_ci_upper'],
      })
    summaries.append(summary)

  episode_rows = []
  for index, episode in enumerate(episodes):
    episode_rows.append({
        'episode': episode,
        **{f'{name}_reward': round(float(values[index]), 6)
           for name, values in arrays.items()},
    })

  args.output_dir.mkdir(parents=True, exist_ok=True)
  for name in ('analytic_retry', 'qt_opt_retry'):
    (args.output_dir / f'{name}.json').write_text(
        json.dumps(ordered[name], indent=2), encoding='utf-8')
  write_csv(args.output_dir / 'comparison.csv', summaries)
  write_csv(args.output_dir / 'episodes.csv', episode_rows)
  for summary in summaries:
    print(summary)


if __name__ == '__main__':
  main()
