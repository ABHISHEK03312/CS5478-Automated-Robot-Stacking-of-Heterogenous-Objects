"""Merge sharded heterogeneous CEM results and compare matched episodes."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def reward(row):
  return float(row['stacked']) / float(row['total'])


def summary(rows, method):
  rewards = np.asarray([reward(row) for row in rows], dtype=float)
  half_width = (1.96 * rewards.std(ddof=1) / math.sqrt(len(rewards))
                if len(rewards) > 1 else 0.0)
  mean = float(rewards.mean())
  return {
      'Method': method,
      'Episodes': len(rows),
      'Avg. Trials': round(float(np.mean([
          len(row.get('actions') or []) for row in rows])), 2),
      'Mean Reward': round(mean, 3),
      '95% CI': f'[{max(0., mean-half_width):.3f}, '
                f'{min(1., mean+half_width):.3f}]',
      'Full Stacks': sum(row.get('success', False) for row in rows),
      'Partial Stacks': sum(0 < reward(row) < 1 for row in rows),
      'Zero Reward': sum(reward(row) == 0 for row in rows),
      'Errors': sum(bool(row.get('error')) for row in rows),
  }


def write_csv(path, rows):
  with path.open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--baseline', type=Path, required=True)
  parser.add_argument('--analytic', type=Path, nargs='+', required=True)
  parser.add_argument('--output-prefix', type=Path, required=True)
  parser.add_argument('--bootstrap-samples', type=int, default=20000)
  args = parser.parse_args()

  baseline_rows = json.loads(args.baseline.read_text(encoding='utf-8'))
  analytic_rows = []
  for path in args.analytic:
    analytic_rows.extend(json.loads(path.read_text(encoding='utf-8')))
  analytic_rows.sort(key=lambda row: row['episode'])
  baseline = {row['episode']: row for row in baseline_rows}
  analytic = {row['episode']: row for row in analytic_rows}
  if len(baseline) != len(baseline_rows) or len(analytic) != len(analytic_rows):
    raise ValueError('duplicate episode IDs')
  if set(baseline) != set(analytic):
    raise ValueError('baseline and analytic episode IDs do not match')

  episodes = sorted(baseline)
  baseline_rewards = np.asarray([reward(baseline[key]) for key in episodes])
  analytic_rewards = np.asarray([reward(analytic[key]) for key in episodes])
  differences = analytic_rewards - baseline_rewards
  rng = np.random.default_rng(0)
  samples = differences[rng.integers(
      0, len(differences), (args.bootstrap_samples, len(differences)))].mean(1)
  lower, upper = np.quantile(samples, (0.025, 0.975))

  episode_rows = []
  for key, base_reward, analytic_reward in zip(
      episodes, baseline_rewards, analytic_rewards):
    row = analytic[key]
    episode_rows.append({
        'episode': key,
        'cem_reward': round(float(base_reward), 6),
        'cem_analytic_reward': round(float(analytic_reward), 6),
        'difference': round(float(analytic_reward - base_reward), 6),
        'grasp_gate_calls': row.get('grasp_gate_calls', 0),
        'grasp_gate_changes': row.get('grasp_gate_changes', 0),
        'grasp_gate_abstentions': row.get('grasp_gate_abstentions', 0),
    })

  comparison = [{
      'episodes': len(episodes),
      'cem_mean_reward': round(float(baseline_rewards.mean()), 6),
      'cem_analytic_mean_reward': round(float(analytic_rewards.mean()), 6),
      'paired_mean_difference': round(float(differences.mean()), 6),
      'paired_bootstrap_ci_lower': round(float(lower), 6),
      'paired_bootstrap_ci_upper': round(float(upper), 6),
      'wins': int(np.sum(differences > 0)),
      'ties': int(np.sum(differences == 0)),
      'losses': int(np.sum(differences < 0)),
      'grasp_gate_calls': sum(row.get('grasp_gate_calls', 0)
                              for row in analytic_rows),
      'grasp_gate_changes': sum(row.get('grasp_gate_changes', 0)
                                for row in analytic_rows),
      'grasp_gate_abstentions': sum(row.get('grasp_gate_abstentions', 0)
                                    for row in analytic_rows),
  }]

  args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
  args.output_prefix.with_suffix('.json').write_text(
      json.dumps(analytic_rows, indent=2), encoding='utf-8')
  write_csv(Path(str(args.output_prefix) + '_summary.csv'),
            [summary(analytic_rows, 'CEM + analytic - Tower')])
  write_csv(Path(str(args.output_prefix) + '_episodes.csv'), episode_rows)
  write_csv(Path(str(args.output_prefix) + '_comparison.csv'), comparison)
  print(summary(analytic_rows, 'CEM + analytic - Tower'))
  print(comparison[0])


if __name__ == '__main__':
  main()
