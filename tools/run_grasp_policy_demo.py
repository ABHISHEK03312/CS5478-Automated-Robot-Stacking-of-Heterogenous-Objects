"""Run a matched heterogeneous-stacking demo for local grasp policies."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys


DEFAULT_QT_OPT = Path(
    'results/grasp-qt-opt-100/checkpoints/grasp-qt-opt-1500.h5')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data', type=Path,
                      default=Path('mixed-stacking-validated-test'))
  parser.add_argument('--episodes', type=int, default=1)
  parser.add_argument('--start-index', type=int, default=0)
  parser.add_argument('--policies', nargs='+',
                      default=['geometry', 'analytic', 'qt_opt'])
  parser.add_argument('--qt-opt-checkpoint', type=Path,
                      default=DEFAULT_QT_OPT)
  parser.add_argument('--output-dir', type=Path,
                      default=Path('results/grasp-policy-demo'))
  parser.add_argument('--gui', action='store_true')
  args = parser.parse_args()
  if args.episodes < 1 or args.start_index < 0:
    parser.error('episodes must be positive and start-index non-negative')

  args.output_dir.mkdir(parents=True, exist_ok=True)
  summaries = []
  for policy in args.policies:
    safe_name = policy.replace('_', '-')
    output = args.output_dir / f'{safe_name}.json'
    summary = args.output_dir / f'{safe_name}-summary.csv'
    command = [
        sys.executable, 'cem_hetero_towers.py',
        '--data', str(args.data),
        '--episodes', str(args.episodes),
        '--start-index', str(args.start_index),
        '--samples', '24', '--iters', '3', '--elites', '4',
        '--grasp-policy', policy,
        '--out', str(output), '--summary-out', str(summary),
    ]
    if policy == 'qt_opt':
      if not args.qt_opt_checkpoint.is_file():
        parser.error(f'QT-Opt checkpoint not found: {args.qt_opt_checkpoint}')
      command.extend(['--grasp-checkpoint', str(args.qt_opt_checkpoint)])
    elif policy == 'candidate_q':
      parser.error('candidate_q needs a checkpoint; run cem_hetero_towers.py '
                   'directly with --grasp-checkpoint')
    elif policy in ('gpd', 'contact_graspnet'):
      parser.error(f'{policy} needs an external backend command; run '
                   'cem_hetero_towers.py directly with repeated '
                   '--grasp-backend-arg')
    if args.gui:
      command.append('--gui')
    print(f'\nRunning {policy}:', ' '.join(command), flush=True)
    subprocess.run(command, check=True)
    with summary.open(newline='', encoding='utf-8') as stream:
      summaries.append(next(csv.DictReader(stream)))

  columns = ('Method', 'Episodes', 'Avg. Trials', 'Mean Reward', '95% CI',
             'Full Stacks', 'Partial Stacks', 'Zero Reward', 'Errors')
  widths = [max(len(column), *(len(row[column]) for row in summaries))
            for column in columns]
  print('\n' + ' | '.join(column.ljust(width) for column, width in
                          zip(columns, widths)))
  print('-+-'.join('-' * width for width in widths))
  for row in summaries:
    print(' | '.join(row[column].ljust(width) for column, width in
                     zip(columns, widths)))
  print(f'\nDetailed outputs: {args.output_dir}')


if __name__ == '__main__':
  main()
