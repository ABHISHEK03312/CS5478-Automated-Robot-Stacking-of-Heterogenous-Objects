"""Compare Low/Medium/High CEM and Beam Search on the same validation episodes.

Run from your Ravens project root:
    python tune_cem_beam_hetero.py --data mixed-stacking-validated --episodes 10

Required existing scripts in the project root:
    cem_hetero_towers.py, beam_hetero.py, cem_grasp_beam.py,
    generate_stacking_ravens.py

Each setting runs on the same first N episode JSON files. Reports executed
stacking completion, full-stack success, errors, and wall-clock time.
"""
import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

CEM_CONFIGS = {
    'low': {'samples': 24, 'iters': 3, 'elites': 4},
    'medium': {'samples': 48, 'iters': 4, 'elites': 8},
    #'high': {'samples': 96, 'iters': 6, 'elites': 12},
}
BEAM_CONFIGS = {
    'low': {'beam_width': 4},
    'medium': {'beam_width': 8},
    #'high': {'beam_width': 16},
}


def aggregate(records, elapsed, method, level, config, returncode):
    """Report every requested episode; errors count as zero reward.

    CI is the normal-approximation 95% interval across requested episodes,
    clipped to [0, 1]. Trials count attempted UR5 pick-and-place actions.
    """
    n = len(records)
    rewards = []
    trials = []
    full = partial = zero = errors = 0
    for record in records:
        total = record.get('total')
        valid = (not record.get('error') and isinstance(total, int)
                 and total > 0)
        if not valid:
            errors += 1
            reward = 0.0
        else:
            reward = max(0.0, min(1.0, record.get('stacked', 0) / total))
        rewards.append(reward)
        trials.append(len(record.get('actions') or []))
        if valid and record.get('success'):
            full += 1
        elif valid and reward > 0:
            partial += 1
        elif valid:
            zero += 1

    mean = statistics.mean(rewards) if n else 0.0
    half_width = (1.96 * statistics.stdev(rewards) / math.sqrt(n)) if n > 1 else 0.0
    lo, hi = max(0.0, mean - half_width), min(1.0, mean + half_width)
    return {
        'Method': f'{method} {level.title()}',
        'Episodes': n,
        'Avg. Trials': round(statistics.mean(trials), 2) if n else 0.0,
        'Mean Reward': round(mean, 3),
        '95% CI': f'[{lo:.3f}, {hi:.3f}]' if n > 1 else 'N/A',
        'Full Stacks': full,
        'Partial Stacks': partial,
        'Zero Reward': zero,
        'Errors': errors,
        'Avg. Time (s)': round(elapsed / n, 2) if n else 0.0,
        'level': level,
        'method': method,
        'samples': config.get('samples', ''),
        'iters': config.get('iters', ''),
        'elites': config.get('elites', ''),
        'beam_width': config.get('beam_width', ''),
        'valid_episodes': n - errors,
        'full_stack_rate': round(full / n, 4) if n else 0.0,
        'total_seconds': round(elapsed, 2),
        'returncode': returncode,
    }


def print_comparison(results):
    columns = ['Method', 'Episodes', 'Avg. Trials', 'Mean Reward', '95% CI',
               'Full Stacks', 'Partial Stacks', 'Zero Reward', 'Errors', 'Avg. Time (s)']
    widths = {col: max(len(col), *(len(str(row[col])) for row in results))
              for col in columns}
    print('\n' + '  '.join(col.ljust(widths[col]) for col in columns))
    print('  '.join('-' * widths[col] for col in columns))
    for row in results:
        print('  '.join(str(row[col]).ljust(widths[col]) for col in columns))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('mixed-stacking-validated'))
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--assets', type=Path, default=Path('ravens/environments/assets'))
    parser.add_argument('--out', type=Path, default=Path('results/hyperparameter_tuning'))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--settle-steps', type=int, default=90)
    parser.add_argument('--gui', action='store_true', help='Show PyBullet GUI (slows comparisons)')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    episodes_found = sorted(args.data.glob('episode_*/episode.json'))
    if len(episodes_found) < args.episodes:
        parser.error(f'Only {len(episodes_found)} episodes found in {args.data}; requested {args.episodes}')
    args.out.mkdir(parents=True, exist_ok=True)

    runs = [('CEM', k, v) for k, v in CEM_CONFIGS.items()] + [('Beam', k, v) for k, v in BEAM_CONFIGS.items()]
    results = []
    for method, level, config in runs:
        stem = f'{method.lower()}_{level}'
        destination = args.out / f'{stem}.json'
        script = 'cem_hetero_towers.py' if method == 'CEM' else 'beam_hetero.py'
        command = [sys.executable, script,
                   '--data', str(args.data), '--assets', str(args.assets),
                   '--episodes', str(args.episodes), '--seed', str(args.seed),
                   '--settle-steps', str(args.settle_steps), '--out', str(destination)]
        if method == 'CEM':
            command += ['--samples', str(config['samples']), '--iters', str(config['iters']),
                        '--elites', str(config['elites']), '--summary-out', str(args.out / f'{stem}_native_summary.csv')]
        else:
            command += ['--beam-width', str(config['beam_width'])]
        if args.gui:
            command.append('--gui')
        print(f'\n=== {method} {level.upper()} {config} ===', flush=True)
        if destination.exists():
            destination.unlink()  # Never accidentally reuse prior run results.
        start = time.perf_counter()
        with (args.out / f'{stem}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed = time.perf_counter() - start
        try:
            records = json.loads(destination.read_text(encoding='utf-8'))
            if not isinstance(records, list):
                raise ValueError('Expected a JSON array')
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f'  Could not read episode results: {exc}. Check {stem}.log')
            records = []
        if records:
            row = aggregate(records, elapsed, method, level, config, result.returncode)
        else:
            row = aggregate([{'error': 'Missing results'}] * args.episodes,
                            elapsed, method, level, config, result.returncode)
        results.append(row)
        print(f'  full-stack rate={row["full_stack_rate"]:.1%}, mean reward={row["Mean Reward"]:.3f}, '
              f'errors={row["Errors"]}, time={elapsed:.1f}s (exit={result.returncode})', flush=True)

        # Save incrementally so earlier runs survive interruption.
        with (args.out / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        (args.out / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    print_comparison(results)

    # Rank by full stack completion, then mean reward, then lower wall time.
    best = {}
    for method in ('CEM', 'Beam'):
        candidates = [r for r in results if r['method'] == method and r['valid_episodes'] == args.episodes and r['returncode'] == 0]
        if candidates:
            best[method] = max(candidates, key=lambda r: (r['full_stack_rate'], r['Mean Reward'], -r['Avg. Time (s)']))
        else:
            best[method] = None
    (args.out / 'best_configs.json').write_text(json.dumps(best, indent=2), encoding='utf-8')
    print(f'\nSummary: {args.out / "summary.csv"}')
    print(f'Best configurations: {args.out / "best_configs.json"}')
    for method, chosen in best.items():
        print(f'  {method}: {chosen["level"] if chosen else "No complete valid configuration"}')


if __name__ == '__main__':
    main()
