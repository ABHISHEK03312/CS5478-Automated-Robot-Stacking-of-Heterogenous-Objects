"""Beam Search target-free heterogeneous pyramid planner (3/6/10 objects).

Uses original episode_*/episode.json scenes; no reference goal poses.
Infers 2/3/4 layers by object count; --layers optionally filters episodes.
Low beam default: width 4.
Requires hetero_pyramid_support.py next to this script.
"""
import argparse
import csv
import json
import math
import traceback
import time
from pathlib import Path

import numpy as np
import pybullet as p

from cem_grasp_beam import DatasetSceneTask, settle
from generate_stacking_ravens import fits, support_halfwidths
from ravens.environments.environment import Environment
from hetero_pyramid_support_height import resolve_layers, placement_target, pyramid_stable, row_and_col, height_completion_possible

CENTER = np.array([0.5, 0.18])


def yaw_delta(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def complete_order_possible(last, remaining, geo):
    if not remaining:
        return True
    if last is not None and geo[last]['shape'] == 'triangle':
        return False
    return any((last is None or fits(geo[i], geo[last], margin=0.001))
               and complete_order_possible(i, tuple(j for j in remaining if j != i), geo)
               for i in remaining)


def grasp_candidates(oid, geo, bodies, jaw_opening):
    obj = geo[oid]
    body = bodies[oid]
    pos, quat = p.getBasePositionAndOrientation(body)
    length, width, _ = obj['dims']
    object_yaw = p.getEulerFromQuaternion(quat)[2]
    top_z = p.getAABB(body)[1][2]
    # Both principal jaw orientations, plus small grasp offsets.
    for relative_yaw in (0.0, math.pi / 2):
        span = (length if obj['shape'] == 'cylinder' else
                abs(length * math.cos(relative_yaw)) + abs(width * math.sin(relative_yaw)))
        if span + 0.002 > jaw_opening:
            continue
        yaw = object_yaw + relative_yaw
        for dx, dy in ((0., 0.), (0.006, 0.), (-0.006, 0.), (0., 0.006), (0., -0.006)):
            yield ((float(pos[0] + dx), float(pos[1] + dy), float(top_z)),
                   p.getQuaternionFromEuler((0., 0., yaw))), span, math.hypot(dx, dy)


def placement_candidates():
    # Discrete grid; no sampling distribution, elites, or CEM updates.
    for dx, dy in ((0., 0.), (0.006, 0.), (-0.006, 0.), (0., 0.006), (0., -0.006)):
        for yaw in (0., math.pi / 2):
            yield dx, dy, yaw


def score_placement(oid, pick, place, placed, geo, bodies, settle_steps, layers):
    """Simulate a proposed placement from current world state; leave world unchanged."""
    snapshot = p.saveState()
    try:
        body = bodies[oid]
        obj = geo[oid]
        old_yaw = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(body)[1])[2]
        pick_yaw = p.getEulerFromQuaternion(pick[1])[2]
        place_yaw = p.getEulerFromQuaternion(place[1])[2]
        object_final_yaw = old_yaw + yaw_delta(place_yaw, pick_yaw)
        layout = placement_target(placed, oid, layers, geo, bodies)
        if layout is None:
            return None
        _, z = layout
        before = {j: np.array(p.getBasePositionAndOrientation(bodies[j])[0]) for j in placed}
        p.resetBasePositionAndOrientation(body, (place[0][0], place[0][1], z),
                                          p.getQuaternionFromEuler((0., 0., object_final_yaw)))
        p.resetBaseVelocity(body, (0, 0, 0), (0, 0, 0))
        settle(settle_steps)
        if not pyramid_stable(placed + [oid], layers, geo, bodies):
            return None
        disturbance = sum(np.linalg.norm(np.array(p.getBasePositionAndOrientation(bodies[j])[0]) - before[j])
                          for j in placed)
        layout_xy, _ = layout
        base_bonus = (0.8 * obj['dims'][0] / max(g['dims'][0] for g in geo.values())
                      if row_and_col(len(placed), layers)[0] == 0 else 0.)
        return (10. + base_bonus - 65. * np.linalg.norm(np.array(place[0][:2]) - layout_xy)
                - 35. * disturbance - 0.2 * np.linalg.norm(np.array(pick[0][:2]) - layout_xy))
    finally:
        p.restoreState(snapshot)
        p.removeState(snapshot)


def beam_plan(task, env, placed, beam_width=4, settle_steps=90, layers=None):
    """Beam search over candidate (object, grasp, placement) actions.

    Each depth extends a sequence of planned actions. Candidate physics is
    replayed from the original state using teleport rollouts, not UR5 motions.
    """
    geo = {int(o['id']): o for o in task.meta['objects']}
    layers = resolve_layers(len(geo), layers)
    ids = tuple(i for i in geo if i not in placed)
    height_cache = {}
    root = p.saveState()
    try:
        # Each node stores (score, actions). Replay is deterministic.
        beam = [(0., [])]
        for depth in range(len(ids)):
            next_beam = []
            for cumulative, actions in beam:
                p.restoreState(root)
                # Replay planned placements as a cheap simulation surrogate.
                current = list(placed)
                for action in actions:
                    oid = action['object_id']
                    obj = geo[oid]
                    old_yaw = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(task.bodies[oid])[1])[2]
                    pyaw = p.getEulerFromQuaternion(action['pick'][1])[2]
                    qyaw = p.getEulerFromQuaternion(action['place'][1])[2]
                    layout = placement_target(current, oid, layers, geo, task.bodies)
                    if layout is None:
                        continue
                    xy, z = layout
                    p.resetBasePositionAndOrientation(task.bodies[oid],
                        (action['place'][0][0], action['place'][0][1], z),
                        p.getQuaternionFromEuler((0, 0, old_yaw + yaw_delta(qyaw, pyaw))))
                    p.resetBaseVelocity(task.bodies[oid], (0, 0, 0), (0, 0, 0))
                    settle(settle_steps)
                    current.append(oid)
                remaining = tuple(i for i in ids if i not in current)
                for oid in remaining:
                    key = tuple(current + [oid])
                    if key not in height_cache:
                        height_cache[key] = height_completion_possible(
                            key, tuple(i for i in remaining if i != oid), layers, geo)
                    if not height_cache[key]:
                        continue
                    picks = list(grasp_candidates(oid, geo, task.bodies, env.ee.OPEN_INNER_WIDTH))
                    if not picks:
                        continue
                    layout = placement_target(current, oid, layers, geo, task.bodies)
                    if layout is None:
                        continue
                    xy, center_z = layout
                    top_z = center_z + geo[oid]['dims'][2] / 2
                    for pick, span, grasp_offset in picks:
                        for dx, dy, yaw in placement_candidates():
                            place = ((float(xy[0] + dx), float(xy[1] + dy), float(top_z)),
                                     p.getQuaternionFromEuler((0, 0, yaw)))
                            score = score_placement(oid, pick, place, current, geo,
                                                    task.bodies, settle_steps, layers)
                            if score is None:
                                continue
                            action = {'object_id': oid, 'pick': pick, 'place': place,
                                      'jaw_width': span, 'score': score - 3 * grasp_offset}
                            next_beam.append((cumulative + action['score'], actions + [action]))
            if not next_beam:
                return None
            next_beam.sort(key=lambda node: node[0], reverse=True)
            beam = next_beam[:beam_width]
        return beam[0][1][0] if beam else None
    finally:
        p.restoreState(root)
        p.removeState(root)


def run_episode(path, assets, gui, beam_width, settle_steps, seed, layers=None):
    meta = json.loads(Path(path).read_text())
    layers = resolve_layers(len(meta['objects']), layers)
    task = DatasetSceneTask(meta)
    env = Environment(str(assets), task=task, disp=gui, hz=240)
    placed, actions = [], []
    start_time = time.perf_counter()
    try:
        env.seed(seed)
        env.reset()
        if gui:
            p.resetDebugVisualizerCamera(cameraDistance=1.25, cameraYaw=45,
                cameraPitch=-65, cameraTargetPosition=[0.5, 0., 0.05])
        for step in range(len(meta['objects'])):
            action = beam_plan(task, env, placed, beam_width, settle_steps, layers)
            if action is None:
                print(f'  step {step}: Beam Search found no feasible action', flush=True)
                break
            oid = action['object_id']
            print(f'  step {step}: Beam Search chose object={oid}, '
                  f'place_xy={action["place"][0][:2]}', flush=True)
            timeout = task.primitive(env.movej, env.movep, env.ee,
                                     action['pick'], action['place'])
            settle(180)
            geo = {int(o['id']): o for o in meta['objects']}
            stable = pyramid_stable(placed + [oid], layers, geo, task.bodies)
            ok = bool(not timeout and task.primitive.last_grasp and stable)
            actions.append({'object_id': oid, 'grasp_attached': bool(task.primitive.last_grasp),
                            'timeout': bool(timeout), 'stack_stable': bool(stable),
                            'success': ok, 'beam_score': action['score']})
            print(f'    grasp={task.primitive.last_grasp}, stable={stable}, success={ok}', flush=True)
            if not ok:
                break
            placed.append(oid)
            if gui:
                time.sleep(0.3)
        return {'episode': path.parent.name, 'layers': layers, 'stacked': len(placed),
                'trials': len(actions), 'seconds': time.perf_counter()-start_time,
                'total': len(meta['objects']), 'success': len(placed) == len(meta['objects']),
                'order_chosen_by_beam': placed, 'actions': actions}
    finally:
        env.close()


def report(results, output):
    """Generate comparable CEM-style episode and aggregate evaluation files."""
    rows = []
    for result in results:
        total = result.get('total', 0)
        stacked = result.get('stacked', 0)
        error = result.get('error', '')
        reward = float(stacked / total) if total and not error else None
        rows.append({
            'episode': result['episode'],
            'total': total, 'stacked': stacked,
            'reward': reward,
            'trials': result.get('trials', len(result.get('actions', []))),
            'seconds': round(result.get('seconds', 0.), 3),
            'success': bool(result.get('success', False)),
            'error': error,
        })

    valid = [r for r in rows if r['reward'] is not None]
    rewards = np.asarray([r['reward'] for r in valid], dtype=float)
    mean = float(np.mean(rewards)) if len(rewards) else float('nan')
    if len(rewards) >= 2:
        half_width = 1.96 * float(np.std(rewards, ddof=1)) / math.sqrt(len(rewards))
        lower, upper = max(0., mean-half_width), min(1., mean+half_width)
        ci = f'[{lower:.3f}, {upper:.3f}]'
    else:
        lower = upper = float('nan')
        ci = 'N/A'
    count = len(rows)
    avg_trials = sum(r['trials'] for r in rows) / count if count else 0.
    summary = {
        'method': 'Beam hetero pyramid', 'episodes': count,
        'valid_episodes': len(valid), 'avg_trials': round(avg_trials, 3),
        'mean_reward': round(mean, 5),
        'ci95_lower': round(lower, 5), 'ci95_upper': round(upper, 5),
        'full_stacks': sum(r['reward'] >= 1.-1e-9 for r in valid),
        'partial_stacks': sum(0. < r['reward'] < 1.-1e-9 for r in valid),
        'zero_reward': sum(r['reward'] == 0 for r in valid),
        'errors': count-len(valid),
    }
    print('\n' + '='*110)
    print(f'{"Method":<19} {"Episodes":>8} {"Avg.Trials":>10} {"Mean Reward":>12} '
          f'{"95% CI":>16} {"Full":>6} {"Partial":>8} {"Zero":>6} {"Errors":>7}')
    print(f'{summary["method"]:<19} {count:>8} {avg_trials:>10.2f} {mean:>12.3f} '
          f'{ci:>16} {summary["full_stacks"]:>6} {summary["partial_stacks"]:>8} '
          f'{summary["zero_reward"]:>6} {summary["errors"]:>7}')
    print('='*110)
    summary_path = output.with_name(output.stem + '_summary.csv')
    episodes_path = output.with_name(output.stem + '_episodes.csv')
    for path, records in [(summary_path, [summary]), (episodes_path, rows)]:
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]) if records else
                                    ['episode','total','stacked','reward','trials','seconds','success','error'])
            writer.writeheader()
            writer.writerows(records)
    print(f'Results: {output}\nSummary: {summary_path}\nEpisodes: {episodes_path}')
    print('Reward = executed stacking completion fraction (not native Ravens task reward).')
    print('Trials = attempted UR5 pick/place actions; CI uses valid episodes only.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--layers', type=int, choices=[2, 3, 4], default=None,
                    help='Required layers; auto-inferred for 3, 6, or 10 objects')
    ap.add_argument('--data', type=Path, default=Path('stacking_data_validated'))
    ap.add_argument('--assets', type=Path, default=Path('ravens/environments/assets'))
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--beam-width', type=int, default=4)
    ap.add_argument('--settle-steps', type=int, default=90)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--out', type=Path, default=Path('results/beam_hetero_results.json'))
    args = ap.parse_args()
    if args.episodes < 1 or args.beam_width < 1 or args.settle_steps < 1:
        ap.error('episodes, beam-width, and settle-steps must be positive')
    files = [f for f in sorted(args.data.glob('episode_*/episode.json'))
             if (len(json.loads(f.read_text())['objects']) in (3, 6, 10)
                 and (args.layers is None or
                      len(json.loads(f.read_text())['objects']) == args.layers*(args.layers+1)//2))][:args.episodes]
    if not files:
        ap.error(f'No episode_*/episode.json files in {args.data}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for index, path in enumerate(files):
        started = time.perf_counter()
        print(f'\nEpisode {index+1}/{len(files)}: {path.parent.name}', flush=True)
        try:
            results.append(run_episode(path, args.assets, args.gui,
                                       args.beam_width, args.settle_steps, args.seed + index, args.layers))
        except Exception as exc:
            print(f'ERROR {path.parent.name}: {type(exc).__name__}: {exc}', flush=True)
            traceback.print_exc()
            total = len(json.loads(path.read_text()).get('objects', []))
            results.append({'episode': path.parent.name, 'total': total, 'stacked': 0,
                            'trials': 0, 'seconds': time.perf_counter()-started,
                            'success': False, 'error': f'{type(exc).__name__}: {exc}'})
        args.out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    report(results, args.out)


if __name__ == '__main__':
    main()
