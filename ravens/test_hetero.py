# coding=utf-8
"""Evaluate a trained Ravens Transporter on held-out mixed-stacking episodes.

This evaluates predicted pick/place poses against the ACTUAL saved test actions.
It does not simulate the UR5 or measure physical grasp/stack success.

Run from the repository root:
  python -m ravens.test_hetero --data_dir=data --root_dir=. \
      --task=stack-block-pyramid-parallel-jaw --agent=transporter \
      --n_demos=100 --n_steps=1000

Put this file at ravens/test_hetero.py.
"""

import csv
import os

from absl import app, flags
import numpy as np
import tensorflow as tf

from ravens import agents
from ravens.dataset import Dataset

flags.DEFINE_string('root_dir', '.', 'Directory containing checkpoints and results.')
flags.DEFINE_string('data_dir', 'data', 'Parent of Ravens datasets.')
flags.DEFINE_string('dataset_name', 'mixed-stacking-validated-test',
                    'Exact name of the held-out Ravens-format dataset folder.')
flags.DEFINE_string('task', 'stack-block-pyramid-parallel-jaw',
                    'Task name used when TRAINING (also determines checkpoint name).')
flags.DEFINE_string('agent', 'transporter', 'Agent type.')
flags.DEFINE_integer('n_demos', 100, 'Training demonstration count in checkpoint name.')
flags.DEFINE_integer('n_steps', 1000, 'Checkpoint training step to load.')
flags.DEFINE_integer('n_runs', 1, 'Number of training runs to evaluate.')
flags.DEFINE_integer('gpu', 0, 'GPU index, if available.')
flags.DEFINE_integer('gpu_limit', None, 'Optional GPU memory limit in GiB.')
flags.DEFINE_integer('max_episodes', 0, '0 uses all held-out episodes.')
flags.DEFINE_string('output_dir', 'results/hetero', 'Directory for CSV summaries.')
FLAGS = flags.FLAGS


def configure_tf():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print('No TensorFlow GPU detected; evaluating on CPU.', flush=True)
        return
    if not 0 <= FLAGS.gpu < len(gpus):
        raise ValueError('GPU index out of range: {}'.format(FLAGS.gpu))
    cfg = tf.config.experimental
    if FLAGS.gpu_limit is not None:
        cfg.set_virtual_device_configuration(
            gpus[FLAGS.gpu],
            [cfg.VirtualDeviceConfiguration(memory_limit=1024 * FLAGS.gpu_limit)])
    cfg.set_visible_devices(gpus[FLAGS.gpu], 'GPU')


def position(action, key):
    if not isinstance(action, dict) or key not in action:
        raise ValueError('Action missing {}: {!r}'.format(key, action))
    p = np.asarray(action[key][0], dtype=np.float64)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise ValueError('Invalid {} position: {!r}'.format(key, p))
    return p


def main(argv):
    del argv
    configure_tf()
    path = os.path.join(FLAGS.data_dir, FLAGS.dataset_name)
    if not os.path.isdir(path):
        raise FileNotFoundError('Test dataset folder not found: ' + path)
    ds = Dataset(path)
    n = min(ds.n_episodes, FLAGS.max_episodes) if FLAGS.max_episodes > 0 else ds.n_episodes
    if n == 0:
        raise ValueError('Test dataset has no episodes: ' + path)
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    print('TEST DATASET:', path, '| episodes:', n, flush=True)

    for run in range(FLAGS.n_runs):
        np.random.seed(run)
        tf.random.set_seed(run)
        name = '{}-{}-{}-{}'.format(FLAGS.task, FLAGS.agent, FLAGS.n_demos, run)
        agent = agents.names[FLAGS.agent](name, FLAGS.task, FLAGS.root_dir)
        print('Loading checkpoint:', name, 'step:', FLAGS.n_steps, flush=True)
        agent.load(FLAGS.n_steps)
        rows = []

        for episode_idx in range(n):
            episode, seed = ds.load(episode_idx)
            goal = episode[-1]  # Same goal format passed by the original ravens/test.py.
            print('Test episode {}/{} (seed {})'.format(episode_idx + 1, n, seed), flush=True)
            for step_idx, transition in enumerate(episode):
                obs, target_act, _, info = transition
                if target_act is None:  # Terminal record.
                    continue
                predicted = agent.act(obs, info, goal)
                if predicted is None:
                    print('  No prediction at step', step_idx)
                    continue
                pick_error = float(np.linalg.norm(position(predicted, 'pose0') -
                                                  position(target_act, 'pose0')))
                place_error = float(np.linalg.norm(position(predicted, 'pose1') -
                                                   position(target_act, 'pose1')))
                rows.append({
                    'run': run,
                    'episode': episode_idx,
                    'seed': seed,
                    'step': step_idx,
                    'pick_error_m': pick_error,
                    'place_error_m': place_error,
                })
                print('  step {}: pick error {:.4f} m; place error {:.4f} m'.format(
                    step_idx, pick_error, place_error), flush=True)

        if not rows:
            raise ValueError('No test actions were evaluated.')
        output = os.path.join(FLAGS.output_dir,
                              '{}-{}-offline.csv'.format(name, FLAGS.n_steps))
        with open(output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        pick = np.asarray([r['pick_error_m'] for r in rows])
        place = np.asarray([r['place_error_m'] for r in rows])
        print('\nOFFLINE TEST RESULTS (run {})'.format(run), flush=True)
        print('Evaluated actions:', len(rows))
        print('Mean pick position error: {:.4f} m'.format(np.mean(pick)))
        print('Mean place position error: {:.4f} m'.format(np.mean(place)))
        print('Median pick position error: {:.4f} m'.format(np.median(pick)))
        print('Median place position error: {:.4f} m'.format(np.median(place)))
        print('Saved:', output)
        print('NOTE: This does NOT measure gripper or stacking success.')


if __name__ == '__main__':
    app.run(main)
