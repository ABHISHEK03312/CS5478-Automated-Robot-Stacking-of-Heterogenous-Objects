"""Collect counterfactual rewards for CEM's near-tied placement candidates."""

import os

from absl import app
from absl import flags
import numpy as np

from ravens import tasks
from ravens.agents.cem import CEMAgent
from ravens.dataset import BOUNDS
from ravens.dataset import PIXEL_SIZE
from ravens.environments.environment import Environment
from ravens.grasping.observations import fused_heightmap
from ravens.grasping.stack_replay import StackActionReplayDataset
from ravens.grasping.stack_replay import StackActionTransition
from ravens.tasks.grippers import ParallelJaw
from ravens.utils import utils


flags.DEFINE_string('assets_root', './ravens/environments/assets/', '')
flags.DEFINE_string('output_dir', './data/cem-pair-replay', '')
flags.DEFINE_string('task', 'stack-block-pyramid-parallel-jaw', '')
flags.DEFINE_integer('episodes', 10, '')
flags.DEFINE_integer('seed', 53000, '')
flags.DEFINE_integer('candidates', 5, '')
flags.DEFINE_integer('n_samples', 24, '')
flags.DEFINE_integer('n_elites', 5, '')
flags.DEFINE_integer('n_iters', 3, '')
flags.DEFINE_integer('verify_k', 3, '')
flags.DEFINE_bool('disp', False, '')
FLAGS = flags.FLAGS


def transition_for(action, image, next_image, reward, done, metadata):
  pick_pixel = np.asarray(utils.xyz_to_pix(
      action['pose0'][0], BOUNDS, PIXEL_SIZE), dtype=np.int32)
  if not (0 <= pick_pixel[0] < image.shape[0] and
          0 <= pick_pixel[1] < image.shape[1]):
    return None
  return StackActionTransition(
      image=image, pick_pixel=pick_pixel,
      pick_yaw=float(utils.quatXYZW_to_eulerXYZ(action['pose0'][1])[2]),
      width=0.048, place_position=np.asarray(action['pose1'][0]),
      place_yaw=float(utils.quatXYZW_to_eulerXYZ(action['pose1'][1])[2]),
      reward=float(reward), next_image=next_image, done=bool(done),
      metadata=metadata)


def main(unused_argv):
  del unused_argv
  dataset = StackActionReplayDataset(FLAGS.output_dir)
  env = Environment(FLAGS.assets_root, disp=FLAGS.disp, hz=480)
  task = tasks.names[FLAGS.task]()
  if task.ee is not ParallelJaw:
    raise ValueError('CEM pair replay requires a ParallelJaw task')
  task.mode = 'train'
  try:
    for episode in range(FLAGS.episodes):
      scene_seed = FLAGS.seed + episode
      np.random.seed(scene_seed)
      env.seed(scene_seed)
      env.set_task(task)
      obs, info = env.reset(), None
      agent = CEMAgent(
          env, task, n_samples=FLAGS.n_samples, n_elites=FLAGS.n_elites,
          n_iters=FLAGS.n_iters, verify_k=FLAGS.verify_k,
          record_candidates=FLAGS.candidates, seed=scene_seed, verbose=False)
      for step in range(task.max_steps):
        action = agent.act(obs, info)
        if action is None:
          break
        image = fused_heightmap(obs)
        group = f'{scene_seed}:{step}'
        rewards = []
        for rank, candidate in enumerate(agent.last_candidates):
          snapshot = agent._snapshot()  # pylint: disable=protected-access
          next_obs, reward, done, _ = env.step(candidate['action'])
          transition = transition_for(
              candidate['action'], image, fused_heightmap(next_obs), reward,
              done, {'episode': episode, 'step': step, 'seed': scene_seed,
                     'candidate_group': group, 'candidate_rank': rank,
                     'physics_score': candidate['physics_score']})
          if transition is not None:
            dataset.add(transition)
            rewards.append(float(reward))
          agent._restore(snapshot)  # pylint: disable=protected-access
        print(f'episode={episode + 1}/{FLAGS.episodes} step={step + 1} '
              f'candidates={len(rewards)} rewards={np.round(rewards, 3)}',
              flush=True)
        obs, _, done, info = env.step(action)
        if done:
          break
  finally:
    env.close()
  print(f'Saved {len(dataset)} CEM candidate transitions to {dataset.path}')


if __name__ == '__main__':
  app.run(main)
