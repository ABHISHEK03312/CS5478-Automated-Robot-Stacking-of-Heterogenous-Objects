"""Evaluate candidate-conditioned grasp Q on held-out lift attempts."""

import json
import os

from absl import app, flags
from ravens.grasping.dataset import GraspDataset
from ravens.grasping.metrics import binary_grasp_metrics
from ravens.models.grasp_candidate_q import GraspCandidateQModel
from ravens.train_grasp_q import configure_tensorflow


flags.DEFINE_string('candidate_eval_data_dir',
                    './data/rgbd-grasp-attempts-validation', '')
flags.DEFINE_string('candidate_checkpoint', None, '')
flags.DEFINE_string('candidate_eval_output', '', '')
flags.DEFINE_integer('candidate_eval_crop_size', 48, '')
FLAGS = flags.FLAGS


def main(unused_argv):
  del unused_argv
  if not FLAGS.candidate_checkpoint:
    raise ValueError('--candidate_checkpoint is required')
  configure_tensorflow()
  dataset = GraspDataset(FLAGS.candidate_eval_data_dir)
  model = GraspCandidateQModel(
      dataset.load(0).image.shape[2], FLAGS.candidate_eval_crop_size)
  model.load(FLAGS.candidate_checkpoint)
  labels, probabilities, uncertainties = [], [], []
  for index in range(len(dataset)):
    sample = dataset.load(index)
    result = model.score(sample, mc_samples=8)
    labels.append(int(sample.success))
    probabilities.append(result['probability'])
    uncertainties.append(result['uncertainty'])
  metrics = binary_grasp_metrics(labels, probabilities)
  metrics['mean_uncertainty'] = sum(uncertainties) / len(uncertainties)
  metrics['checkpoint'] = os.path.abspath(FLAGS.candidate_checkpoint)
  print(json.dumps(metrics, indent=2, sort_keys=True))
  if FLAGS.candidate_eval_output:
    output = os.path.abspath(FLAGS.candidate_eval_output)
    directory = os.path.dirname(output)
    if directory:
      os.makedirs(directory, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as stream:
      json.dump(metrics, stream, indent=2, sort_keys=True)
      stream.write('\n')


if __name__ == '__main__':
  app.run(main)
