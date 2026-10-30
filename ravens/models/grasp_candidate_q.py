"""Candidate-conditioned RGB-D grasp success model."""

import os

import numpy as np
from ravens.grasping.crops import candidate_crop
from ravens.utils import utils
import tensorflow as tf


class GraspCandidateQModel:
  """Predict success for physically proposed (pixel, yaw, width) grasps."""

  def __init__(self, image_channels=6, crop_size=48, min_width=0.015,
               max_width=0.12, learning_rate=1e-4):
    self.crop_size = int(crop_size)
    self.min_width = float(min_width)
    self.max_width = float(max_width)
    crop = tf.keras.Input((self.crop_size, self.crop_size, image_channels),
                          name='candidate_crop')
    width = tf.keras.Input((1,), name='grasp_width')
    x = crop
    for filters in (32, 64, 96):
      x = tf.keras.layers.Conv2D(filters, 3, padding='same',
                                 activation='relu')(x)
      x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Concatenate()([x, width])
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    logit = tf.keras.layers.Dense(1)(x)
    self.model = tf.keras.Model((crop, width), logit)
    self.optim = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    self.total_steps = 0

  def _width_feature(self, width):
    scale = max(self.max_width - self.min_width, 1e-6)
    return np.float32(np.clip((width - self.min_width) / scale, 0, 1))

  def prepare(self, sample):
    """Return a normalized candidate crop and normalized width feature."""
    sample.validate()
    prepared = utils.preprocess(np.array(sample.image, copy=True))
    crop = candidate_crop(prepared, sample.pixel, sample.yaw, self.crop_size)
    return crop, self._width_feature(sample.width)

  def prepare_candidate(self, image, candidate):
    prepared = utils.preprocess(np.array(image, dtype=np.float32, copy=True))
    crop = candidate_crop(prepared, candidate['pixel'], candidate['yaw'],
                          self.crop_size)
    return crop, self._width_feature(candidate['width'])

  def train_batch(self, samples):
    crops, widths, labels = [], [], []
    for sample in samples:
      crop, width = self.prepare(sample)
      crops.append(crop)
      widths.append([width])
      labels.append([float(sample.success)])
    inputs = (np.asarray(crops), np.asarray(widths, dtype=np.float32))
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    with tf.GradientTape() as tape:
      logits = self.model(inputs, training=True)
      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits))
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))
    self.total_steps += 1
    return float(loss)

  def score_candidates(self, image, candidates, mc_samples=8,
                       uncertainty_weight=1.0):
    """Score candidates conservatively as mean probability minus uncertainty."""
    if not candidates:
      return []
    crops, widths = zip(*(self.prepare_candidate(image, c)
                          for c in candidates))
    inputs = (np.asarray(crops), np.asarray(widths, dtype=np.float32)[:, None])
    passes = max(1, int(mc_samples))
    probabilities = []
    for _ in range(passes):
      logits = self.model(inputs, training=passes > 1)
      probabilities.append(tf.math.sigmoid(logits).numpy()[:, 0])
    probabilities = np.asarray(probabilities)
    means = probabilities.mean(axis=0)
    stds = probabilities.std(axis=0)
    ranked = []
    for candidate, mean, std in zip(candidates, means, stds):
      ranked.append({**candidate, 'probability': float(mean),
                     'uncertainty': float(std),
                     'quality': float(mean - uncertainty_weight * std)})
    return sorted(ranked, key=lambda item: item['quality'], reverse=True)

  def score(self, sample, mc_samples=1):
    return self.score_candidates(
        sample.image, [{'pixel': tuple(sample.pixel), 'yaw': sample.yaw,
                        'width': sample.width}], mc_samples=mc_samples)[0]

  def save(self, path):
    directory = os.path.dirname(path)
    if directory:
      os.makedirs(directory, exist_ok=True)
    self.model.save_weights(path)

  def load(self, path):
    self.model.load_weights(path)
