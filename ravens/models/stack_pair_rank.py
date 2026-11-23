"""Candidate-relative RGB-D ranker for near-tied CEM actions."""

import os

import numpy as np
from ravens.grasping.crops import candidate_crop
from ravens.models.stack_action_q import StackActionQModel
from ravens.utils import utils
import tensorflow as tf


class StackActionPairRankModel:
  """Predict whether one complete action should outrank another."""

  def __init__(self, image_channels=6, crop_size=48, learning_rate=1e-4):
    self.crop_size = int(crop_size)
    crop = tf.keras.Input((self.crop_size, self.crop_size, image_channels))
    pair = tf.keras.Input((16,))
    x = crop
    for filters in (32, 64, 96):
      x = tf.keras.layers.Conv2D(filters, 3, padding='same',
                                 activation='relu')(x)
      x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Concatenate()([x, pair])
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    self.model = tf.keras.Model((crop, pair), tf.keras.layers.Dense(1)(x))
    self.optim = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

  @staticmethod
  def pair_features(first, second, first_physics, second_physics):
    first_action = StackActionQModel.action_features(
        first.width, first.place_position, first.place_yaw)
    second_action = StackActionQModel.action_features(
        second.width, second.place_position, second.place_yaw)
    scale = np.asarray((5, 10, 10, 10, 5), dtype=np.float32)
    relative = np.clip((first_action - second_action) * scale, -1, 1)
    physics_delta = np.tanh(float(first_physics) - float(second_physics))
    return np.concatenate((first_action, second_action, relative,
                           np.asarray([physics_delta], dtype=np.float32)))

  def prepare_pair(self, first, second, first_physics, second_physics):
    first.validate()
    second.validate()
    image = utils.preprocess(np.array(first.image, copy=True))
    crop = candidate_crop(
        image, first.pick_pixel, first.pick_yaw, self.crop_size)
    features = self.pair_features(
        first, second, first_physics, second_physics)
    return crop, features

  def train_pairs(self, preferred, rejected):
    if not preferred or len(preferred) != len(rejected):
      raise ValueError('preferred and rejected must have equal nonzero length')
    forward = [self.prepare_pair(
        first, second, first.metadata['physics_score'],
        second.metadata['physics_score'])
               for first, second in zip(preferred, rejected)]
    reverse = [self.prepare_pair(
        second, first, second.metadata['physics_score'],
        first.metadata['physics_score'])
               for first, second in zip(preferred, rejected)]
    examples = forward + reverse
    crops = np.asarray([item[0] for item in examples])
    features = np.asarray([item[1] for item in examples], dtype=np.float32)
    labels = tf.convert_to_tensor(
        np.concatenate((np.ones(len(forward)), np.zeros(len(reverse))))[:, None],
        dtype=tf.float32)
    with tf.GradientTape() as tape:
      logits = self.model((crops, features), training=True)
      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits))
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))
    return float(loss)

  def predict_pairs(self, first, second, first_physics, second_physics):
    prepared = [self.prepare_pair(a, b, a_score, b_score)
                for a, b, a_score, b_score in
                zip(first, second, first_physics, second_physics)]
    crops = np.asarray([item[0] for item in prepared])
    features = np.asarray([item[1] for item in prepared], dtype=np.float32)
    return tf.math.sigmoid(
        self.model((crops, features), training=False)).numpy()[:, 0]

  def save(self, path):
    directory = os.path.dirname(path)
    if directory:
      os.makedirs(directory, exist_ok=True)
    self.model.save_weights(path)

  def load(self, path):
    self.model.load_weights(path)

  def load_backbone(self, path):
    """Initialize matching RGB-D layers from a scalar-Q checkpoint."""
    self.model.load_weights(path, by_name=True, skip_mismatch=True)
