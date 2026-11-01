"""RGB-D value model conditioned on a complete pick-and-place action."""

import os

import numpy as np
from ravens.dataset import BOUNDS
from ravens.grasping.crops import candidate_crop
from ravens.utils import utils
import tensorflow as tf


class StackActionQModel:
  """Predict discounted stacking return for one pick-and-place action."""

  def __init__(self, image_channels=6, crop_size=48, learning_rate=1e-4):
    self.crop_size = int(crop_size)
    crop = tf.keras.Input((self.crop_size, self.crop_size, image_channels))
    action = tf.keras.Input((5,))
    x = crop
    for filters in (32, 64, 96):
      x = tf.keras.layers.Conv2D(filters, 3, padding='same',
                                 activation='relu')(x)
      x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Concatenate()([x, action])
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    self.model = tf.keras.Model((crop, action), tf.keras.layers.Dense(1)(x))
    self.optim = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

  @staticmethod
  def action_features(width, place_position, place_yaw):
    position = np.asarray(place_position, dtype=np.float32)
    spans = np.maximum(BOUNDS[:, 1] - BOUNDS[:, 0], 1e-6)
    normalized_position = (position - BOUNDS[:, 0]) / spans
    return np.asarray((
        np.clip(width / 0.15, 0, 1),
        np.clip(normalized_position[0], 0, 1),
        np.clip(normalized_position[1], 0, 1),
        np.clip(normalized_position[2], 0, 1),
        float(place_yaw % np.pi) / np.pi,
    ), dtype=np.float32)

  def prepare(self, transition):
    transition.validate()
    image = utils.preprocess(np.array(transition.image, copy=True))
    crop = candidate_crop(image, transition.pick_pixel,
                          transition.pick_yaw, self.crop_size)
    action = self.action_features(
        transition.width, transition.place_position, transition.place_yaw)
    return crop, action

  def train_targets(self, transitions, targets):
    if not transitions or len(transitions) != len(targets):
      raise ValueError('transitions and targets must have equal nonzero length')
    prepared = [self.prepare(transition) for transition in transitions]
    crops = np.asarray([item[0] for item in prepared])
    actions = np.asarray([item[1] for item in prepared], dtype=np.float32)
    labels = tf.convert_to_tensor(
        np.clip(np.asarray(targets, dtype=np.float32), 0, 1)[:, None])
    with tf.GradientTape() as tape:
      logits = self.model((crops, actions), training=True)
      loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits))
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))
    return float(loss)

  def train_pairs(self, preferred, rejected):
    """Train preferred actions to have higher logits than rejected actions."""
    if not preferred or len(preferred) != len(rejected):
      raise ValueError('preferred and rejected must have equal nonzero length')
    preferred_prepared = [self.prepare(item) for item in preferred]
    rejected_prepared = [self.prepare(item) for item in rejected]
    preferred_crops = np.asarray([item[0] for item in preferred_prepared])
    preferred_actions = np.asarray(
        [item[1] for item in preferred_prepared], dtype=np.float32)
    rejected_crops = np.asarray([item[0] for item in rejected_prepared])
    rejected_actions = np.asarray(
        [item[1] for item in rejected_prepared], dtype=np.float32)
    with tf.GradientTape() as tape:
      preferred_logits = self.model(
          (preferred_crops, preferred_actions), training=False)
      rejected_logits = self.model(
          (rejected_crops, rejected_actions), training=False)
      loss = tf.reduce_mean(tf.nn.softplus(
          -(preferred_logits - rejected_logits)))
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))
    return float(loss)

  def predict(self, transitions):
    prepared = [self.prepare(transition) for transition in transitions]
    crops = np.asarray([item[0] for item in prepared])
    actions = np.asarray([item[1] for item in prepared], dtype=np.float32)
    return tf.math.sigmoid(
        self.model((crops, actions), training=False)).numpy()[:, 0]

  def save(self, path):
    directory = os.path.dirname(path)
    if directory:
      os.makedirs(directory, exist_ok=True)
    self.model.save_weights(path)

  def load(self, path):
    self.model.load_weights(path)
