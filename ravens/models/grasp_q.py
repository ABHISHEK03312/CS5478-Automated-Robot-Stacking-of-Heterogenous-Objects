"""Dense RGB-D Q model for planar parallel-jaw grasp selection."""

import os

import numpy as np
from ravens.grasping.encoding import GraspActionSpace
from ravens.models.resnet import ResNet43_8s
from ravens.utils import utils
import tensorflow as tf


class GraspQModel:
  """Predict Q(pixel, yaw, width) from a fused RGB-D heightmap."""

  def __init__(self, in_shape, action_space=None, learning_rate=1e-4):
    self.in_shape = tuple(in_shape)
    self.action_space = action_space or GraspActionSpace()
    image, logits = ResNet43_8s(
        self.in_shape, self.action_space.size, prefix='grasp_q_')
    self.model = tf.keras.Model(inputs=image, outputs=logits)
    self.optim = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    self.total_steps = 0

  @staticmethod
  def _prepare_image(image):
    image = np.asarray(image, dtype=np.float32)
    return utils.preprocess(image)[None, ...]

  def forward(self, image):
    """Return probabilities with shape (H, W, rotation, width)."""
    logits = self.model(self._prepare_image(image), training=False)[0]
    shape = tuple(logits.shape[:2]) + (
        self.action_space.n_rotations, self.action_space.n_widths)
    return tf.math.sigmoid(tf.reshape(logits, shape)).numpy()

  def score(self, image, pixel, yaw, width):
    action_bin = self.action_space.encode(yaw, width)
    logits = self.model(self._prepare_image(image), training=False)[0]
    logit = logits[int(pixel[0]), int(pixel[1]), action_bin]
    return float(tf.math.sigmoid(logit))

  def train(self, sample):
    """Update from one attempted grasp and return binary cross-entropy."""
    sample.validate()
    action_bin = self.action_space.encode(sample.yaw, sample.width)
    row, col = map(int, sample.pixel)
    target = tf.convert_to_tensor(float(sample.success), dtype=tf.float32)
    image = self._prepare_image(sample.image)
    with tf.GradientTape() as tape:
      logits = self.model(image, training=True)[0]
      selected = logits[row, col, action_bin]
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=target, logits=selected)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optim.apply_gradients(zip(gradients, self.model.trainable_variables))
    self.total_steps += 1
    return float(loss)

  def best(self, image, count=1, border=0):
    """Return the highest-scoring grasp candidates."""
    quality = self.forward(image)
    if border:
      quality[:border, ...] = -np.inf
      quality[-border:, ...] = -np.inf
      quality[:, :border, ...] = -np.inf
      quality[:, -border:, ...] = -np.inf
    count = max(1, min(int(count), quality.size))
    flat = quality.reshape(-1)
    indices = np.argpartition(flat, -count)[-count:]
    indices = indices[np.argsort(flat[indices])[::-1]]
    candidates = []
    for index in indices:
      row, col, rotation_bin, width_bin = np.unravel_index(
          index, quality.shape)
      action_bin = rotation_bin * self.action_space.n_widths + width_bin
      yaw, width = self.action_space.decode(action_bin)
      candidates.append({
          'pixel': (int(row), int(col)),
          'yaw': yaw,
          'width': width,
          'quality': float(flat[index]),
      })
    return candidates

  def save(self, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    self.model.save_weights(path)

  def load(self, path):
    self.model.load_weights(path)
