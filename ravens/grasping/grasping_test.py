"""Tests for RGB-D grasp action encoding and dataset storage."""

import tempfile

from absl.testing import absltest
import numpy as np
from ravens.grasping.dataset import GraspDataset
from ravens.grasping.dataset import GraspSample
from ravens.grasping.encoding import GraspActionSpace


class GraspActionSpaceTest(absltest.TestCase):

  def test_parallel_jaw_yaw_is_pi_periodic(self):
    space = GraspActionSpace(n_rotations=16)
    self.assertEqual(space.rotation_bin(0.3), space.rotation_bin(0.3 + np.pi))

  def test_encode_decode_stays_in_requested_bins(self):
    space = GraspActionSpace(n_rotations=8, n_widths=4,
                             min_width=0.02, max_width=0.10)
    encoded = space.encode(0.7, 0.061)
    yaw, width = space.decode(encoded)
    self.assertEqual(space.encode(yaw, width), encoded)


class GraspDatasetTest(absltest.TestCase):

  def test_round_trip(self):
    with tempfile.TemporaryDirectory() as directory:
      dataset = GraspDataset(directory)
      original = GraspSample(
          image=np.zeros((12, 8, 6), dtype=np.float32),
          pixel=np.asarray([4, 5]),
          yaw=0.4,
          width=0.052,
          success=True,
          metadata={'seed': 7})
      dataset.add(original)
      loaded = dataset.load(0)
      np.testing.assert_array_equal(loaded.image, original.image)
      np.testing.assert_array_equal(loaded.pixel, original.pixel)
      self.assertAlmostEqual(loaded.yaw, original.yaw, places=6)
      self.assertAlmostEqual(loaded.width, original.width, places=6)
      self.assertTrue(loaded.success)
      self.assertEqual(loaded.metadata, {'seed': 7})

  def test_rejects_out_of_bounds_pixel(self):
    sample = GraspSample(
        image=np.zeros((4, 4, 6), dtype=np.float32),
        pixel=np.asarray([4, 0]), yaw=0, width=0.04,
        success=False, metadata={})
    with self.assertRaisesRegex(ValueError, 'outside'):
      sample.validate()


if __name__ == '__main__':
  absltest.main()
