"""Tests for RGB-D grasp action encoding and dataset storage."""

import tempfile

from absl.testing import absltest
import numpy as np
from ravens.grasping.dataset import GraspDataset
from ravens.grasping.dataset import GraspSample
from ravens.grasping.encoding import GraspActionSpace
from ravens.grasping.metrics import binary_grasp_metrics
from ravens.grasping.selection import component_center_mask
from ravens.grasping.selection import component_grasp_candidates
from ravens.grasping.selection import rank_grasp_candidates
from ravens.grasping.selection import select_best_grasp
from ravens.grasping.selection import select_random_grasp
from ravens.models.grasp_q import GraspQModel
from ravens.tasks.primitives import PickPlaceParallelJaw


class _FakeEndEffector:

  def __init__(self, attaches=True, lift_height=0.08):
    self.attaches = attaches
    self.lift_height = lift_height
    self.secured = False
    self.position_reads = 0

  def activate(self):
    self.secured = self.attaches

  def release(self):
    self.secured = False

  def check_grasp(self):
    return self.secured

  def detect_contact(self):
    return self.secured

  def grasped_object_id(self):
    return 9 if self.secured else None

  def grasped_object_position(self):
    if not self.secured:
      return None
    height = 0.02 if self.position_reads == 0 else 0.02 + self.lift_height
    self.position_reads += 1
    return (0.4, 0.1, height)


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


class GraspMetricsTest(absltest.TestCase):

  def test_perfect_binary_predictions(self):
    metrics = binary_grasp_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    self.assertEqual(metrics['accuracy'], 1.0)
    self.assertEqual(metrics['balanced_accuracy'], 1.0)
    self.assertEqual(metrics['roc_auc'], 1.0)

  def test_auc_handles_ties(self):
    metrics = binary_grasp_metrics([0, 1], [0.5, 0.5])
    self.assertEqual(metrics['roc_auc'], 0.5)


class GraspSelectionTest(absltest.TestCase):

  def test_component_centers_propose_one_pixel_per_surface(self):
    surface = np.zeros((8, 10), dtype=bool)
    surface[1:4, 1:4] = True
    surface[5:7, 7:10] = True
    centers = component_center_mask(surface, min_pixels=3)
    self.assertEqual(int(np.sum(centers)), 2)
    self.assertTrue(np.all(surface[centers]))

  def test_component_grasp_candidate_uses_principal_axis(self):
    surface = np.zeros((12, 12), dtype=bool)
    surface[4:7, 2:10] = True
    candidates = component_grasp_candidates(surface, pixel_size=0.01)
    self.assertLen(candidates, 1)
    self.assertAlmostEqual(candidates[0]['yaw'], 0.0, places=5)
    self.assertGreater(candidates[0]['width'], 0.02)

  def test_rank_candidates_uses_corresponding_action_value(self):
    space = GraspActionSpace(n_rotations=4, n_widths=2)
    quality = np.zeros((4, 4, 4, 2), dtype=np.float32)
    candidates = [
        {'pixel': (1, 1), 'yaw': 0.1, 'width': 0.04, 'quality': None},
        {'pixel': (2, 2), 'yaw': 1.0, 'width': 0.08, 'quality': None},
    ]
    for index, candidate in enumerate(candidates):
      action = space.encode(candidate['yaw'], candidate['width'])
      rotation, width = divmod(action, space.n_widths)
      quality[candidate['pixel'][0], candidate['pixel'][1], rotation, width] = (
          0.4 + 0.4 * index)
    best = rank_grasp_candidates(quality, candidates, space)
    self.assertEqual(best['pixel'], (2, 2))
    self.assertAlmostEqual(best['quality'], 0.8)

  def test_best_grasp_respects_valid_mask(self):
    space = GraspActionSpace(n_rotations=2, n_widths=2)
    quality = np.zeros((3, 4, 2, 2), dtype=np.float32)
    quality[0, 0, 0, 0] = 1.0
    quality[2, 3, 1, 1] = 0.8
    mask = np.zeros((3, 4), dtype=bool)
    mask[2, 3] = True
    grasp = select_best_grasp(quality, space, mask)
    self.assertEqual(grasp['pixel'], (2, 3))
    self.assertAlmostEqual(grasp['quality'], 0.8)

  def test_best_grasp_respects_action_coverage(self):
    space = GraspActionSpace(n_rotations=2, n_widths=2)
    quality = np.zeros((2, 2, 2, 2), dtype=np.float32)
    quality[1, 1, 0, 0] = 1.0
    quality[1, 1, 1, 1] = 0.7
    pixels = np.zeros((2, 2), dtype=bool)
    pixels[1, 1] = True
    actions = np.zeros((2, 2), dtype=bool)
    actions[1, 1] = True
    grasp = select_best_grasp(quality, space, pixels, actions)
    self.assertAlmostEqual(grasp['quality'], 0.7)
    self.assertEqual(space.encode(grasp['yaw'], grasp['width']), 3)

  def test_random_grasp_uses_valid_pixel(self):
    space = GraspActionSpace(n_rotations=2, n_widths=2)
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 2] = True
    grasp = select_random_grasp(np.random.default_rng(1), space, mask)
    self.assertEqual(grasp['pixel'], (1, 2))


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


class GraspQInputTest(absltest.TestCase):

  def test_preprocessing_does_not_mutate_raw_heightmap(self):
    image = np.zeros((4, 5, 6), dtype=np.float32)
    image[..., 3:] = 0.04
    original = image.copy()
    prepared = GraspQModel._prepare_image(image)
    np.testing.assert_array_equal(image, original)
    self.assertEqual(prepared.shape, (1, 4, 5, 6))
    self.assertNotEqual(float(prepared[0, 0, 0, 3]), 0.04)


class PickPlaceOutcomeTest(absltest.TestCase):

  @staticmethod
  def _run(end_effector):
    primitive = PickPlaceParallelJaw(min_lift_height=0.05)
    pose = ((0.5, 0.0, 0.04), (0, 0, 0, 1))
    primitive(None, lambda unused_pose, unused_speed=0.01: False,
              end_effector, pose, pose)
    return primitive

  def test_lift_success_requires_attachment_and_height(self):
    outcome = self._run(_FakeEndEffector(attaches=True, lift_height=0.08))
    self.assertTrue(outcome.last_grasp)
    self.assertTrue(outcome.last_lift_success)
    self.assertTrue(outcome.last_transport_retained)
    self.assertEqual(outcome.last_grasp_object_id, 9)
    self.assertAlmostEqual(outcome.last_lift_height, 0.08)

  def test_short_lift_is_not_success(self):
    outcome = self._run(_FakeEndEffector(attaches=True, lift_height=0.01))
    self.assertTrue(outcome.last_grasp)
    self.assertFalse(outcome.last_lift_success)

  def test_failed_attachment_is_not_success(self):
    outcome = self._run(_FakeEndEffector(attaches=False))
    self.assertFalse(outcome.last_grasp)
    self.assertFalse(outcome.last_lift_success)
    self.assertIsNone(outcome.last_grasp_object_id)

  def test_lift_only_still_records_success(self):
    primitive = PickPlaceParallelJaw(min_lift_height=0.05)
    end_effector = _FakeEndEffector(attaches=True, lift_height=0.08)
    pose = ((0.5, 0.0, 0.04), (0, 0, 0, 1))
    primitive(None, lambda unused_pose, unused_speed=0.01: False,
              end_effector, pose, pose, lift_only=True)
    self.assertTrue(primitive.last_lift_success)
    self.assertTrue(primitive.last_transport_retained)
    self.assertFalse(end_effector.secured)


if __name__ == '__main__':
  absltest.main()
