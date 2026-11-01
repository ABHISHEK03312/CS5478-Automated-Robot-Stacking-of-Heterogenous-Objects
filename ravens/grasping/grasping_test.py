"""Tests for RGB-D grasp action encoding and dataset storage."""

import tempfile
import sys
from unittest import mock

from absl.testing import absltest
import numpy as np
from ravens.grasping.dataset import GraspDataset
from ravens.grasping.dataset import GraspSample
from ravens.grasping.crops import candidate_crop
from ravens.grasping.encoding import GraspActionSpace
from ravens.grasping.metrics import binary_grasp_metrics
from ravens.grasping.selection import component_center_mask
from ravens.grasping.selection import component_grasp_candidates
from ravens.grasping.selection import rank_grasp_candidates
from ravens.grasping.selection import select_best_grasp
from ravens.grasping.selection import select_random_grasp
from ravens.grasping.planner import CandidateGraspPlanner
from ravens.grasping.analytic import RobustAnalyticGraspModel
from ravens.grasping.policies import resolve_policy
from ravens.grasping.pointcloud import heightmap_pointcloud
from ravens.grasping.pointcloud import NpzCommandGraspBackend
from ravens.grasping.pointcloud import planar_grasp_candidates
from ravens.grasping.pointcloud_planner import PointCloudGraspPlanner
from ravens.grasping.replay import GraspReplayDataset
from ravens.grasping.replay import GraspTransition
from ravens.grasping.robustness import ParallelJawRobustnessModel
from ravens.grasping.stack_replay import StackActionReplayDataset
from ravens.grasping.stack_replay import StackActionTransition
from ravens.grasping.stack_action_planner import StackActionQScorer
from ravens.grasping.stack_pair_planner import StackActionPairScorer
from ravens.models.grasp_q import GraspQModel
from ravens.models.stack_action_q import StackActionQModel
from ravens.models.stack_pair_rank import StackActionPairRankModel
from ravens.utils import utils
from ravens.tasks.primitives import PickPlaceParallelJaw


class GraspRobustnessTest(absltest.TestCase):

  def setUp(self):
    self.meta = {
        'id': 4, 'shape': 'rectangle', 'dims': [0.12, 0.04, 0.03]
    }

  def test_disturbance_is_deterministic_for_seed_and_object(self):
    model = ParallelJawRobustnessModel(
        7, position_noise_mm=4, yaw_noise_degrees=6, slip_scale=0.5)
    first = model.evaluate(self.meta, (0.0, 0.0, 0.12), 0.0)
    second = model.evaluate(self.meta, (0.0, 0.0, 0.12), 0.0)
    self.assertEqual(first, second)

  def test_large_center_error_is_rejected(self):
    model = ParallelJawRobustnessModel(7, slip_scale=0.0)
    result = model.evaluate(self.meta, (0.04, 0.0, 0.12), 0.0)
    self.assertFalse(result.success)
    self.assertEqual(result.reason, 'missed_center')

  def test_short_axis_grasp_has_more_clearance(self):
    model = ParallelJawRobustnessModel(7, slip_scale=0.0)
    long_axis = model.evaluate(self.meta, (0.0, 0.0, 0.12), 0.0)
    short_axis = model.evaluate(self.meta, (0.0, 0.0, 0.12), np.pi / 2)
    self.assertLess(short_axis.jaw_width_m, long_axis.jaw_width_m)
    self.assertGreater(short_axis.quality, long_axis.quality)


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


class CandidateCropTest(absltest.TestCase):

  def test_yaw_aligns_source_axis_horizontally(self):
    image = np.zeros((15, 15, 1), dtype=np.float32)
    image[3:12, 7, 0] = 1
    crop = candidate_crop(image, (7, 7), np.pi / 2, size=9)
    self.assertGreater(np.sum(crop[4, :, 0]), np.sum(crop[:, 4, 0]))

  def test_crop_does_not_mutate_image(self):
    image = np.random.default_rng(1).random((8, 9, 2), dtype=np.float32)
    original = image.copy()
    self.assertEqual(candidate_crop(image, (4, 4), 0.3, 6).shape, (6, 6, 2))
    np.testing.assert_array_equal(image, original)


class CandidateGraspPlannerTest(absltest.TestCase):

  @staticmethod
  def _candidate(probability, uncertainty, quality):
    return {'probability': probability, 'uncertainty': uncertainty,
            'quality': quality}

  def test_selector_changes_only_for_confident_margin(self):
    planner = CandidateGraspPlanner(None)
    planner._by_object = {  # pylint: disable=protected-access
        1: self._candidate(0.9, 0.02, 0.85),
        2: self._candidate(0.7, 0.03, 0.60),
    }
    self.assertEqual(planner.select_object([1, 2], fallback=2), 1)
    self.assertEqual(planner.selection_changes, 1)

  def test_selector_abstains_when_margin_is_small(self):
    planner = CandidateGraspPlanner(None)
    planner._by_object = {  # pylint: disable=protected-access
        1: self._candidate(0.9, 0.02, 0.80),
        2: self._candidate(0.9, 0.02, 0.72),
    }
    self.assertEqual(planner.select_object([1, 2], fallback=2), 2)
    self.assertEqual(planner.selection_abstentions, 1)


class RobustAnalyticGraspModelTest(absltest.TestCase):

  def test_scores_candidates_with_common_policy_fields(self):
    image = np.zeros((32, 32, 6), dtype=np.float32)
    image[10:20, 11:21, 3:] = 0.04
    candidate = {'pixel': (14, 15), 'yaw': 0.0, 'width': 0.04,
                 'quality': None}
    ranked = RobustAnalyticGraspModel().score_candidates(
        image, [candidate], mc_samples=3)
    self.assertLen(ranked, 1)
    self.assertBetween(ranked[0]['probability'], 0.0, 1.0)
    self.assertGreaterEqual(ranked[0]['uncertainty'], 0.0)
    self.assertLessEqual(ranked[0]['quality'], ranked[0]['probability'])

  def test_centered_candidate_is_more_robust_than_edge_candidate(self):
    image = np.zeros((32, 32, 6), dtype=np.float32)
    image[10:20, 10:20, 3:] = 0.04
    candidates = [
        {'pixel': (14, 14), 'yaw': 0.0, 'width': 0.04, 'quality': None},
        {'pixel': (10, 10), 'yaw': 0.0, 'width': 0.04, 'quality': None},
    ]
    ranked = RobustAnalyticGraspModel().score_candidates(
        image, candidates, mc_samples=5)
    self.assertEqual(ranked[0]['pixel'], (14, 14))

  def test_checkpoint_keeps_legacy_candidate_q_selection(self):
    self.assertEqual(resolve_policy('geometry', 'model.h5'), 'candidate_q')

  def test_external_policy_requires_backend_command(self):
    from ravens.grasping.policies import create_grasp_planner
    with self.assertRaisesRegex(ValueError, 'grasp_backend_arg'):
      create_grasp_planner('gpd')


class PointCloudGraspPolicyTest(absltest.TestCase):

  def test_heightmap_conversion_uses_world_coordinates(self):
    image = np.zeros((4, 5, 6), dtype=np.float32)
    image[2, 3, :3] = (10, 20, 30)
    image[2, 3, 3:] = 0.04
    cloud = heightmap_pointcloud(image)
    self.assertEqual(cloud['points'].shape, (1, 3))
    np.testing.assert_allclose(cloud['points'][0],
                               (0.259375, -0.49375, 0.04))
    np.testing.assert_array_equal(cloud['colors'][0], (10, 20, 30))
    np.testing.assert_array_equal(cloud['pixels'][0], (2, 3))

  def test_six_dof_filter_keeps_only_reachable_top_down_grasps(self):
    proposals = [
        {'position': (0.5, 0.0, 0.04), 'approach': (0, 0, -1),
         'closing_axis': (0, 1, 0), 'width': 0.04, 'score': 0.9},
        {'position': (0.5, 0.0, 0.04), 'approach': (1, 0, 0),
         'closing_axis': (0, 1, 0), 'width': 0.04, 'score': 1.0},
        {'position': (0.9, 0.0, 0.04), 'approach': (0, 0, -1),
         'closing_axis': (0, 1, 0), 'width': 0.04, 'score': 1.0},
    ]
    candidates = planar_grasp_candidates(proposals)
    self.assertLen(candidates, 1)
    self.assertAlmostEqual(candidates[0]['yaw'], np.pi / 2)
    self.assertEqual(candidates[0]['pixel'], (160, 80))

  def test_pointcloud_planner_matches_proposal_to_nearest_object(self):
    class Backend:
      def predict(self, unused_points, unused_colors):
        return [{
            'position': (0.4, 0.1, 0.04), 'approach': (0, 0, -1),
            'closing_axis': (1, 0, 0), 'width': 0.04, 'score': 0.9,
        }]

    image = np.zeros((320, 160, 6), dtype=np.float32)
    image[192, 48, 3:] = 0.04
    planner = PointCloudGraspPlanner(Backend(), refine_pose=True)
    poses = {3: ((0.4, 0.1, 0.02), (0, 0, 0, 1)),
             4: ((0.6, -0.1, 0.02), (0, 0, 0, 1))}
    with mock.patch(
        'ravens.grasping.pointcloud_planner.fused_heightmap',
        return_value=image), mock.patch(
            'ravens.grasping.pointcloud_planner.p.getBasePositionAndOrientation',
            side_effect=lambda oid: poses[oid]):
      selected = planner.prepare({}, [3, 4])
    self.assertIn(3, selected)
    self.assertNotIn(4, selected)
    self.assertAlmostEqual(planner.quality(3), 0.9)

  def test_npz_subprocess_protocol(self):
    code = (
        "import sys,numpy as np;"
        "src=sys.argv[sys.argv.index('--input')+1];"
        "dst=sys.argv[sys.argv.index('--output')+1];"
        "data=np.load(src);"
        "assert data['points'].shape==(1,3);"
        "assert str(data['frame'])=='world';"
        "np.savez(dst,positions=[[.5,0,.04]],approaches=[[0,0,-1]],"
        "closing_axes=[[1,0,0]],widths=[.04],scores=[.8])")
    backend = NpzCommandGraspBackend([sys.executable, '-c', code])
    proposals = backend.predict(
        np.asarray([[0.5, 0, 0.04]], dtype=np.float32),
        np.asarray([[255, 0, 0]], dtype=np.uint8))
    self.assertLen(proposals, 1)
    self.assertAlmostEqual(float(proposals[0]['score']), 0.8, places=6)


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


class GraspReplayDatasetTest(absltest.TestCase):

  def test_transition_round_trip(self):
    image = np.zeros((8, 7, 6), dtype=np.float32)
    next_image = image.copy()
    next_image[3, 4, 3:] = 0.04
    transition = GraspTransition(
        image=image, pixel=np.asarray([3, 4]), yaw=0.2, width=0.04,
        reward=1 / 6, next_image=next_image, done=False,
        metadata={'episode': 2, 'step': 1})
    with tempfile.TemporaryDirectory() as directory:
      dataset = GraspReplayDataset(directory)
      dataset.add(transition)
      loaded = dataset.load(0)
    np.testing.assert_array_equal(loaded.image, image)
    np.testing.assert_array_equal(loaded.next_image, next_image)
    self.assertAlmostEqual(loaded.reward, 1 / 6, places=6)
    self.assertFalse(loaded.done)
    self.assertEqual(loaded.metadata, {'episode': 2, 'step': 1})


class StackActionReplayDatasetTest(absltest.TestCase):

  def test_joint_action_transition_round_trip(self):
    image = np.zeros((8, 7, 6), dtype=np.float32)
    transition = StackActionTransition(
        image=image, pick_pixel=np.asarray([3, 4]), pick_yaw=0.2,
        width=0.04, place_position=np.asarray([0.5, 0.1, 0.08]),
        place_yaw=0.6, reward=1 / 6, next_image=image.copy(), done=False,
        metadata={'episode': 1, 'step': 2, 'seed': 9})
    with tempfile.TemporaryDirectory() as directory:
      dataset = StackActionReplayDataset(directory)
      dataset.add(transition)
      loaded = dataset.load(0)
    np.testing.assert_array_equal(loaded.pick_pixel, (3, 4))
    np.testing.assert_allclose(loaded.place_position, (0.5, 0.1, 0.08))
    self.assertAlmostEqual(loaded.reward, 1 / 6, places=6)
    self.assertEqual(loaded.metadata['step'], 2)

  def test_joint_action_features_are_bounded(self):
    features = StackActionQModel.action_features(
        0.04, (0.5, 0.0, 0.08), 0.5)
    self.assertEqual(features.shape, (5,))
    self.assertTrue(np.all(features >= 0))
    self.assertTrue(np.all(features <= 1))

  def test_planner_action_scorer_builds_joint_action(self):
    class Model:
      def predict(self, transitions):
        self.transition = transitions[0]
        return np.asarray([0.75])

    model = Model()
    scorer = StackActionQScorer(model)
    image = np.zeros((320, 160, 6), dtype=np.float32)
    with mock.patch(
        'ravens.grasping.stack_action_planner.fused_heightmap',
        return_value=image):
      scorer.prepare({})
    pick = ((0.5, 0.0, 0.04), (0, 0, 0, 1))
    place = ((0.4, 0.1, 0.08), utils.eulerXYZ_to_quatXYZW((0, 0, 0.4)))
    self.assertAlmostEqual(scorer.quality(pick, place), 0.75)
    np.testing.assert_array_equal(model.transition.pick_pixel, (160, 80))
    np.testing.assert_allclose(model.transition.place_position,
                               (0.4, 0.1, 0.08))

  def test_joint_action_rerank_abstains_on_ensemble_disagreement(self):
    class Model:
      def __init__(self, values):
        self.values = iter(values)

      def predict(self, transitions):
        del transitions
        return np.asarray([next(self.values)])

    scorer = StackActionQScorer([
        Model([0.5, 0.9]), Model([0.5, 0.1])])
    scorer.image = np.zeros((320, 160, 6), dtype=np.float32)
    pick = ((0.5, 0.0, 0.04), (0, 0, 0, 1))
    places = [((0.5, 0.0, 0.08), (0, 0, 0, 1)),
              ((0.51, 0.0, 0.08), (0, 0, 0, 1))]
    selected = scorer.rerank(
        pick, places, [0.0, 0.01], weight=1.0,
        max_uncertainty=0.1, min_margin=0.0)
    self.assertEqual(selected, 0)
    self.assertEqual(scorer.rerank_abstentions, 1)

  def test_candidate_relative_features_reverse_sign(self):
    image = np.zeros((8, 7, 6), dtype=np.float32)
    first = StackActionTransition(
        image=image, pick_pixel=np.asarray([3, 4]), pick_yaw=0.0,
        width=0.04, place_position=np.asarray([0.5, 0.1, 0.08]),
        place_yaw=0.4, reward=0.0, next_image=image, done=False, metadata={})
    second = StackActionTransition(
        image=image, pick_pixel=np.asarray([3, 4]), pick_yaw=0.0,
        width=0.04, place_position=np.asarray([0.51, 0.08, 0.08]),
        place_yaw=0.2, reward=0.0, next_image=image, done=False, metadata={})
    forward = StackActionPairRankModel.pair_features(
        first, second, 0.3, 0.2)
    reverse = StackActionPairRankModel.pair_features(
        second, first, 0.2, 0.3)
    self.assertEqual(forward.shape, (16,))
    np.testing.assert_allclose(forward[10:], -reverse[10:])

  def test_candidate_relative_scorer_selects_confident_challenger(self):
    class Model:
      def predict_pairs(self, first, second, first_scores, second_scores):
        del first, second, first_scores, second_scores
        return np.asarray([0.4, 0.8])

    scorer = StackActionPairScorer(Model())
    scorer.image = np.zeros((320, 160, 6), dtype=np.float32)
    pick = ((0.5, 0.0, 0.04), (0, 0, 0, 1))
    places = [((0.5, 0.0, 0.08), (0, 0, 0, 1)),
              ((0.51, 0.0, 0.08), (0, 0, 0, 1)),
              ((0.49, 0.0, 0.08), (0, 0, 0, 1))]
    self.assertEqual(scorer.rerank(
        pick, places, [0.3, 0.29, 0.28], min_probability=0.65), 2)
    self.assertEqual(scorer.rerank_changes, 1)


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
