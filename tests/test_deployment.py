import unittest
import numpy as np
from bas_vla.deployment.robot import ActionCalibration, execute_chunk, model_observation
from bas_vla.deployment.trials import summarize_trials


class DeploymentTests(unittest.TestCase):
    def calibration(self):
        return ActionCalibration(space='joint_positions_rad', coordinate_frame='robot_base',
                                 scale=[2] * 7, offset=[1] * 7,
                                 lower=[-5] * 7, upper=[5] * 7)

    def test_explicit_units_and_bounds(self):
        np.testing.assert_equal(self.calibration().convert(np.zeros((2, 7))), np.ones((2, 7)))
        with self.assertRaises(ValueError):
            self.calibration().convert(np.ones((2, 7)) * 5)
        with self.assertRaises(ValueError):
            self.calibration().convert(np.zeros((2, 6)))
        with self.assertRaises(ValueError):
            self.calibration().convert([[float('nan')] * 7])

    def test_rgb_only_inputs(self):
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        observation = model_observation(image, image, np.zeros(7))
        self.assertEqual(set(observation), {'full_image', 'wrist_image', 'state'})
        with self.assertRaises(ValueError):
            model_observation(image[:, :, 0], image, np.zeros(7))

    def test_validate_entire_chunk_before_sending(self):
        class Robot:
            sent = []
            def send_action(self, action, **kwargs): self.sent.append(action)
        robot = Robot()
        with self.assertRaises(ValueError):
            execute_chunk(robot, [[0] * 7, [10] * 7], self.calibration(), control_hz=20)
        self.assertEqual(robot.sent, [])

    def test_chunk_timing_without_hardware(self):
        class Robot:
            sent = []
            def send_action(self, action, **kwargs): self.sent.append(action)
        robot = Robot()
        ticks = iter([0., 0., .01, .02, .05])
        sleeps = []
        result = execute_chunk(robot, [[0] * 7, [1] * 7], self.calibration(),
                               control_hz=20, clock=lambda: next(ticks), sleep=sleeps.append)
        self.assertEqual(len(robot.sent), 2)
        self.assertEqual(len(result), 2)
        self.assertGreater(sleeps[0], 0)

    def record(self, **updates):
        record = dict(trial_id='a', case_id='T1', method='core', condition='break',
                      reset_id='r1', instruction='move red block', annotator='reviewer',
                      new_success=True, old_target_executed=False, new_target_first_commit=True,
                      original_subset=True)
        record.update(updates)
        return record

    def test_trial_rates_and_unknown_not_failure(self):
        rows = [self.record(), self.record(trial_id='b', new_success=None,
                                           old_target_executed=None, new_target_first_commit=None)]
        result = summarize_trials(rows)[0]
        self.assertEqual(result['new_success'], {'successes': 1, 'n': 1, 'rate': 1.0})
        self.assertEqual(result['trials'], 2)
        self.assertEqual(result['separation_score'], 1.0)

    def test_duplicate_trial_and_string_boolean_rejected(self):
        with self.assertRaises(ValueError):
            summarize_trials([self.record(), self.record()])
        with self.assertRaises(ValueError):
            summarize_trials([self.record(new_success='false')])


if __name__ == '__main__':
    unittest.main()
