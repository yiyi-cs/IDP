"""Integration boundaries; no model inference or external participant data required."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from blink.run_data import join_blink_mask, stable_events
from blink.diagnostic_statistics import synchronized_frames, predicted_intervals
from fixation.fixation_detector import detect_fixations
from fixation.eyelink_sanity import sample_eyelink, run_sanity


def inputs():
    t = np.arange(0., 2000., 20.)
    gaze = pd.DataFrame(dict(frame=np.arange(len(t)), timestamp_ms_synced=t+50000,
        trial_number=1, phase_type=np.where(t < 1000, 'fixation', 'stimulus'),
        gaze_deg_x_calib=np.sin(t/80)*0.5, valid_eye_frame=True))
    log = pd.DataFrame(dict(frame_number=gaze.frame, valid_eye_frame=True,
                            video_time_ms=t+800000))
    log.loc[20:25, 'valid_eye_frame'] = False
    return gaze, log


class IntegrationTests(unittest.TestCase):
    def test_mask_replaces_old_validity_and_retains_order(self):
        gaze, log = inputs()
        merged = join_blink_mask(gaze, log.iloc[::-1])
        self.assertEqual(merged.frame.tolist(), gaze.frame.tolist())
        self.assertFalse(merged.loc[20, 'valid_eye_frame'])
        self.assertEqual(len(merged), len(gaze))

    def test_missing_duplicate_and_null_masks_fail(self):
        gaze, log = inputs()
        null = log.astype({'valid_eye_frame': 'object'})
        null.loc[0, 'valid_eye_frame'] = None
        for bad in [log.iloc[1:], pd.concat([log, log.iloc[:1]]), null]:
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                join_blink_mask(gaze, bad)
        with self.assertRaises(ValueError):
            join_blink_mask(pd.concat([gaze, gaze.iloc[:1]]), log)

    def test_string_false_is_invalid(self):
        gaze, log = inputs()
        log['valid_eye_frame'] = log.valid_eye_frame.astype(str)
        self.assertFalse(join_blink_mask(gaze, log).loc[20, 'valid_eye_frame'])

    def test_no_unsynchronized_fallback(self):
        gaze, log = inputs()
        with self.assertRaisesRegex(ValueError, 'Synchronized'):
            join_blink_mask(gaze.drop(columns='timestamp_ms_synced'), log)

    def test_pts_cannot_change_interval_comparison(self):
        gaze, log = inputs()
        regions = pd.DataFrame([dict(start_frame=20, end_frame=25, reason_group='blink')])
        a = synchronized_frames(join_blink_mask(gaze, log), log)
        log['video_time_ms'] += 999999
        b = synchronized_frames(join_blink_mask(gaze, log), log)
        self.assertEqual(predicted_intervals(a, 50000, 52000, regions),
                         predicted_intervals(b, 50000, 52000, regions))
        self.assertEqual(predicted_intervals(a, 50000, 52000, regions)[0][:2], (400., 520.))

    def test_eyelink_sampling_tolerance_invalid_and_duplicate_timestamps(self):
        gaze, log = inputs()
        eye = pd.DataFrame(dict(timestamp_ms=gaze.timestamp_ms_synced, x_pos=960.,
                               y_pos=540., pupil_size=100.))
        eye.loc[30, 'pupil_size'] = 0
        eye = eye.drop(index=35)
        eye = pd.concat([eye, eye.iloc[[30]]], ignore_index=True)
        sampled = sample_eyelink(eye, join_blink_mask(gaze, log))
        self.assertFalse(sampled.loc[20, 'valid_eye_frame'])
        self.assertTrue(np.isnan(sampled.loc[30, 'gaze_deg_x_calib']))
        self.assertTrue(np.isnan(sampled.loc[35, 'gaze_deg_x_calib']))
        self.assertEqual(sampled.timestamp_ms_synced.tolist(), gaze.timestamp_ms_synced.tolist())

    def test_same_detector_and_independent_threshold(self):
        gaze, log = inputs()
        validated = join_blink_mask(gaze, log)
        eye = pd.DataFrame(dict(timestamp_ms=gaze.timestamp_ms_synced,
            x_pos=960+np.sin(np.arange(len(gaze))/4)*0.1, y_pos=540., pupil_size=100.))
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            eye.to_csv(run / 'debug_3_eyetracker_data.csv', index=False)
            with patch('fixation.eyelink_sanity.detect_fixations', wraps=detect_fixations) as same:
                result = run_sanity(run, validated, 'ptgaze')
                self.assertEqual(same.call_count, 1)
            expected = detect_fixations(sample_eyelink(eye, validated))
            pd.testing.assert_frame_equal(result.fixations, expected.fixations)
            self.assertLess(result.threshold_deg, detect_fixations(validated).threshold_deg)
            self.assertTrue((run / 'extra' / 'eyelink_fixations_sanity.csv').exists())

    def test_empty_events_keep_readable_schema(self):
        events = stable_events(pd.DataFrame())
        self.assertIn('trial_number', events.columns)
        self.assertIn('start_time', events.columns)


if __name__ == '__main__':
    unittest.main()
