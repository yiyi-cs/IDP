"""Regression checks for variable-frame-rate selection and synchronized time."""
import json
import tempfile
import unittest
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from shared.frame_timing import (FrameClock, assign_trials, correct_saved_gaze,
                                 detector_input, TIMEBASE_VERSION, load_frame_clock,
                                 exact_frame_window)
from fixation.fixation_detector import detect_fixations
from blink.run_data import load_validated_gaze
from utils.analysis_pipeline import CrossRunCacheScanner


def phases():
    return {'phases': {'experiment_block1': {'trials': [dict(trial_number=1,
        is_practice=False, fixation=dict(start_eyelink_ms=1000, end_eyelink_ms=1060),
        stimulus=dict(start_eyelink_ms=1090, end_eyelink_ms=1500))]}}}


def setup_run(run):
    (run / 'extra').mkdir()
    (run / 'phases_detected.json').write_text(json.dumps(phases()))
    (run / 'debug_4_sync_info.json').write_text(json.dumps({'offset_used_ms': 1000}))


class FrameTimingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg/ffprobe unavailable')
    def test_h264_bframes_vfr_and_nonzero_pts_match_sequential_decode(self):
        encoders = subprocess.run(['ffmpeg', '-v', 'error', '-encoders'], capture_output=True, text=True, check=True).stdout
        if 'libx264' not in encoders:
            self.skipTest('libx264 unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            for offset in [0, 3]:
                video = run / f'h264_{offset}.mp4'
                subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc=size=64x64:rate=25:duration=1', '-vf', f'setpts=N*N/(100*TB)+{offset}/TB',
                    '-fps_mode', 'vfr', '-c:v', 'libx264', '-bf', '3', '-g', '10',
                    '-pix_fmt', 'yuv420p', str(video)], check=True)
                clock = load_frame_clock(video, run)
                n = len(clock.times_ms)
                self.assertEqual(clock.times_ms[0], offset*1000)
                self.assertGreater(len(np.unique(np.diff(clock.times_ms))), 1)
                full = list(exact_frame_window(video, clock, 0, n, 64, 64))
                for start, end in [(1,2), (3,9), (12,20), (n-1,n)]:
                    selected = list(exact_frame_window(video, clock, start, end, 64, 64))
                    self.assertEqual([i for i,_ in selected], list(range(start,end)))
                    for (_, actual), (_, expected) in zip(selected, full[start:end]):
                        np.testing.assert_array_equal(actual, expected)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg/ffprobe unavailable')
    def test_real_vfr_seek_pixels_match_full_sequential_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            video = run / 'test.mkv'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                'testsrc=size=32x32:rate=25:duration=0.4', '-vf', 'setpts=N*N/(25*TB)',
                '-fps_mode', 'vfr', '-c:v', 'ffv1', str(video)], check=True)
            clock = load_frame_clock(video, run)
            np.testing.assert_array_equal(clock.times_ms,
                [0.,40.,160.,360.,640.,1000.,1440.,1960.,2560.,3240.])
            full = list(exact_frame_window(video, clock, 0, 10, 32, 32))
            for start, end in [(1,2), (2,7), (7,10)]:
                selected = list(exact_frame_window(video, clock, start, end, 32, 32))
                self.assertEqual([i for i,_ in selected], list(range(start,end)))
                for (_, actual), (_, expected) in zip(selected, full[start:end]):
                    np.testing.assert_array_equal(actual, expected)

    def test_vfr_bounds_use_real_timestamps(self):
        clock = FrameClock([0, 10, 50, 75, 110, 140])
        self.assertEqual(clock.bounds(.04, .12), (2, 5))
        self.assertEqual(clock.at(3), 75)
        self.assertEqual(clock.bounds(-1, .01), (0, 1))

    def test_bad_frame_clock_is_rejected(self):
        for times in [[], [0, 0], [1, 0], [0, np.nan]]:
            with self.subTest(times=times), self.assertRaises(ValueError):
                FrameClock(times)

    def test_temporal_repair_preserves_model_values_and_original_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            setup_run(run)
            frames = pd.DataFrame(dict(frame_number=range(6), video_time_ms=[0,10,50,75,110,140],
                                        frame_end_time_ms=[10,50,75,110,140,160], valid_eye_frame=True))
            gaze = pd.DataFrame(dict(frame=[0,2,5], timestamp_ms=[0,40,100],
                timestamp_ms_synced=[1000,1040,1100], trial_number=1, phase_type='fixation',
                valid_eye_frame=True, gaze_deg_x_calib=[1.,2.,3.]))
            with self.assertRaisesRegex(ValueError, 'unverified frame identity'):
                correct_saved_gaze(gaze, frames, run)
            fixed = correct_saved_gaze(gaze, frames, run, frame_identity_verified=True)
            self.assertEqual(fixed.timestamp_ms_synced.tolist(), [1000,1050,1140])
            self.assertEqual(fixed.timestamp_ms_synced_original.tolist(), [1000,1040,1100])
            self.assertEqual(fixed.gaze_deg_x_calib.tolist(), gaze.gaze_deg_x_calib.tolist())
            self.assertEqual(fixed.phase_type.tolist(), ['fixation','fixation','stimulus'])
            again = correct_saved_gaze(fixed, frames, run, frame_identity_verified=True)
            self.assertEqual(again.timestamp_ms_synced_original.tolist(), [1000,1040,1100])
            frames.to_csv(run / 'extra' / 'blink_filter_frame_log.csv', index=False)
            gaze.to_csv(run / 'gaze.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'mismatched timebase'):
                load_validated_gaze(run, 'gaze.csv')
            fixed.to_csv(run / 'gaze.csv', index=False)
            self.assertEqual(len(load_validated_gaze(run, 'gaze.csv')), 3)

    def test_phase_transition_has_trial_identity_but_is_not_noise_phase(self):
        df = pd.DataFrame({'timestamp_ms_synced':[990,1000,1050,1075,1090,1500]})
        labelled = assign_trials(df, phases())
        self.assertEqual(labelled.trial_number.tolist(), [-1,1,1,1,1,-1])
        self.assertEqual(labelled.phase_type.tolist(),
                         ['unassigned','fixation','fixation','transition','stimulus','unassigned'])

    def test_missing_model_samples_remain_invalid_and_split_core_detector(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            setup_run(run)
            t = np.arange(0, 420, 20)
            frames = pd.DataFrame(dict(frame_number=range(len(t)), video_time_ms=t,
                frame_end_time_ms=t+20, valid_eye_frame=True))
            gaze = pd.DataFrame(dict(frame=np.arange(len(t)), timestamp_ms_synced=t+1000,
                trial_number=1, phase_type='fixation', valid_eye_frame=True, gaze_deg_x_calib=0.))
            gaze = gaze[(gaze.frame < 5) | (gaze.frame > 12)].copy()
            prepared = detector_input(gaze, frames, run)
            self.assertEqual((~prepared.has_calibrated_gaze).sum(), 8)
            self.assertFalse(prepared.loc[prepared.frame.eq(8), 'valid_eye_frame'].iloc[0])
            # Use fixture's whole trial as a known fixation phase, preserving core settings.
            prepared['phase_type'] = 'fixation'
            result = detect_fixations(prepared)
            self.assertGreater(len(result.fixations), 0)
            for fx in result.fixations.itertuples():
                self.assertFalse(fx.start_time <= 1080 and fx.end_time >= 1260)

    def test_raw_but_legacy_fps_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            vp = Path(tmp)
            source, target = vp / 'source', vp / 'target'
            source.mkdir(); target.mkdir()
            (source / 'config_used.json').write_text(json.dumps({'preprocess_mode':'none'}))
            (target / 'config_used.json').write_text(json.dumps({'preprocess_mode':'none',
                                                               'timebase_version':TIMEBASE_VERSION}))
            (source / 'debug_5_ptgaze_calibrated.csv').write_text('frame\n1\n')
            self.assertFalse(CrossRunCacheScanner(vp).copy_step_outputs(source, target, 'debug_5_ptgaze'))


if __name__ == '__main__':
    unittest.main()
