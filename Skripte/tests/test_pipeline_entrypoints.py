"""Exercise real calibration CLIs and shared visualization data contracts."""
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils.analysis_pipeline import AnalysisPipeline, RunConfiguration
from utils.vp_data_manager import VPData
from blink.generate_blink_diagnostics import _read_gaze
from fixation.eyelink_sanity import pixels_to_degrees
from fixation.generate_fixation_timeline_pdf import load_blink_intervals, load_eyelink_gaze, load_trials
from shared.pipeline_preflight import check_runtime
from blink.run_data import validate_gaze_clock

SCRIPTS = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def fixture(self, run):
        x = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        calibration = dict(model_x=LinearRegression().fit(x, 960+10*x[:, 0]),
            model_y=LinearRegression().fit(x, 540+10*x[:, 1]), polynomial_degree=1,
            input_type='gaze_angles', r2_scores=[1., 1.])
        pkl = run / 'selected.pkl'
        pkl.write_bytes(pickle.dumps(calibration))
        gaze = pd.DataFrame(dict(frame=range(4), timestamp_ms=[0.,20.,40.,60.],
            timestamp_ms_synced=[1000.,1020.,1040.,1060.], timestamp_source='frame_pts_v1',
            gaze_pitch_deg=x[:, 0], gaze_yaw_deg=x[:, 1], confidence=1.,
            avg_pupil_x_px_raw=x[:, 0], avg_pupil_y_px_raw=x[:, 1],
            trial_number=1, phase_type='fixation', detected=True))
        for name in ['debug_4_ptgaze_gaze_synced.csv', 'debug_4_pupil_data_synced.csv']:
            gaze.to_csv(run / name, index=False)
        return pkl, gaze

    def cli(self, run, method, pkl):
        name = 'debug_5_ptgaze.py' if method == 'ptgaze' else 'debug_5_partial_calibration.py'
        env = dict(os.environ, PIPELINE_OUTPUT_BASE_DIR=str(run),
                   PIPELINE_ANALYSIS_MODE='2', PIPELINE_CALIBRATION_PKL_PATH=str(pkl))
        return subprocess.run([sys.executable, str(SCRIPTS / 'debug' / name)],
                              env=env, capture_output=True, text=True, timeout=30)

    def test_real_calibration_clis_preserve_frame_clock_and_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            pkl, gaze = self.fixture(run)
            # An unrelated, newer cache must not override the selected PKL.
            (run / 'calibration_ptgaze_wrong.pkl').write_bytes(b'invalid pickle')
            for method in ['mediapipe', 'ptgaze']:
                result = self.cli(run, method, pkl)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                ok, message = AnalysisPipeline._check_calibration_output(run, method)
                self.assertTrue(ok, message)
                stem = 'ptgaze' if method == 'ptgaze' else 'pupil_data'
                out = pd.read_csv(run / f'debug_5_{stem}_calibrated.csv')
                self.assertEqual(out.frame.tolist(), gaze.frame.tolist())
                np.testing.assert_allclose(out.timestamp_ms_synced, gaze.timestamp_ms_synced)
                np.testing.assert_allclose(out.gaze_deg_x_calib,
                    pixels_to_degrees(960+10*gaze.gaze_pitch_deg), atol=1e-10)

    def test_real_calibration_failure_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.fixture(run)
            for method in ['mediapipe', 'ptgaze']:
                result = self.cli(run, method, run / 'missing.pkl')
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn('calibration_', result.stdout)

    def test_ptgaze_cannot_fall_back_to_unsynchronized_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            pkl, gaze = self.fixture(run)
            (run / 'debug_4_ptgaze_gaze_synced.csv').unlink()
            gaze.to_csv(run / 'debug_1_ptgaze_data.csv', index=False)
            result = self.cli(run, 'ptgaze', pkl)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Synchronized PTGaze input required', result.stderr)

    def test_missing_output_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, _ = AnalysisPipeline._check_calibration_output(Path(tmp), 'ptgaze')
            self.assertFalse(ok)

    def test_clock_validation_rejects_unknown_frames_and_nonfinite_times(self):
        reference = pd.Series([0.,20.])
        for frame, timestamp in [(2,1040.), (0,np.nan), (0,np.inf)]:
            with self.subTest(frame=frame, timestamp=timestamp):
                with self.assertRaisesRegex(ValueError, 'coverage|non-finite'):
                    validate_gaze_clock(pd.DataFrame({'frame':[frame],
                        'timestamp_ms_synced':[timestamp]}), reference, 1000.)

    def test_pipeline_stops_before_run_creation_when_runtime_preflight_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video, asc = root / 'video.mp4', root / 'eye.asc'
            video.touch(); asc.touch()
            pipeline = AnalysisPipeline(VPData('test', root, asc_file=asc, videos={'60hz':video}), SCRIPTS)
            config = RunConfiguration('test', '60hz', 'FullCalib', True)
            with patch('utils.analysis_pipeline.subprocess.run', return_value=subprocess.CompletedProcess([], 1)) as call:
                ok, message = pipeline.validate_prerequisites(config)
            self.assertFalse(ok)
            self.assertIn('preflight failed', message)
            self.assertIn('pipeline_preflight.py', call.call_args.args[0][1])
            self.assertFalse((root / 'Analyse').exists())

    def test_all_invalid_samples_fail_explicitly_in_real_calibration_clis(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            pkl, gaze = self.fixture(run)
            for column in ['gaze_pitch_deg', 'gaze_yaw_deg', 'avg_pupil_x_px_raw', 'avg_pupil_y_px_raw']:
                gaze[column] = np.nan
            for name in ['debug_4_ptgaze_gaze_synced.csv', 'debug_4_pupil_data_synced.csv']:
                gaze.to_csv(run / name, index=False)
            for method in ['mediapipe', 'ptgaze']:
                result = self.cli(run, method, pkl)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('no_valid_', result.stdout)

    def test_both_plots_use_same_geometry_and_invalid_eye_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'eye.csv'
            pd.DataFrame(dict(timestamp_ms=[1000,1020,1040,1060],
                x_pos=[960,1000,0,1100], y_pos=540, pupil_size=[10,10,10,0])).to_csv(path,index=False)
            blink = _read_gaze(path, 1000, eyelink=True)
            fixation = load_eyelink_gaze(path)
            np.testing.assert_allclose(blink.x, pixels_to_degrees(fixation.x_pos), equal_nan=True)
            self.assertEqual(blink.x.iloc[0], 0.)
            self.assertTrue(blink.x.iloc[2:].isna().all())

    def test_single_invalid_frame_keeps_full_support_in_fixation_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); (run / 'extra').mkdir()
            (run / 'debug_4_sync_info.json').write_text(json.dumps({'offset_used_ms':1000}))
            phases = {'phases': {'experiment_block1': {'trials': [dict(trial_number=1,
                fixation=dict(start_video_s=0,end_video_s=.04,start_eyelink_ms=1000,end_eyelink_ms=1040),
                stimulus=dict(start_video_s=.04,end_video_s=.1,start_eyelink_ms=1040,end_eyelink_ms=1100))]}}}
            (run / 'phases_detected.json').write_text(json.dumps(phases))
            path = run / 'extra' / 'blink_filter_frame_log.csv'
            pd.DataFrame(dict(frame_number=[0,1,2],video_time_ms=[0,20,70],
                frame_end_time_ms=[20,70,100],valid_eye_frame=[True,False,True])).to_csv(path,index=False)
            intervals = load_blink_intervals(path, load_trials(run / 'phases_detected.json'), pd.DataFrame())
            self.assertEqual(intervals[1], [(20.,70.)])

    def test_preflight_stops_on_missing_decoder_before_model_imports(self):
        with patch('shared.pipeline_preflight.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'ffmpeg'):
                check_runtime('/missing/video.mp4')


if __name__ == '__main__':
    unittest.main()
