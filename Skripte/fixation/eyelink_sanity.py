"""EyeLink adapter: sample on calibrated timestamps, then call the core detector."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import SCREEN_WIDTH_PX, SCREEN_WIDTH_CM, VIEWING_DISTANCE_CM
from fixation.fixation_detector import FixationConfig, detect_fixations
from blink.run_data import stable_events


def pixels_to_degrees(x):
    return np.degrees(np.arctan2(
        (np.asarray(x, dtype=float) - SCREEN_WIDTH_PX / 2) * SCREEN_WIDTH_CM / SCREEN_WIDTH_PX,
        VIEWING_DISTANCE_CM))


def sample_eyelink(eye, gaze):
    """Nearest sample within 2 ms; preserve invalid EyeLink samples and target rows."""
    eye = eye.copy()
    for col in ['timestamp_ms', 'x_pos', 'y_pos', 'pupil_size']:
        eye[col] = pd.to_numeric(eye[col], errors='coerce')
    valid = ((eye['pupil_size'] > 0) & (eye['x_pos'] > 0) & (eye['y_pos'] > 0)
             & np.isfinite(eye['x_pos']) & np.isfinite(eye['y_pos']))
    eye['x_valid'] = eye['x_pos'].where(valid)
    # At 2000 Hz, timestamps rounded to milliseconds can repeat.
    eye = eye.dropna(subset=['timestamp_ms']).groupby('timestamp_ms')['x_valid'].agg(
        lambda x: x.mean() if x.notna().all() else np.nan).sort_index()
    if eye.empty:
        raise ValueError('No EyeLink timestamps available')
    t = gaze['timestamp_ms_synced'].to_numpy(float)
    et = eye.index.to_numpy(float)
    pos = np.searchsorted(et, t)
    left = np.clip(pos - 1, 0, len(et) - 1)
    right = np.clip(pos, 0, len(et) - 1)
    nearest = np.where(np.abs(et[left] - t) <= np.abs(et[right] - t), left, right)
    x = eye.to_numpy()[nearest].copy()
    x[np.abs(et[nearest] - t) > 2.0] = np.nan
    out = gaze[['frame', 'timestamp_ms_synced', 'trial_number', 'phase_type',
                'valid_eye_frame']].copy()
    out['gaze_deg_x_calib'] = pixels_to_degrees(x)
    out['valid_eye_frame'] &= np.isfinite(out['gaze_deg_x_calib'])
    return out


def run_sanity(run_dir, gaze, source_method):
    run = Path(run_dir)
    sampled = sample_eyelink(pd.read_csv(run / 'debug_3_eyetracker_data.csv'), gaze)
    result = detect_fixations(sampled, config=FixationConfig())
    extra = run / 'extra'
    extra.mkdir(exist_ok=True)
    stable_events(result.fixations).to_csv(extra / 'eyelink_fixations_sanity.csv', index=False)
    # Save the actual sampled input so the sanity check can be reproduced.
    sampled.to_csv(extra / 'eyelink_sanity_samples.csv', index=False)
    (extra / 'eyelink_sanity_metadata.json').write_text(json.dumps({
        'detector': 'fixation.fixation_detector.detect_fixations',
        'timestamp_source': 'timestamp_ms_synced', 'sampling_tolerance_ms': 2.0,
        'sample_grid_method': source_method,
        'adaptive_threshold_deg': result.threshold_deg,
        'n_noise_windows': len(result.noise_distribution),
        'config': asdict(result.config)}, indent=2))
    return result
