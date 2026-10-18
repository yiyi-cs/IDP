"""Strict frame alignment shared by detection and diagnostic consumers."""
from pathlib import Path

import numpy as np
import pandas as pd


def boolean_mask(values):
    normalized = values.astype(str).str.lower().str.strip()
    mapping = {'true': True, 'false': False, '1': True, '0': False,
               '1.0': True, '0.0': False}
    result = normalized.map(mapping)
    if result.isna().any():
        raise ValueError('Blink mask contains null or unrecognized boolean values')
    return result.astype(bool)


def join_blink_mask(gaze, frames):
    """Every calibrated frame must have exactly one explicit validity value."""
    for df, key in ((gaze, 'frame'), (frames, 'frame_number')):
        if key not in df:
            raise ValueError(f'Missing frame key: {key}')
        if df[key].isna().any() or df[key].duplicated().any():
            raise ValueError(f'Null or duplicate frame key: {key}')
        numeric = pd.to_numeric(df[key], errors='coerce')
        if not np.isfinite(numeric).all() or (numeric % 1 != 0).any():
            raise ValueError(f'Non-integer frame key: {key}')
    if gaze.empty:
        raise ValueError('Calibrated gaze is empty')
    mask = frames[['frame_number', 'valid_eye_frame']].copy()
    mask['valid_eye_frame'] = boolean_mask(mask['valid_eye_frame'])
    # Current Run's final mask replaces any earlier propagated validity field.
    out = gaze.drop(columns=['valid_eye_frame', 'frame_number'], errors='ignore').merge(
        mask, left_on='frame', right_on='frame_number', how='left',
        validate='one_to_one', sort=False)
    if out['valid_eye_frame'].isna().any():
        missing = out.loc[out['valid_eye_frame'].isna(), 'frame'].tolist()
        raise ValueError(f'Blink mask does not cover calibrated frames: {missing[:10]}')
    out['valid_eye_frame'] = out['valid_eye_frame'].astype(bool)
    if 'trial_number' not in out and 'trial_assignment' in out:
        out['trial_number'] = out['trial_assignment']
    if 'timestamp_ms_synced' not in out:
        raise ValueError('Synchronized gaze timestamps are required; no PTS fallback')
    if not np.isfinite(pd.to_numeric(out['timestamp_ms_synced'], errors='coerce')).all():
        raise ValueError('Synchronized gaze timestamps contain null/non-finite values')
    return out


def load_validated_gaze(run_dir, filename):
    run = Path(run_dir)
    return join_blink_mask(pd.read_csv(run / filename),
                          pd.read_csv(run / 'extra' / 'blink_filter_frame_log.csv'))


def stable_events(events):
    """Retain a readable CSV schema even when no events were detected."""
    if not events.empty:
        return events
    return events.reindex(columns=['fixation_id', 'trial_number', 'start_time',
        'end_time', 'duration', 'x_pos', 'dispersion_x_deg', 'n_samples',
        'start_frame', 'end_frame', 'threshold_used_deg'])
