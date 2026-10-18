"""A single presentation-time clock for video extraction and synchronization.

Detectors consume timestamp_ms_synced. PTS is converted to that clock upstream;
frame/average_fps is never used as a timestamp or to select a time window.
"""
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TIMEBASE_VERSION = 'frame_pts_v1'


@dataclass
class FrameClock:
    times_ms: np.ndarray

    def __post_init__(self):
        self.times_ms = np.asarray(self.times_ms, dtype=float)
        if (not len(self.times_ms) or not np.isfinite(self.times_ms).all()
                or np.any(np.diff(self.times_ms) <= 0)):
            raise ValueError('Frame timestamps must be finite and strictly increasing')

    def bounds(self, start_s, end_s):
        if not np.isfinite([start_s, end_s]).all() or end_s <= start_s:
            raise ValueError('Invalid video time window')
        return (int(np.searchsorted(self.times_ms, max(0., start_s)*1000, side='left')),
                int(np.searchsorted(self.times_ms, end_s*1000, side='left')))

    def at(self, frame):
        if frame < 0 or frame >= len(self.times_ms):
            raise ValueError(f'Decoded frame {frame} has no presentation timestamp')
        return float(self.times_ms[frame])


_CLOCKS = {}


def exact_frame_window(video, clock, start_frame, end_frame, width, height):
    """Accurate PTS seek and sequential decode; never CAP_PROP_POS_FRAMES on VFR.

    Output FPS conversion is disabled. The yielded ordinal belongs to the same
    ffprobe presentation-time table used to select this window.
    """
    if start_frame >= end_frame:
        return
    count = end_frame-start_frame
    command = ['ffmpeg', '-v', 'error', '-seek_timestamp', '1',
        '-ss', f'{clock.at(start_frame)/1000:.9f}', '-i', str(video),
        '-an', '-sn', '-dn', '-frames:v', str(count), '-fps_mode', 'passthrough',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = int(width)*int(height)*3
    try:
        for frame_number in range(start_frame, end_frame):
            data = bytearray()
            while len(data) < size:
                chunk = process.stdout.read(size-len(data))
                if not chunk:
                    error = process.stderr.read().decode(errors='replace')
                    raise RuntimeError(f'Exact video decode stopped at frame {frame_number}: {error}')
                data.extend(chunk)
            yield frame_number, np.frombuffer(data, dtype=np.uint8).reshape(int(height), int(width), 3)
        process.stdout.close()
        error = process.stderr.read().decode(errors='replace')
        if process.wait() != 0:
            raise RuntimeError(f'Exact video decode failed: {error}')
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def load_frame_clock(video, run_dir):
    video, run = Path(video).resolve(), Path(run_dir)
    st = video.stat()
    identity = dict(video=str(video), size=st.st_size, mtime_ns=st.st_mtime_ns,
                    timebase_version=TIMEBASE_VERSION)
    key = (str(video), st.st_size, st.st_mtime_ns, str(run.resolve()))
    if key in _CLOCKS:
        return _CLOCKS[key]
    extra = run / 'extra'
    extra.mkdir(parents=True, exist_ok=True)
    # Separate table per video, including separate calibration recordings.
    import hashlib
    tag = hashlib.sha256(str(video).encode()).hexdigest()[:12]
    table = extra / f'frame_timestamps_{tag}.csv'
    meta = table.with_suffix('.json')
    if table.exists() and meta.exists() and json.loads(meta.read_text()) == identity:
        clock = FrameClock(pd.read_csv(table)['video_time_ms'].to_numpy())
    else:
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(video)],
            capture_output=True, text=True, check=True)
        records = json.loads(result.stdout).get('frames', [])
        if not records or any('best_effort_timestamp_time' not in r for r in records):
            raise ValueError('Missing frame PTS; refusing average-FPS fallback')
        clock = FrameClock([float(r['best_effort_timestamp_time'])*1000 for r in records])
        pd.DataFrame({'frame': np.arange(len(clock.times_ms)),
                      'video_time_ms': clock.times_ms}).to_csv(table, index=False)
        meta.write_text(json.dumps(identity, indent=2))
    _CLOCKS[key] = clock
    return clock


def phase_offset(phases):
    info = phases['sync_info']
    key = 'offset_880hz_ms' if info.get('method') == 'dual_anchor' and 'offset_880hz_ms' in info else 'offset_ms'
    return float(info[key])


def sync_offset(run_dir):
    info = json.loads((Path(run_dir) / 'debug_4_sync_info.json').read_text())
    offset = float(info['offset_used_ms'])
    if not np.isfinite(offset):
        raise ValueError('Non-finite synchronization offset')
    return offset


def assign_trials(df, phases):
    """Assign from the corrected EyeLink clock, never retain stale frame labels."""
    out = df.copy()
    out['trial_number'] = -1
    out['trial_assignment'] = -1
    out['phase_type'] = 'unassigned'
    t = out['timestamp_ms_synced']
    for block in ['experiment_block1', 'experiment_block2']:
        for trial in phases['phases'].get(block, {}).get('trials', []):
            whole = (t >= trial['fixation']['start_eyelink_ms']) & (t < trial['stimulus']['end_eyelink_ms'])
            if (out.loc[whole, 'trial_number'] >= 0).any():
                raise ValueError('Overlapping trial definitions')
            out.loc[whole, ['trial_number', 'trial_assignment']] = int(trial['trial_number'])
            out.loc[whole, 'phase_type'] = 'transition'
            for name in ['fixation', 'stimulus']:
                phase = trial[name]
                mask = (t >= phase['start_eyelink_ms']) & (t < phase['end_eyelink_ms'])
                out.loc[mask, ['trial_number', 'trial_assignment']] = int(trial['trial_number'])
                out.loc[mask, 'phase_type'] = name
    return out


def correct_saved_gaze(gaze, frame_log, run_dir, *, frame_identity_verified=False):
    """Temporal-only migration: preserve model outputs and original timestamps."""
    if not frame_identity_verified:
        raise ValueError('Cannot repair legacy gaze with unverified frame identity; fresh exact decoding is required')
    from blink.run_data import join_blink_mask
    out = join_blink_mask(gaze, frame_log)
    reference = frame_log.set_index('frame_number')['video_time_ms']
    actual_time = out['frame'].map(reference)
    if actual_time.isna().any():
        raise ValueError('Frame PTS does not cover all gaze frames')
    # Preserve the earliest original values when a repair is run repeatedly.
    for col in ['timestamp_ms', 'timestamp_ms_synced', 'trial_number', 'phase_type']:
        if col in out and col+'_original' not in out:
            out[col+'_original'] = out[col]
    out['timestamp_ms'] = actual_time.to_numpy(float)
    out['timestamp_ms_synced'] = out['timestamp_ms'] + sync_offset(run_dir)
    out['timestamp_source'] = TIMEBASE_VERSION
    phases = json.loads((Path(run_dir) / 'phases_detected.json').read_text())
    return assign_trials(out, phases)


def synchronized_frame_log(frame_log, run_dir):
    out = frame_log.copy().sort_values('frame_number').reset_index(drop=True)
    clock = FrameClock(out['video_time_ms'].to_numpy())
    offset = sync_offset(run_dir)
    out['timestamp_ms_synced'] = clock.times_ms + offset
    if 'frame_end_time_ms' in out:
        end = pd.to_numeric(out['frame_end_time_ms'], errors='raise').to_numpy(float)
    else:
        end = np.r_[clock.times_ms[1:], clock.times_ms[-1]+np.median(np.diff(clock.times_ms))]
    if not np.isfinite(end).all() or np.any(end <= clock.times_ms):
        raise ValueError('Invalid frame end timestamp')
    out['sample_end_ms'] = end + offset
    return assign_trials(out, json.loads((Path(run_dir) / 'phases_detected.json').read_text()))


def detector_input(gaze, frame_log, run_dir):
    """Retain missing model frames as invalid, so the core's gap rule can split.

    No gaze interpolation is performed. The unchanged detector needs explicit
    invalid rows; dropping missing frames would allow it to bridge long gaps.
    """
    full = synchronized_frame_log(frame_log, run_dir)
    full = full[full['trial_number'].isin(gaze.loc[gaze['trial_number'] >= 0,
                                                 'trial_number'].unique())].copy()
    cols = [c for c in gaze.columns if c not in full.columns or c == 'frame']
    merged = full.merge(gaze[cols], left_on='frame_number', right_on='frame',
                        how='left', validate='one_to_one', indicator='_gaze_merge')
    merged['has_calibrated_gaze'] = merged['_gaze_merge'].eq('both')
    merged['frame'] = merged['frame_number']
    from blink.run_data import boolean_mask
    merged['valid_eye_frame'] = boolean_mask(merged['valid_eye_frame']) & merged['has_calibrated_gaze']
    for col in ['detected', 'plausibility_check']:
        if col in merged:
            merged[col] = boolean_mask(merged[col].where(merged[col].notna(), False))
    return merged.drop(columns='_gaze_merge')
