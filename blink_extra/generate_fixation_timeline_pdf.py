#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

# ----------- config -----------
EYELINK_COLOR = '#111111'
PTGAZE_COLOR = '#4575B4'
BLINK_COLOR = '#4C78A8'  # keep aligned with webcam regions used earlier
BLINK_EDGE = '#1F4E79'
NATIVE_FIX_COLOR = '#9E9E9E'
OURS_EYELINK_COLOR = '#E17C05'
OURS_PTGAZE_COLOR = '#2AA879'
FIXATION_PHASE_COLOR = '#D9D9D9'
SMOOTH_WINDOW_MS = 100.0
MIN_FIX_MS = 100.0
MAX_INVALID_GAP_MS = 50.0
EPS = 1e-9


@dataclass
class TrialInfo:
    trial_number: int
    is_practice: bool
    fix_start_video_ms: float
    fix_end_video_ms: float
    stim_start_video_ms: float
    stim_end_video_ms: float
    fix_start_eye_ms: float
    fix_end_eye_ms: float
    stim_start_eye_ms: float
    stim_end_eye_ms: float

    @property
    def trial_start_video_ms(self) -> float:
        return self.fix_start_video_ms

    @property
    def trial_end_video_ms(self) -> float:
        return self.stim_end_video_ms

    @property
    def trial_start_eye_ms(self) -> float:
        return self.fix_start_eye_ms

    @property
    def trial_end_eye_ms(self) -> float:
        return self.stim_end_eye_ms

    @property
    def total_duration_ms(self) -> float:
        return self.trial_end_video_ms - self.trial_start_video_ms

    @property
    def fix_duration_ms(self) -> float:
        return self.fix_end_video_ms - self.fix_start_video_ms


def load_trials(phases_json: Path, include_practice: bool = False) -> List[TrialInfo]:
    with open(phases_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = []
    for block_name in ['experiment_block1', 'experiment_block2']:
        block = data['phases'].get(block_name, {})
        for t in block.get('trials', []):
            if (not include_practice) and t.get('is_practice', False):
                continue
            out.append(TrialInfo(
                trial_number=int(t['trial_number']),
                is_practice=bool(t.get('is_practice', False)),
                fix_start_video_ms=float(t['fixation']['start_video_s']) * 1000.0,
                fix_end_video_ms=float(t['fixation']['end_video_s']) * 1000.0,
                stim_start_video_ms=float(t['stimulus']['start_video_s']) * 1000.0,
                stim_end_video_ms=float(t['stimulus']['end_video_s']) * 1000.0,
                fix_start_eye_ms=float(t['fixation']['start_eyelink_ms']),
                fix_end_eye_ms=float(t['fixation']['end_eyelink_ms']),
                stim_start_eye_ms=float(t['stimulus']['start_eyelink_ms']),
                stim_end_eye_ms=float(t['stimulus']['end_eyelink_ms']),
            ))
    out.sort(key=lambda z: z.trial_number)
    return out


def load_blink_intervals(frame_log_csv: Path, trials: List[TrialInfo]) -> Dict[int, List[Tuple[float, float]]]:
    df = pd.read_csv(frame_log_csv)
    if 'video_time_ms' not in df.columns:
        raise ValueError('blink_filter_frame_log.csv missing video_time_ms')
    # Determine invalid / interrupted frames.
    if 'valid_eye_frame' in df.columns:
        invalid_mask = ~pd.Series(df['valid_eye_frame']).fillna(False).astype(bool)
    else:
        invalid_mask = pd.Series(False, index=df.index)
    if 'eyes_closed' in df.columns:
        invalid_mask |= pd.Series(df['eyes_closed']).fillna(False).astype(bool)
    if 'is_blink' in df.columns:
        invalid_mask |= pd.Series(df['is_blink']).fillna(False).astype(bool)
    if 'is_transition_frame' in df.columns:
        invalid_mask |= pd.Series(df['is_transition_frame']).fillna(False).astype(bool)
    times = pd.to_numeric(df['video_time_ms'], errors='coerce').to_numpy()
    out: Dict[int, List[Tuple[float, float]]] = {t.trial_number: [] for t in trials}

    # Group contiguous invalid rows by frame order.
    idx = np.where(invalid_mask.to_numpy())[0]
    if len(idx) == 0:
        return out
    groups = [[idx[0]]]
    for i in idx[1:]:
        if i == groups[-1][-1] + 1:
            groups[-1].append(i)
        else:
            groups.append([i])

    for g in groups:
        start_ms = times[g[0]]
        end_ms = times[g[-1]]
        if not np.isfinite(start_ms) or not np.isfinite(end_ms):
            continue
        for tr in trials:
            s = max(start_ms, tr.trial_start_video_ms)
            e = min(end_ms, tr.trial_end_video_ms)
            if e > s:
                out[tr.trial_number].append((s - tr.trial_start_video_ms, e - tr.trial_start_video_ms))
    return out


def load_ptgaze(pt_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pt_csv)
    needed = ['trial_number', 'timestamp_ms_synced', 'gaze_deg_x_calib']
    for c in needed:
        if c not in df.columns:
            raise ValueError(f'PTGaze calibrated CSV missing {c}')
    q = df.copy()
    q['trial_number'] = pd.to_numeric(q['trial_number'], errors='coerce').astype('Int64')
    q['timestamp_ms_synced'] = pd.to_numeric(q['timestamp_ms_synced'], errors='coerce')
    q['gaze_deg_x_calib'] = pd.to_numeric(q['gaze_deg_x_calib'], errors='coerce')
    # validity / blink labels if available
    for c in ['is_blink', 'detected', 'plausibility_check']:
        if c in q.columns:
            q[c] = q[c].fillna(False)
    return q


def load_eyelink_gaze(eye_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(eye_csv)
    needed = ['timestamp_ms', 'x_pos']
    for c in needed:
        if c not in df.columns:
            raise ValueError(f'EyeLink gaze CSV missing {c}')
    q = df.copy()
    q['timestamp_ms'] = pd.to_numeric(q['timestamp_ms'], errors='coerce')
    q['x_pos'] = pd.to_numeric(q['x_pos'], errors='coerce')
    return q


def load_fixations_native(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.copy()
    for c in ['trial_number', 'start_time', 'end_time', 'duration']:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    if 'is_practice' in out.columns:
        out = out[~out['is_practice'].fillna(False)]
    return out


def load_fixations_ours(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.copy()
    for c in ['trial_number', 'start_time', 'end_time', 'duration', 'threshold_used_deg']:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    return out


def mask_intervals(times_ms: np.ndarray, intervals_rel_ms: List[Tuple[float, float]], trial_start_ms: float) -> np.ndarray:
    good = np.ones(len(times_ms), dtype=bool)
    for a_rel, b_rel in intervals_rel_ms:
        a = trial_start_ms + a_rel
        b = trial_start_ms + b_rel
        good &= ~((times_ms >= a) & (times_ms <= b))
    return good


def rolling_median_time(times_ms: np.ndarray, x: np.ndarray, window_ms: float = SMOOTH_WINDOW_MS) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    n = len(x)
    left = 0
    for i in range(n):
        t0 = times_ms[i] - window_ms / 2.0
        t1 = times_ms[i] + window_ms / 2.0
        while left < n and times_ms[left] < t0:
            left += 1
        right = left
        while right < n and times_ms[right] <= t1:
            right += 1
        vals = x[left:right]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[i] = np.median(vals)
    return out


def estimate_threshold_from_fixphases(times_ms: np.ndarray, x_smooth: np.ndarray, fixation_segments_ms: List[Tuple[float, float]], percentile: float = 75.0) -> float:
    disp_values = []
    for a, b in fixation_segments_ms:
        sel = (times_ms >= a) & (times_ms <= b) & np.isfinite(x_smooth)
        seg_t = times_ms[sel]
        seg_x = x_smooth[sel]
        if len(seg_t) < 2:
            continue
        left = 0
        for i in range(len(seg_t)):
            while seg_t[i] - seg_t[left] > SMOOTH_WINDOW_MS:
                left += 1
            vals = seg_x[left:i+1]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 2:
                disp_values.append(float(np.nanmax(vals) - np.nanmin(vals)))
    if not disp_values:
        return np.nan
    return float(np.percentile(disp_values, percentile))


def detect_fixations_idt_x(times_ms: np.ndarray, x_deg: np.ndarray, threshold_deg: float,
                           min_fix_ms: float = MIN_FIX_MS, max_invalid_gap_ms: float = MAX_INVALID_GAP_MS) -> List[dict]:
    valid = np.isfinite(times_ms) & np.isfinite(x_deg)
    if valid.sum() < 2:
        return []
    times = times_ms[valid]
    x = x_deg[valid]
    # Split into segments across long gaps.
    gaps = np.diff(times)
    break_idx = np.where(gaps > max_invalid_gap_ms)[0]
    starts = np.r_[0, break_idx + 1]
    ends = np.r_[break_idx + 1, len(times)]
    out = []
    for seg_start, seg_end in zip(starts, ends):
        t = times[seg_start:seg_end]
        xx = x[seg_start:seg_end]
        if len(t) < 2:
            continue
        i = 0
        while i < len(t) - 1:
            # find earliest j that reaches min duration
            j = i + 1
            while j < len(t) and (t[j] - t[i]) < min_fix_ms - EPS:
                j += 1
            if j >= len(t):
                break
            vals = xx[i:j+1]
            disp = np.nanmax(vals) - np.nanmin(vals)
            if disp <= threshold_deg + EPS:
                k = j
                while k + 1 < len(t):
                    cand = xx[i:k+2]
                    disp2 = np.nanmax(cand) - np.nanmin(cand)
                    if disp2 <= threshold_deg + EPS:
                        k += 1
                    else:
                        break
                out.append({
                    'start_time': float(t[i]),
                    'end_time': float(t[k]),
                    'duration': float(t[k] - t[i]),
                    'x_pos': float(np.nanmean(xx[i:k+1])),
                    'dispersion_x_deg': float(np.nanmax(xx[i:k+1]) - np.nanmin(xx[i:k+1])),
                    'threshold_used_deg': float(threshold_deg),
                })
                i = k + 1
            else:
                i += 1
    return out


def eyelink_pixels_to_degrees(x_px: np.ndarray) -> np.ndarray:
    """Convert EyeLink screen x pixels to visual angle using experiment geometry.

    Same geometry used for the preliminary bjs4 four-row visualisation:
    1920 px monitor width, 53.2 cm width, 70 cm viewing distance.
    """
    x_px = np.asarray(x_px, dtype=float)
    return np.degrees(np.arctan2((x_px - (1920.0 - 1.0)/2.0) * 53.2/1920.0, 70.0))


def build_eyelink_sanity(trials: List[TrialInfo], eyelink_df: pd.DataFrame, pt_df: pd.DataFrame,
                         blink_intervals: Dict[int, List[Tuple[float, float]]]) -> Tuple[pd.DataFrame, float]:
    """Sample valid EyeLink x at PTGaze's actual synchronized timestamps.

    Threshold is fitted once across all experimental fixation phases (P75),
    then applied to every trial. This is a stand-alone sanity-check implementation,
    not a byte-for-byte reproduction of the project's detector.
    """
    eye = eyelink_df.sort_values('timestamp_ms').copy()
    valid = (eye['pupil_size'] > 0) & eye['x_pos'].notna() & eye['y_pos'].notna() & (eye['x_pos'] > 0) & (eye['y_pos'] > 0)
    eye['x_valid'] = eye['x_pos'].where(valid)
    # EyeLink runs at ~2000 Hz, so integer millisecond timestamps can repeat.
    eg = eye.groupby('timestamp_ms', as_index=False)['x_valid'].mean()
    et = eg['timestamp_ms'].to_numpy(float)
    ex = eg['x_valid'].to_numpy(float)
    sampled = []
    noise_disps = []
    for tr in trials:
        pt = pt_df[pt_df['trial_number'].eq(tr.trial_number)].sort_values('timestamp_ms_synced')
        pt = pt[(pt['timestamp_ms_synced'] >= tr.trial_start_eye_ms) &
                (pt['timestamp_ms_synced'] <= tr.trial_end_eye_ms)]
        if pt.empty:
            continue
        # Crucial: timestamp_ms_synced ALREADY uses the EyeLink clock.
        times = pt['timestamp_ms_synced'].to_numpy(float)
        ix = np.searchsorted(et, times, side='left')
        right = np.clip(ix, 0, len(et)-1)
        left = np.clip(ix-1, 0, len(et)-1)
        closest = np.where(np.abs(times-et[left]) <= np.abs(times-et[right]), left, right)
        vals = ex[closest].copy()
        vals[np.abs(times-et[closest]) > 2.0] = np.nan
        # Webcam blink / invalid regions have been expressed on the same video
        # trial-relative scale, so apply them to the sample's relative EyeLink time.
        relative_eye_ms = times - tr.trial_start_eye_ms
        for a_rel, b_rel in blink_intervals.get(tr.trial_number, []):
            vals[(relative_eye_ms >= a_rel) & (relative_eye_ms <= b_rel)] = np.nan
        x_deg = eyelink_pixels_to_degrees(vals)
        sampled.append((tr, times, x_deg))
        # Fit the threshold using the experimental fixation-cross phase only.
        phase_mask = (times >= tr.fix_start_eye_ms) & (times <= tr.fix_end_eye_ms)
        good = phase_mask & np.isfinite(x_deg)
        t = times[good]
        x = x_deg[good]
        if len(t) < 3:
            continue
        smoothed = rolling_median_time(t, x)
        # Forward 100 ms windows, matching preliminary sanity-check method.
        for i in range(len(t)):
            j = np.searchsorted(t, t[i] + SMOOTH_WINDOW_MS, side='left')
            if j < len(t) and j > i:
                noise_disps.append(float(np.ptp(smoothed[i:j+1])))
    if not noise_disps:
        raise ValueError('Could not estimate EyeLink adaptive threshold from fixation phases. Check time alignment and blink mask.')
    threshold = float(np.percentile(noise_disps, 75.0))
    records = []
    for tr, times, x_deg in sampled:
        # Smooth only continuous valid chunks, so blink gaps cannot be smoothed over.
        good_ids = np.flatnonzero(np.isfinite(x_deg))
        if len(good_ids) == 0:
            continue
        cuts = np.flatnonzero((np.diff(times[good_ids]) > MAX_INVALID_GAP_MS) |
                                 (np.diff(good_ids) > 1)) + 1
        for ids in np.split(good_ids, cuts):
            if len(ids) < 2:
                continue
            sm = rolling_median_time(times[ids], x_deg[ids])
            for fx in detect_fixations_idt_x(times[ids], sm, threshold):
                fx['trial_number'] = tr.trial_number
                records.append(fx)
    cols = ['trial_number', 'start_time', 'end_time', 'duration', 'x_pos',
            'dispersion_x_deg', 'threshold_used_deg']
    return pd.DataFrame.from_records(records, columns=cols), threshold

def rel_events(df: pd.DataFrame, tr: TrialInfo, time_mode: str) -> List[Tuple[float, float]]:
    x = df[df['trial_number'].eq(tr.trial_number)].copy()
    out = []
    for _, r in x.iterrows():
        a = float(r['start_time'])
        b = float(r['end_time'])
        if time_mode == 'eyelink_abs':
            a -= tr.trial_start_eye_ms
            b -= tr.trial_start_eye_ms
        elif time_mode == 'video_abs':
            a -= tr.trial_start_eye_ms
            b -= tr.trial_start_eye_ms
        else:
            raise ValueError(time_mode)
        out.append((a, b))
    return out


def plot_trial(fig, tr: TrialInfo, eyelink_df: pd.DataFrame, pt_df: pd.DataFrame,
               native_fix: pd.DataFrame, ours_eye: pd.DataFrame, ours_pt: pd.DataFrame,
               blink_intervals: Dict[int, List[Tuple[float, float]]], title_prefix: str = ''):
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 0.9, 0.9, 0.9], hspace=0.20)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    # --- row 1: gaze + blink regions ---
    eye = eyelink_df[(eyelink_df['timestamp_ms'] >= tr.trial_start_eye_ms) & (eyelink_df['timestamp_ms'] <= tr.trial_end_eye_ms)].copy()
    eye_rel_s = (eye['timestamp_ms'].to_numpy() - tr.trial_start_eye_ms) / 1000.0
    eye_x = eyelink_pixels_to_degrees(eye['x_pos'].to_numpy(float))
    pt = pt_df[pt_df['trial_number'].eq(tr.trial_number)].copy()
    pt_rel_s = (pt['timestamp_ms_synced'].to_numpy() - tr.trial_start_eye_ms) / 1000.0
    pt_x = pt['gaze_deg_x_calib'].to_numpy()

    ax1.plot(eye_rel_s, eye_x, color=EYELINK_COLOR, lw=0.8, alpha=0.75, label='EyeLink gaze')
    ax1.plot(pt_rel_s, pt_x, color=PTGAZE_COLOR, lw=1.0, alpha=0.95, label='PTGaze gaze')
    for a_ms, b_ms in blink_intervals.get(tr.trial_number, []):
        ax1.axvspan(a_ms/1000.0, b_ms/1000.0, facecolor=BLINK_COLOR, edgecolor=BLINK_EDGE, alpha=0.18, hatch='////', linewidth=0.7)
    # phase divider
    fix_end = (tr.fix_end_eye_ms - tr.trial_start_eye_ms) / 1000.0
    stim_start = (tr.stim_start_eye_ms - tr.trial_start_eye_ms) / 1000.0
    ax1.axvspan(0, fix_end, facecolor=FIXATION_PHASE_COLOR, alpha=0.20)
    ax1.axvline(fix_end, color='0.35', lw=1.0, ls='--')
    ax1.text(fix_end/2, 0.02, 'Fixation phase', transform=ax1.get_xaxis_transform(), ha='center', va='bottom', fontsize=8, color='0.35')
    ax1.text((stim_start + ((tr.trial_end_eye_ms-tr.trial_start_eye_ms)/1000.0))/2, 0.02, 'Stimulus phase', transform=ax1.get_xaxis_transform(), ha='center', va='bottom', fontsize=8, color='0.35')
    handles = [
        plt.Line2D([0],[0], color=EYELINK_COLOR, lw=1.0, label='EyeLink gaze'),
        plt.Line2D([0],[0], color=PTGAZE_COLOR, lw=1.0, label='PTGaze gaze'),
        Patch(facecolor=BLINK_COLOR, edgecolor=BLINK_EDGE, alpha=0.18, hatch='////', label='Webcam blink / invalid'),
    ]
    ax1.legend(handles=handles, loc='upper right', fontsize=8, ncol=3)
    ax1.set_ylabel('Horizontal gaze (deg)')
    ax1.set_title('1. EyeLink and PTGaze horizontal gaze with webcam blink interruptions', loc='left', fontsize=10.5, fontweight='bold')
    ax1.grid(True, alpha=0.25)

    # helper to draw event bars
    def draw_event_row(ax, events_rel_ms: List[Tuple[float, float]], color: str, title: str):
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for a, b in events_rel_ms:
            width = max((b - a) / 1000.0, 0.01)
            ax.broken_barh([(a / 1000.0, width)], (0.18, 0.64), facecolors=color, edgecolors=color, alpha=0.85)
        ax.set_title(title, loc='left', fontsize=10.5, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.2)

    draw_event_row(ax2, rel_events(native_fix, tr, 'eyelink_abs'), NATIVE_FIX_COLOR, '2. EyeLink native fixation events')
    draw_event_row(ax3, rel_events(ours_eye, tr, 'video_abs'), OURS_EYELINK_COLOR, '3. Our fixation detector on EyeLink gaze')
    draw_event_row(ax4, rel_events(ours_pt, tr, 'video_abs'), OURS_PTGAZE_COLOR, '4. Our fixation detector on PTGaze gaze')

    total_s = (tr.trial_end_eye_ms - tr.trial_start_eye_ms) / 1000.0
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(0, total_s)
    ax4.set_xlabel('Time from trial fixation start (s)')
    for ax in [ax1, ax2, ax3]:
        plt.setp(ax.get_xticklabels(), visible=False)

    fig.suptitle(f'{title_prefix}Trial {tr.trial_number}', fontsize=13, fontweight='bold', y=0.995)
    fig.subplots_adjust(top=0.94, left=0.08, right=0.985, bottom=0.06)


def main():
    ap = argparse.ArgumentParser(description='Generate four-row fixation timeline PDF with blink interruptions.')
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--vp', type=str, required=True)
    ap.add_argument('--output-dir', type=Path, default=None)
    ap.add_argument('--trial', type=int, nargs='*', default=None, help='Trial numbers to plot. Default: all non-practice trials.')
    ap.add_argument('--include-practice', action='store_true')
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.output_dir or (run_dir.parent.parent.parent / 'blink_extra_outputs' / 'fixation_visualization')
    out_dir.mkdir(parents=True, exist_ok=True)

    phases_path = run_dir / 'phases_detected.json'
    eyelink_gaze_path = run_dir / 'debug_3_eyetracker_data.csv'
    native_fix_path = run_dir / 'debug_3_fixations.csv'
    ptgaze_path = run_dir / 'debug_5_ptgaze_calibrated.csv'
    ours_pt_path = run_dir / 'debug_6_ptgaze_fixations.csv'
    frame_log_path = run_dir.parent.parent / 'test60' / 'blink_filter_frame_log.csv'

    for p in [phases_path, eyelink_gaze_path, native_fix_path, ptgaze_path, ours_pt_path, frame_log_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    trials = load_trials(phases_path, include_practice=args.include_practice)
    wanted = set(args.trial) if args.trial else None
    if wanted is not None:
        trials = [t for t in trials if t.trial_number in wanted]

    blink_intervals = load_blink_intervals(frame_log_path, trials)
    pt_df = load_ptgaze(ptgaze_path)
    eye_df = load_eyelink_gaze(eyelink_gaze_path)
    native_fix = load_fixations_native(native_fix_path)
    ours_pt = load_fixations_ours(ours_pt_path)

    # Fit participant-level P75 using all non-practice trials, even for --trial 1.
    all_trials = load_trials(phases_path, include_practice=False)
    all_blinks = load_blink_intervals(frame_log_path, all_trials)
    ours_eye, mean_thr = build_eyelink_sanity(all_trials, eye_df, pt_df, all_blinks)
    ours_eye_path = out_dir / f'{args.vp}_debug_6_eyelink_fixations_sanity.csv'
    ours_eye.to_csv(ours_eye_path, index=False)

    pdf_path = out_dir / f'{args.vp}_fixation_four_row_timelines.pdf'
    with PdfPages(pdf_path) as pdf:
        for tr in trials:
            fig = plt.figure(figsize=(13.0, 8.0))
            plot_trial(fig, tr, eye_df, pt_df, native_fix, ours_eye, ours_pt, blink_intervals, title_prefix=f'{args.vp} — ')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f'[OK] Saved sanity-check EyeLink fixations: {ours_eye_path}')
    print(f'[OK] Mean adaptive threshold on EyeLink sanity check: {mean_thr:.6f} deg')
    print(f'[OK] Saved PDF: {pdf_path}')


if __name__ == '__main__':
    main()
