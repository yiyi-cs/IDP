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


def load_blink_intervals(frame_log_csv: Path, trials: List[TrialInfo],
                         pt_df: pd.DataFrame) -> Dict[int, List[Tuple[float, float]]]:
    """Use the same complete frame support as the blink diagnostic PDF."""
    from shared.frame_timing import synchronized_frame_log
    from blink.run_data import boolean_mask
    log = synchronized_frame_log(pd.read_csv(frame_log_csv), frame_log_csv.parent.parent)
    out = {}
    for tr in trials:
        q = log[log['timestamp_ms_synced'].between(
            tr.trial_start_eye_ms, tr.trial_end_eye_ms, inclusive='left')].copy()
        invalid = q[~boolean_mask(q['valid_eye_frame'])]
        groups = invalid['frame_number'].diff().ne(1).cumsum()
        out[tr.trial_number] = [
            (float(g['timestamp_ms_synced'].min()) - tr.trial_start_eye_ms,
             min(float(g['sample_end_ms'].max()), tr.trial_end_eye_ms) - tr.trial_start_eye_ms)
            for _, g in invalid.groupby(groups, sort=False)]
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
    valid = np.isfinite(q['x_pos']) & q['x_pos'].gt(0)
    for column in ['y_pos', 'pupil_size']:
        if column in q:
            valid &= pd.to_numeric(q[column], errors='coerce').gt(0)
    q.loc[~valid, 'x_pos'] = np.nan
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


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixation.eyelink_sanity import pixels_to_degrees as eyelink_pixels_to_degrees
from blink.run_data import load_validated_gaze


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


def render_run(run_dir, vp, method="ptgaze", trial_numbers=None):
    """Render saved current-Run events; plotting never recomputes a detector."""
    run_dir = Path(run_dir)
    extra = run_dir / 'extra'
    gaze_file = ('debug_5_ptgaze_calibrated.csv' if method == 'ptgaze'
                 else 'debug_5_pupil_data_calibrated.csv')
    from shared.frame_timing import detector_input
    pt_df = detector_input(load_validated_gaze(run_dir, gaze_file),
                           pd.read_csv(extra / 'blink_filter_frame_log.csv'), run_dir)
    config_path = run_dir / 'config_used.json'
    diagnostic_only = config_path.exists() and json.loads(config_path.read_text()).get('diagnostic_only', False)
    trials = load_trials(run_dir / 'phases_detected.json')
    if trial_numbers is not None:
        trials = [t for t in trials if t.trial_number in trial_numbers]
    if not trials:
        raise ValueError('No experimental trials available for fixation PDF')
    blink_intervals = load_blink_intervals(extra / 'blink_filter_frame_log.csv', trials, pt_df)
    eye_df = load_eyelink_gaze(run_dir / 'debug_3_eyetracker_data.csv')
    native_fix = load_fixations_native(run_dir / 'debug_3_fixations.csv')
    ours_pt = load_fixations_ours(run_dir / f'debug_6_{method}_fixations.csv')
    ours_eye = load_fixations_ours(extra / 'eyelink_fixations_sanity.csv')
    pdf_path = extra / 'fixation_four_row_timelines.pdf'
    with PdfPages(pdf_path) as pdf:
        for tr in trials:
            fig = plt.figure(figsize=(13.0, 8.0))
            plot_trial(fig, tr, eye_df, pt_df, native_fix, ours_eye, ours_pt,
                       blink_intervals, title_prefix=f'{vp} - ')
            if method == 'mediapipe':
                for ax in fig.axes:
                    ax.set_title(ax.get_title().replace('PTGaze', 'MediaPipe'))
                    legend = ax.get_legend()
                    if legend:
                        for text in legend.get_texts():
                            text.set_text(text.get_text().replace('PTGaze', 'MediaPipe'))
            if diagnostic_only:
                fig.subplots_adjust(bottom=.11)
                fig.text(.08, .012, 'Temporal diagnostic only: legacy model outputs; not a raw-frame acceptance result.', fontsize=8, color='#9B4700')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    return pdf_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--vp', required=True)
    ap.add_argument('--method', choices=['ptgaze', 'mediapipe'], default='ptgaze')
    args = ap.parse_args()
    print(render_run(args.run_dir, args.vp, args.method))


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
