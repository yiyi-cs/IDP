#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

DEFAULT_VPS = [
    'beo7','bjs4','egf5','fbn6','fgt6','jkl7','kdn8','kro3','mhe9','oem4','ogt7'
]

SCREEN_WIDTH_PX = 1920.0
SCREEN_WIDTH_CM = 53.2
VIEWING_DISTANCE_CM = 70.0
SCREEN_CENTER_X_PX = (SCREEN_WIDTH_PX - 1.0) / 2.0

MEDIAPIPE_STYLE = {'color': '#D73027', 'lw': 0.95, 'alpha': 0.95, 'label': 'MediaPipe'}
PTGAZE_STYLE   = {'color': '#4575B4', 'lw': 0.95, 'alpha': 0.95, 'label': 'PTGaze'}
EYELINK_STYLE  = {'color': '#111111', 'lw': 0.75, 'alpha': 0.82, 'label': 'EyeLink'}
EYELINK_BLINK_STYLE = {
    'facecolor': '#D9A441',
    'edgecolor': '#9A6D00',
    'alpha': 0.24,
    'hatch': '////',
    'label': 'EyeLink EBLINK',
}
MIN_EVENT_DISPLAY_S = 0.010

LEFT = [33,160,158,133,153,144]
RIGHT= [362,385,387,263,373,380]


def find_complete_run(vpdir: Path) -> Path:
    analyse = vpdir / 'Analyse'
    runs = [p for p in analyse.glob('Run_60hz_FullCalib_*') if p.is_dir()]
    required = ['debug_3_eyetracker_data.csv', 'debug_5_pupil_data_calibrated.csv', 'debug_5_ptgaze_calibrated.csv', 'phases_detected.json']
    good = []
    for run in runs:
        if all((run / name).exists() for name in required):
            good.append(run)
    if not good:
        raise FileNotFoundError(f'No complete 60 Hz FullCalib run under {analyse}')
    return max(good, key=lambda p: p.stat().st_mtime)


def paths_for(results_root: Path, video_root: Path, vp: str) -> Dict[str, Path]:
    vpdir = results_root / vp
    run = find_complete_run(vpdir)
    test = vpdir / 'test60'
    d = {
        'run': run,
        'phases': run / 'phases_detected.json',
        'mp': run / 'debug_5_pupil_data_calibrated.csv',
        'pt': run / 'debug_5_ptgaze_calibrated.csv',
        'eyegaze': run / 'debug_3_eyetracker_data.csv',
        'frame': test / 'blink_filter_frame_log.csv',
        'eye': test / 'eyelink_blink_benchmark.csv',
        'markers': test / 'eyelink_trial_markers.csv',
        'video': video_root / f'{vp}.mp4',
    }
    for k, p in d.items():
        if k != 'run' and not p.exists():
            raise FileNotFoundError(p)
    return d


def trials_from_json(path: Path) -> Dict[int, dict]:
    d = json.load(open(path, encoding='utf-8'))
    p = d['phases']
    xs = p.get('experiment_block1', {}).get('trials', []) + p.get('experiment_block2', {}).get('trials', [])
    out = {}
    for x in xs:
        if x.get('is_practice', False):
            continue
        f = x['fixation']
        s = x['stimulus']
        out[int(x['trial_number'])] = {
            'start_ms': float(f['start_video_s']) * 1000.0,
            'end_ms': float(s['end_video_s']) * 1000.0,
            'fix_end_ms': float(f['end_video_s']) * 1000.0,
            'stim_start_ms': float(s['start_video_s']) * 1000.0,
        }
    return out


def eyelink_x_px_to_deg(x_px: pd.Series) -> pd.Series:
    x_cm = (x_px - SCREEN_CENTER_X_PX) * (SCREEN_WIDTH_CM / SCREEN_WIDTH_PX)
    return np.degrees(np.arctan2(x_cm, VIEWING_DISTANCE_CM))


def read_cam_method(path: Path) -> pd.DataFrame:
    """Read calibrated gaze using original pipeline timestamps and trial IDs.

    Do not require video_time_ms: it belongs to the independent blink export,
    not to the calibrated gaze CSVs.
    """
    df = pd.read_csv(path)
    trial_col = 'trial_number' if 'trial_number' in df.columns else 'trial_assignment'
    need = ['timestamp_ms_synced', 'gaze_deg_x_calib', trial_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f'{path.name} missing columns: {missing}')
    return pd.DataFrame({
        'timestamp_ms_synced': pd.to_numeric(df['timestamp_ms_synced'], errors='coerce'),
        'gaze_deg_x': pd.to_numeric(df['gaze_deg_x_calib'], errors='coerce'),
        'trial_number': pd.to_numeric(df[trial_col], errors='coerce'),
    }).dropna(subset=['timestamp_ms_synced'])


def read_eyelink_gaze(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = 'timestamp_ms' if 'timestamp_ms' in df.columns else None
    x_col = 'x_pos' if 'x_pos' in df.columns else ('x_px' if 'x_px' in df.columns else None)
    if ts_col is None or x_col is None:
        raise ValueError(f'{path.name} missing EyeLink columns timestamp_ms/x_pos')
    out = pd.DataFrame({
        'timestamp_ms': pd.to_numeric(df[ts_col], errors='coerce'),
        'gaze_deg_x': eyelink_x_px_to_deg(pd.to_numeric(df[x_col], errors='coerce')),
    })
    return out.dropna(subset=['timestamp_ms'])


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def read_marker_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    trial_col = _pick_col(df, ['trial', 'trial_number'])
    fix_col = _pick_col(df, ['fixation_start_tracker_ms', 'fixation_start_timestamp_ms',
                             'fixation_start_ms'])
    end_col = _pick_col(df, ['stimulus_end_tracker_ms', 'stimulus_end_timestamp_ms',
                             'stimulus_end_ms'])
    if trial_col is None or fix_col is None or end_col is None:
        raise ValueError(f'{path.name} does not expose usable trial / fixation-start / stimulus-end columns')
    out = pd.DataFrame({
        'trial': pd.to_numeric(df[trial_col], errors='coerce'),
        'fix_start_ms': pd.to_numeric(df[fix_col], errors='coerce'),
        'stim_end_ms': pd.to_numeric(df[end_col], errors='coerce'),
    })
    if 'is_practice' in df.columns:
        practice = df['is_practice'].astype('string').str.lower().isin(['true', '1'])
        out = out.loc[~practice].copy()
    out = out.dropna(subset=['trial', 'fix_start_ms', 'stim_end_ms']).copy()
    out['trial'] = out['trial'].astype(int)
    if out['trial'].duplicated().any():
        raise ValueError(f'{path.name} has duplicate markers for non-practice trials')
    return out


def truth_events(df: pd.DataFrame, trial: int) -> List[dict]:
    x = df[pd.to_numeric(df['trial'], errors='coerce').eq(trial)].copy()
    start_col = _pick_col(x, ['start_from_fixation_ms', 'start_ms'])
    end_col = _pick_col(x, ['end_from_fixation_ms', 'end_ms'])
    if start_col is None or end_col is None:
        raise ValueError('EyeLink benchmark CSV missing start/end columns')
    s = pd.to_numeric(x[start_col], errors='coerce')
    e = pd.to_numeric(x[end_col], errors='coerce')
    out = []
    for a, b in zip(s, e):
        if np.isfinite(a) and np.isfinite(b):
            dur = float(b - a)
            out.append({
                'start_ms': float(a),
                'end_ms': float(b),
                'mid_ms': float(a + b) / 2.0,
                'duration_ms': dur,
            })
    return out


def read_frame_log(path: Path, start_ms: float, end_ms: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    t = pd.to_numeric(df['video_time_ms'], errors='coerce')
    q = df[t.between(start_ms, end_ms)].copy()
    q['rel_s'] = (pd.to_numeric(q['video_time_ms'], errors='coerce') - start_ms) / 1000.0
    return q


def nearest_row(q: pd.DataFrame, target_abs_ms: float):
    if q.empty:
        return None
    t = pd.to_numeric(q['video_time_ms'], errors='coerce')
    valid = t.notna()
    if not valid.any():
        return None
    idx = (t[valid] - float(target_abs_ms)).abs().idxmin()
    return q.loc[idx]


def extract_frame_at_pts(video: Path, pts_s: float):
    cmd = [
        'ffmpeg', '-v', 'error', '-ss', f'{max(0.0, pts_s):.6f}',
        '-i', str(video), '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    arr = np.frombuffer(r.stdout, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def ear(p):
    h = np.linalg.norm(p[0] - p[3])
    return float((np.linalg.norm(p[1]-p[5]) + np.linalg.norm(p[2]-p[4])) / (2*h)) if h else np.nan


def eye_roi(mesh, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    if not res.multi_face_landmarks:
        return rgb, None
    h, w = frame.shape[:2]
    lm = res.multi_face_landmarks[0]
    lp = np.array([[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in LEFT])
    rp = np.array([[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in RIGHT])
    p = np.vstack([lp, rp])
    x0 = max(0, int(p[:,0].min() - 55)); x1 = min(w, int(p[:,0].max() + 55))
    y0 = max(0, int(p[:,1].min() - 45)); y1 = min(h, int(p[:,1].max() + 45))
    crop = rgb[y0:y1, x0:x1].copy()
    for pts in (lp, rp):
        for k, (x, y) in enumerate(pts):
            cv2.circle(crop, (int(x-x0), int(y-y0)), 5, (255, 0, 0), -1)
            cv2.putText(crop, str(k), (int(x-x0)+4, int(y-y0)-4), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,0), 1)
    return crop, (lp, rp)


def build_trial_line_data(mp_df, pt_df, eye_df, markers_df, trial_bounds, trial: int):
    z = trial_bounds[trial]
    start_ms, end_ms = z['start_ms'], z['end_ms']
    fix_start_row = markers_df[markers_df['trial'].eq(trial)]
    if fix_start_row.empty:
        raise ValueError(f'No EyeLink markers for trial {trial}')
    marker = fix_start_row.iloc[0]
    eye_start = float(marker['fix_start_ms'])
    eye_end = float(marker['stim_end_ms'])

    def cam_slice(df, label):
        x = df[df['trial_number'].eq(trial) & df['timestamp_ms_synced'].between(eye_start, eye_end)].copy()
        x['rel_s'] = (pd.to_numeric(x['timestamp_ms_synced'], errors='coerce') - eye_start) / 1000.0
        x['method'] = label
        return x[['rel_s', 'gaze_deg_x', 'method']]

    mp_line = cam_slice(mp_df, 'MediaPipe')
    pt_line = cam_slice(pt_df, 'PTGaze')
    eg = eye_df[eye_df['timestamp_ms'].between(eye_start, eye_end)].copy()
    eg['rel_s'] = (eg['timestamp_ms'] - eye_start) / 1000.0
    eg['method'] = 'EyeLink'
    eg = eg[['rel_s', 'gaze_deg_x', 'method']]
    line = pd.concat([mp_line, pt_line, eg], ignore_index=True)
    return line, eye_start, eye_end


def plot_trial_top(ax, line_data: pd.DataFrame, events: List[dict], numbered_events: List[dict]):
    for method, style in [('MediaPipe', MEDIAPIPE_STYLE), ('PTGaze', PTGAZE_STYLE), ('EyeLink', EYELINK_STYLE)]:
        x = line_data[line_data['method'] == method]
        if not x.empty:
            ax.plot(x['rel_s'], x['gaze_deg_x'], color=style['color'], lw=style['lw'], alpha=style['alpha'], label=style['label'])
    for ev in events:
        start_s = ev['start_ms'] / 1000.0
        width_s = max((ev['end_ms'] - ev['start_ms']) / 1000.0, MIN_EVENT_DISPLAY_S)
        ax.broken_barh([(start_s, width_s)], (0, 1), transform=ax.get_xaxis_transform(),
                       facecolors=EYELINK_BLINK_STYLE['facecolor'], edgecolors=EYELINK_BLINK_STYLE['edgecolor'],
                       alpha=EYELINK_BLINK_STYLE['alpha'], hatch=EYELINK_BLINK_STYLE['hatch'], linewidth=0.9)

    # Number only the >40 ms events that also appear as eye ROI examples below.
    for ev in numbered_events:
        start_s = ev['start_ms'] / 1000.0
        end_s = ev['end_ms'] / 1000.0
        center_s = ev['mid_ms'] / 1000.0
        label_x = max(start_s + 0.01, center_s)
        ax.text(label_x, 0.02, f"#{ev['event_index']}", transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color=EYELINK_BLINK_STYLE['edgecolor'],
                bbox=dict(boxstyle='round,pad=0.14', facecolor='white', edgecolor=EYELINK_BLINK_STYLE['edgecolor'], alpha=0.9))
    ax.set_title('1. Horizontal gaze and EyeLink blink intervals', loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel('Gaze X (deg)')
    ax.grid(True, alpha=.25)
    legend_handles = [
        plt.Line2D([0], [0], color=MEDIAPIPE_STYLE['color'], lw=MEDIAPIPE_STYLE['lw'], label='MediaPipe'),
        plt.Line2D([0], [0], color=PTGAZE_STYLE['color'], lw=PTGAZE_STYLE['lw'], label='PTGaze'),
        plt.Line2D([0], [0], color=EYELINK_STYLE['color'], lw=EYELINK_STYLE['lw'], label='EyeLink'),
        Patch(facecolor=EYELINK_BLINK_STYLE['facecolor'], edgecolor=EYELINK_BLINK_STYLE['edgecolor'],
              alpha=EYELINK_BLINK_STYLE['alpha'], hatch=EYELINK_BLINK_STYLE['hatch'], label='EyeLink EBLINK'),
    ]
    ax.legend(handles=legend_handles, ncol=4, fontsize=8, loc='upper right')


def render_trial_pages(pdf: PdfPages, vp: str, trial: int, line_data: pd.DataFrame, events_all: List[dict], events_gt40: List[dict],
                       frame_q: pd.DataFrame, video_path: Path, trial_start_ms: float):
    # 1 top panel + up to 8 eye images per page.
    chunks = [events_gt40[i:i+8] for i in range(0, len(events_gt40), 8)] or [[]]
    mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                           min_detection_confidence=.5, min_tracking_confidence=.5)
    for page_idx, chunk in enumerate(chunks, start=1):
        fig = plt.figure(figsize=(15, 11.8))
        outer = fig.add_gridspec(3, 1, height_ratios=[1.20, 0.13, 1.90], hspace=.34)
        ax = fig.add_subplot(outer[0])
        plot_trial_top(ax, line_data, events_all, events_gt40)
        x_max = max(9.5, float(np.nanmax(line_data['rel_s'])) if len(line_data) else 9.5)
        ax.set_xlim(0, x_max)
        ax.set_xlabel('Time from fixation start (s)')

        title_ax = fig.add_subplot(outer[1]); title_ax.axis('off')
        title_ax.text(0, 0.5, f'2. Eye ROIs at EyeLink EBLINK midpoints > 40 ms  ({len(events_gt40)} events total)',
                      ha='left', va='center', fontsize=11, fontweight='bold')

        sg = outer[2].subgridspec(2, 4, wspace=.17, hspace=.50)
        for k in range(8):
            ax2 = fig.add_subplot(sg[k//4, k%4]); ax2.axis('off')
            if k >= len(chunk):
                continue
            ev = chunk[k]
            abs_ms = trial_start_ms + ev['mid_ms']
            row = nearest_row(frame_q, abs_ms) if np.isfinite(abs_ms) else None
            if row is None:
                ax2.set_title(f'#{ev["event_index"]} | midpoint {ev["mid_ms"]/1000:.2f}s\n(no frame found)', fontsize=8)
                continue
            pts_s = float(row['video_time_ms']) / 1000.0
            frame_no = int(row['frame_number']) if 'frame_number' in row.index and pd.notna(row['frame_number']) else -1
            im = extract_frame_at_pts(video_path, pts_s)
            if im is None:
                ax2.set_title(f'#{ev["event_index"]} | midpoint {ev["mid_ms"]/1000:.2f}s | frame {frame_no}\n(frame decode failed)', fontsize=8)
                continue
            crop, pts = eye_roi(mesh, im)
            ax2.imshow(crop)
            recomputed = ''
            if pts is not None:
                recomputed = f'\nROI EAR L={ear(pts[0]):.3f} R={ear(pts[1]):.3f}'
            ax2.set_title(
                f'#{ev["event_index"]} | t={ev["mid_ms"]/1000:.2f}s | frame {frame_no}\n'
                f'EBLINK {ev["start_ms"]/1000:.2f}-{ev["end_ms"]/1000:.2f}s  dur={ev["duration_ms"]:.0f} ms'
                + recomputed,
                fontsize=7.3,
            )
            ax2.axis('off')

        suffix = '' if len(chunks) == 1 else f'  (page {page_idx}/{len(chunks)})'
        fig.suptitle(f'{vp} — Trial {trial} — EyeLink inspection{suffix}', fontsize=14, fontweight='bold', y=.995)
        fig.subplots_adjust(top=.955, bottom=.04, left=.06, right=.985)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    mesh.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', type=Path, default=Path('/Volumes/Empra10/Ergebnisse60'))
    ap.add_argument('--video-root', type=Path, default=Path('/Volumes/Empra9/Videos_60'))
    ap.add_argument('--output-dir', type=Path, default=None)
    ap.add_argument('--vp', action='append', default=None)
    ap.add_argument('--trial', type=int, action='append', default=None)
    ap.add_argument('--min-duration-ms', type=float, default=40.0)
    args = ap.parse_args()

    vps = args.vp or DEFAULT_VPS
    output_dir = args.output_dir or (args.results_root / 'blink_extra_outputs' / 'eyelink_inspection')
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / 'all_vps_eyelink_inspection.pdf'

    with PdfPages(pdf_path) as pdf:
        for vp in vps:
            try:
                fs = paths_for(args.results_root, args.video_root, vp)
                print(f'[SOURCE] {vp}: {fs["run"].name}')
                mp_df = read_cam_method(fs['mp'])
                pt_df = read_cam_method(fs['pt'])
                eye_df = read_eyelink_gaze(fs['eyegaze'])
                bench_df = pd.read_csv(fs['eye'])
                markers_df = read_marker_table(fs['markers'])
                trials = trials_from_json(fs['phases'])
                wanted = args.trial or sorted(trials)
                for tr in wanted:
                    if tr not in trials:
                        continue
                    events_all = truth_events(bench_df, tr)
                    events_gt40 = [ev for ev in events_all if ev['duration_ms'] > args.min_duration_ms]
                    for idx, ev in enumerate(events_gt40, start=1):
                        ev['event_index'] = idx
                    line_data, _, _ = build_trial_line_data(mp_df, pt_df, eye_df, markers_df, trials, tr)
                    frame_q = read_frame_log(fs['frame'], trials[tr]['start_ms'], trials[tr]['end_ms'])
                    render_trial_pages(pdf, vp, tr, line_data, events_all, events_gt40, frame_q, fs['video'], trials[tr]['start_ms'])
                print(f'[OK] {vp}')
            except Exception as e:
                print(f'[ERROR] {vp}: {type(e).__name__}: {e}', file=sys.stderr)

    print(f'\nSaved PDF: {pdf_path}')


if __name__ == '__main__':
    main()
