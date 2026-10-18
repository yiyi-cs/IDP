"""Enhanced Blink Diagnostics (stage 1).

Place in IDP_Code/blink_extra/ alongside generate_blink_diagnostics.py.
Reads existing 60-Hz analysis outputs; does not recompute blink or gaze.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from blink import diagnostic_support as old

# Same webcam region styling in panels 1 and 2.
WEBCAM = "#3478A8"
EYELINK = "#DA8B27"
GAZE = {"MediaPipe": "#D73027", "PTGaze": "#4575B4", "EyeLink": "#111111"}


def _read_gaze(path: Path, zero_ms: float, *, eyelink: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if eyelink:
        # Match plot_blink_visualization.py's EyeLink screen geometry.
        time = pd.to_numeric(df["timestamp_ms"], errors="coerce")
        x = pd.to_numeric(df["x_pos"], errors="coerce")
        x_cm = (x - (1920.0-1.0)/2.0) * (53.2/1920.0)
        deg = np.degrees(np.arctan2(x_cm, 70.0))
    else:
        time = pd.to_numeric(df["timestamp_ms_synced"], errors="coerce")
        deg = pd.to_numeric(df["gaze_deg_x_calib"], errors="coerce")
    return pd.DataFrame({"t": (time-zero_ms)/1000.0, "x": deg}).dropna(subset=["t"])


def _reps(q, pred, truth, start):
    """Return 3 independent chronological groups, each with at most 4 examples.

    One consistent rule for this NEW visualization: EyeLink event midpoint
    lies within webcam interval (no tolerance and no 40-ms cutoff).
    """
    categories = {"EYELINK ONLY": [], "WEBCAM ONLY": [], "BOTH": []}
    mids = [(a+b)/2.0 for a,b in truth]
    linked = set()
    for p in pred:
        a,b,reason,*_ = p
        yes = [j for j,m in enumerate(mids) if a <= m <= b]
        linked.update(yes)
        row = old.lowest_row(q,start+a,start+b)
        if row is None: continue
        tag = "BOTH" if yes else "WEBCAM ONLY"
        categories[tag].append((tag,float(row.video_time_ms)/1000.0,
                                (a+b)/2000.0,reason,int(row.frame_number),
                                float(row.left_ear_raw),float(row.right_ear_raw),"min raw EAR"))
    for j,(a,b) in enumerate(truth):
        if j in linked: continue
        middle=(a+b)/2.0
        row=old.nearest_row(q,start+middle)
        if row is None: continue
        categories["EYELINK ONLY"].append(("EYELINK ONLY",float(row.video_time_ms)/1000.0,
                                middle/1000.0,"eyelink",int(row.frame_number),
                                float(row.left_ear_raw),float(row.right_ear_raw),"EBLINK midpoint"))
    for key in categories:
        categories[key] = sorted(categories[key], key=lambda r:r[2])[:4]
    return categories


def _trial_page(pdf,vp,tr,q,pred,truth,reps,video,method_gaze):
    fig = plt.figure(figsize=(16.5,17.8))
    outer=fig.add_gridspec(6,1,height_ratios=[1.8,1.25,1.0,.20,3.6,.06],hspace=.43)
    a=fig.add_subplot(outer[0])
    for label,data in method_gaze.items():
        shown=data[(data.t>=0)&(data.t<=10)]
        a.plot(shown.t,shown.x, color=GAZE[label],lw=.85,
               alpha=.90 if label!="EyeLink" else .75,label=label,zorder=3)
    for start,end,*_ in pred:
        a.axvspan(start/1000,end/1000,facecolor=WEBCAM,edgecolor=WEBCAM,
                  alpha=.25,zorder=1)
    for start,end in truth:
        a.axvspan(start/1000,end/1000,facecolor=EYELINK,edgecolor="#9D5914",
                  hatch="////",lw=.7,alpha=.24,zorder=2)
    a.set_title("1. Horizontal gaze and webcam / EyeLink blink intervals",loc="left",fontsize=11,fontweight="bold")
    a.set_ylabel("Gaze X (deg)")
    line_handles,labels=a.get_legend_handles_labels()
    a.legend(line_handles+[Patch(facecolor=WEBCAM,alpha=.25,label="Webcam blink"),
                  Patch(facecolor=EYELINK,edgecolor="#9D5914",hatch="////",alpha=.24,label="EyeLink EBLINK")],
             labels+["Webcam blink","EyeLink EBLINK"],ncol=5,fontsize=7,loc="upper right")
    a.grid(alpha=.2)

    b=fig.add_subplot(outer[1],sharex=a)
    for col,label,lstyle in [("left_ear_raw","L raw","-"),("right_ear_raw","R raw","-"),
                  ("left_ear","L smooth","-"),("right_ear","R smooth","-"),
                  ("left_threshold","L threshold","--"),("right_threshold","R threshold","--")]:
        b.plot(q.rel_s,q[col],lstyle,lw=.8,label=label,alpha=.75)
    # Same webcam color as panel 1, not an implicit default cycle color.
    for start,end,*_ in pred:
        b.axvspan(start/1000,end/1000,facecolor=WEBCAM,alpha=.12,zorder=0)
    b.set_title("2. Left/right EAR + adaptive thresholds (webcam intervals in blue)",loc="left",fontsize=11,fontweight="bold")
    b.set_ylabel("EAR");b.legend(ncol=6,fontsize=7);b.grid(alpha=.2)

    c=fig.add_subplot(outer[2],sharex=a)
    states=[("Final invalid",~old.as_bool(q["valid_eye_frame"])),
            ("L raw closed",old.as_bool(q["left_closed_raw"])),
            ("R raw closed",old.as_bool(q["right_closed_raw"])),
            ("L smooth closed",old.as_bool(q["left_closed_smooth"])),
            ("R smooth closed",old.as_bool(q["right_closed_smooth"]))]
    for i,(name,condition) in enumerate(states):
        c.broken_barh(old.true_regions(q,condition),(4-i-.34,.68),facecolors=WEBCAM,alpha=.70)
    c.set_yticks(list(range(5)));c.set_yticklabels([s[0] for s in states[::-1]],fontsize=8)
    c.set_ylim(-.6,4.6);c.set_xlabel("Time from fixation start (s)")
    c.set_title("3. Eye-closure decisions / invalid frames",loc="left",fontsize=11,fontweight="bold")
    c.grid(axis="x",alpha=.2)

    title=fig.add_subplot(outer[3]);title.axis("off")
    title.text(0,.5,"4. Eye ROIs — first 4 events by time within each category",fontsize=11,weight="bold")
    sg=outer[4].subgridspec(3,4,wspace=.16,hspace=.56)
    import mediapipe as mp
    mesh=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,
                              min_detection_confidence=.5,min_tracking_confidence=.5)
    for i,category in enumerate(("EYELINK ONLY","WEBCAM ONLY","BOTH")):
        for k in range(4):
            ax=fig.add_subplot(sg[i,k]);ax.axis("off")
            if k>=len(reps[category]):
                if k==0: ax.text(.5,.5,f"{category}\nNo events",ha="center",va="center",transform=ax.transAxes)
                continue
            tag,pts_s,t,reason,frame,le,re,selection=reps[category][k]
            im=old.extract_frame_at_pts(video,pts_s)
            if im is None:
                ax.text(.5,.5,"Frame unavailable",ha="center",va="center",transform=ax.transAxes)
                continue
            crop,points=old.eye_roi(mesh,im)
            ax.imshow(crop)
            label=f"{category} | t={t:.2f}s | frame {frame}\n{selection}: L={le:.3f} R={re:.3f}"
            if points is not None: label+=f"\nROI EAR L={old.ear(points[0]):.3f} R={old.ear(points[1]):.3f}"
            ax.set_title(label,fontsize=8,color=WEBCAM if category=="WEBCAM ONLY" else
                          EYELINK if category=="EYELINK ONLY" else "#38634A")
    mesh.close()
    xmax=max(9.5,float(q.rel_s.max()) if len(q) else 9.5)
    for ax in (a,b,c):ax.set_xlim(0,xmax)
    fig.suptitle(f"{vp} — Trial {tr} — Blink diagnostic (synchronized timing)",weight="bold",fontsize=15,y=.993)
    fig.subplots_adjust(top=.962,left=.075,right=.985,bottom=.035)
    pdf.savefig(fig,bbox_inches="tight");plt.close(fig)


def render_run(run_dir, video, vp, method='ptgaze'):
    from blink.run_data import load_validated_gaze
    from blink.diagnostic_statistics import (synchronized_frames, predicted_intervals,
                                            metrics_for_trial, save_statistics)
    from fixation.generate_fixation_timeline_pdf import load_trials
    run = Path(run_dir)
    extra = run / 'extra'
    filename = ('debug_5_ptgaze_calibrated.csv' if method == 'ptgaze'
                else 'debug_5_pupil_data_calibrated.csv')
    gaze = load_validated_gaze(run, filename)
    frames = synchronized_frames(gaze, pd.read_csv(extra / 'blink_filter_frame_log.csv'))
    regions = pd.read_csv(extra / 'blink_filter_regions.csv')
    eye = pd.read_csv(extra / 'eyelink_blink_benchmark.csv')
    trials = load_trials(run / 'phases_detected.json')
    if not trials:
        raise ValueError('No experimental trials available for blink PDF')
    metrics, records = [], []
    pdf_path = extra / 'blink_diagnostics.pdf'
    with PdfPages(pdf_path) as pdf:
        for tr in trials:
            start, end = tr.trial_start_eye_ms, tr.trial_end_eye_ms
            q = frames[frames['timestamp_ms_synced'].between(start, end, inclusive='left')].copy()
            q['rel_s'] = (q['timestamp_ms_synced'] - start)/1000.0
            pred = predicted_intervals(q, start, end, regions)
            truth = old.truth_events(eye, tr.trial_number)
            row, matches = metrics_for_trial(vp, tr.trial_number, pred, truth, old.match_events)
            metrics.append(row)
            records.extend(matches)
            method_gaze = {}
            for label, name in [('MediaPipe', 'debug_5_pupil_data_calibrated.csv'),
                                ('PTGaze', 'debug_5_ptgaze_calibrated.csv')]:
                if (run / name).exists():
                    method_gaze[label] = _read_gaze(run / name, start)
            method_gaze['EyeLink'] = _read_gaze(run / 'debug_3_eyetracker_data.csv', start, eyelink=True)
            reps = _reps(q, pred, truth, start)
            _trial_page(pdf, vp, tr.trial_number, q, pred, truth, reps, video, method_gaze)
    save_statistics(extra, vp, metrics, records)
    return pdf_path
