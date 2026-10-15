"""Enhanced Blink Diagnostics (stage 1).

Place in IDP_Code/blink_extra/ alongside generate_blink_diagnostics.py.
Reads existing 60-Hz analysis outputs; does not recompute blink or gaze.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mediapipe as mp
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

import generate_blink_diagnostics as old

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
    fig.suptitle(f"{vp} — Trial {tr} — Blink diagnostic (PTS-based webcam timing)",weight="bold",fontsize=15,y=.993)
    fig.subplots_adjust(top=.962,left=.075,right=.985,bottom=.035)
    pdf.savefig(fig,bbox_inches="tight");plt.close(fig)


REQUIRED_GAZE_FILES = (
    "phases_detected.json",
    "debug_5_pupil_data_calibrated.csv",
    "debug_5_ptgaze_calibrated.csv",
    "debug_3_eyetracker_data.csv",
)


def find_complete_gaze_run(vp_dir: Path) -> Path:
    """Choose a complete source run, rather than the newest incomplete run.

    Derived experimental runs are excluded to preserve the original gaze timeline.
    """
    runs = sorted((vp_dir / "Analyse").glob("Run_60hz_FullCalib_mode2_*"),
                  key=lambda p: p.name, reverse=True)
    complete = [p for p in runs if p.is_dir()
                and not any(tag in p.name.lower() for tag in ("blinkfiltered", "_pts"))
                and all((p / name).is_file() for name in REQUIRED_GAZE_FILES)]
    if not complete:
        raise FileNotFoundError(
            f"No complete original FullCalib mode2 run for {vp_dir.name}. "
            f"Required files: {', '.join(REQUIRED_GAZE_FILES)}"
        )
    print(f"[SOURCE] {vp_dir.name}: {complete[0].name}")
    return complete[0]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-root",type=Path,default=Path("/Volumes/Empra10/Ergebnisse60"))
    parser.add_argument("--video-root",type=Path,default=Path("/Volumes/Empra9/Videos_60"))
    parser.add_argument("--output-root",type=Path,default=None)
    parser.add_argument("--vp",action="append",default=None)
    parser.add_argument("--trial",action="append",type=int,default=None)
    parser.add_argument("--blink-only",action="store_true")
    args=parser.parse_args()
    output_root=args.output_root or args.results_root/"blink_extra_outputs"/"diagnostics"
    output_root.mkdir(parents=True,exist_ok=True)
    vps=args.vp or old.DEFAULT_VPS
    groups={"blink"} if args.blink_only else {"blink","suspected","invalid"}
    for vp in vps:
        vd=args.results_root/vp
        run=find_complete_gaze_run(vd)
        test=vd/"test60"
        fs={"frame":test/"blink_filter_frame_log.csv",
            "regions":test/"blink_filter_regions.csv",
            "eye":test/"eyelink_blink_benchmark.csv",
            "phases":run/"phases_detected.json"}
        for name,path in fs.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name}: {path}")
        frames=pd.read_csv(fs["frame"])
        regions=pd.read_csv(fs["regions"])
        eye=pd.read_csv(fs["eye"])
        phases=json.loads(fs["phases"].read_text())
        trials=old.trials_from_json(fs["phases"])
        video=args.video_root/f"{vp}.mp4"
        if not video.exists():raise FileNotFoundError(video)
        required=["debug_5_pupil_data_calibrated.csv","debug_5_ptgaze_calibrated.csv","debug_3_eyetracker_data.csv"]
        for name in required:
            if not (run/name).exists(): raise FileNotFoundError(run/name)
        pdfpath=output_root/f"{vp}_blink_diagnostics.pdf"
        with PdfPages(pdfpath) as pdf:
            for tr in args.trial or sorted(trials):
                if tr not in trials:continue
                z=trials[tr]
                phase_trials=(phases["phases"].get("experiment_block1",{}).get("trials",[])+
                              phases["phases"].get("experiment_block2",{}).get("trials",[]))
                match=next(x for x in phase_trials if not x.get("is_practice",False) and int(x["trial_number"])==tr)
                zero_ms=float(match["fixation"]["start_eyelink_ms"])
                gaze={"MediaPipe":_read_gaze(run/required[0],zero_ms),
                      "PTGaze":_read_gaze(run/required[1],zero_ms),
                      "EyeLink":_read_gaze(run/required[2],zero_ms,eyelink=True)}
                q=old.trial_frames(frames,z["start"],z["end"])
                pred=old.pred_events(regions,z["start"],z["end"],groups)
                truth=old.truth_events(eye,tr)
                reps=_reps(q,pred,truth,z["start"])
                _trial_page(pdf,vp,tr,q,pred,truth,reps,video,gaze)
                print(f"[OK] {vp} trial {tr}")
        print(f"[PDF] {pdfpath}")

if __name__=="__main__":main()
