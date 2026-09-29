#!/usr/bin/env python3
"""
Generate PTS-corrected Raw 60 Hz blink diagnostics.

Per VP:
  <VP>/blink_diagnostics/<VP>_raw60_blink_diagnostics.pdf
  One page per non-practice trial:
    1) Webcam blink/invalid regions vs EyeLink EBLINK
    2) Left/right raw + smoothed EAR + adaptive thresholds
    3) Raw/smoothed eye-closed states + final invalid state
    4) Representative eye ROIs with MediaPipe EAR landmarks

Global:
  Ergebnisse60/Blink_Diagnostics_60hz/
    blink_diagnostic_metrics.csv
    blink_event_matches.csv
    blink_diagnostic_summary_by_vp.csv
"""
from __future__ import annotations
import argparse, json, sys, subprocess
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

LEFT = [362, 380, 374, 263, 386, 385]
RIGHT = [33, 159, 158, 133, 153, 145]
DEFAULT_VPS = ["beo7","bjs4","egf5","fbn6","fgt6","jkl7","kdn8","kro3","mhe9","oem4","ogt7"]

def latest_run(vpdir: Path) -> Path:
    runs=[p for p in (vpdir/"Analyse").glob("Run_60hz_FullCalib_*") if (p/"phases_detected.json").is_file()]
    if not runs: raise FileNotFoundError(f"No 60 Hz FullCalib run under {vpdir/'Analyse'}")
    return max(runs,key=lambda p:p.stat().st_mtime)

def paths_for(vpdir: Path):
    test=vpdir/"test60"; run=latest_run(vpdir)
    d={"frame":test/"blink_filter_frame_log.csv","regions":test/"blink_filter_regions.csv",
       "eye":test/"eyelink_blink_benchmark.csv","markers":test/"eyelink_trial_markers.csv",
       "phases":run/"phases_detected.json","run":run}
    for k,p in d.items():
        if k!="run" and not p.is_file(): raise FileNotFoundError(p)
    return d

def trials_from_json(path: Path):
    d=json.load(open(path,encoding="utf-8")); p=d["phases"]
    xs=p.get("experiment_block1",{}).get("trials",[])+p.get("experiment_block2",{}).get("trials",[])
    out={}
    for x in xs:
        if x.get("is_practice",False): continue
        f=x["fixation"]; s=x["stimulus"]
        out[int(x["trial_number"])]={
            "start":float(f["start_video_s"])*1000,"end":float(s["end_video_s"])*1000,
            "fix_end":float(f["end_video_s"])*1000,"stim_start":float(s["start_video_s"])*1000}
    return out

def truth_events(df,trial):
    x=df[pd.to_numeric(df["trial"],errors="coerce").eq(trial)]
    a=pd.to_numeric(x["start_from_fixation_ms"],errors="coerce")
    b=pd.to_numeric(x["end_from_fixation_ms"],errors="coerce")
    return [(float(i),float(j)) for i,j in zip(a,b) if np.isfinite(i) and np.isfinite(j)]

def pred_events(df,start,end,groups):
    x=df.copy()
    x["start_video_time_ms"]=pd.to_numeric(x["start_video_time_ms"],errors="coerce")
    x["end_video_time_ms"]=pd.to_numeric(x["end_video_time_ms"],errors="coerce")
    x=x[x["reason_group"].astype(str).isin(groups)&x["end_video_time_ms"].gt(start)&x["start_video_time_ms"].lt(end)]
    out=[]
    for r in x.itertuples():
        a=max(float(r.start_video_time_ms),start)-start
        b=min(float(r.end_video_time_ms),end)-start
        if b>a: out.append((a,b,str(r.reason_group),int(r.start_frame),int(r.end_frame)))
    return out

def match_events(pred,truth,tol):
    c=[]
    for i,p in enumerate(pred):
        for j,t in enumerate(truth):
            if p[0] <= t[1]+tol and p[1] >= t[0]-tol:
                cd=abs((p[0]+p[1]-t[0]-t[1])/2)
                ov=max(0,min(p[1],t[1])-max(p[0],t[0]))
                c.append((cd,-ov,i,j))
    up,ut,m=set(),set(),[]
    for _,_,i,j in sorted(c):
        if i not in up and j not in ut:
            up.add(i);ut.add(j);m.append((i,j))
    return m

def trial_frames(df,start,end):
    t=pd.to_numeric(df["video_time_ms"],errors="coerce")
    q=df[t.between(start,end)].copy()
    q["rel_s"]=(pd.to_numeric(q["video_time_ms"],errors="coerce")-start)/1000
    for c in ["left_ear_raw","right_ear_raw","left_ear","right_ear","avg_ear","left_threshold","right_threshold"]:
        q[c]=pd.to_numeric(q[c],errors="coerce")
    return q

def as_bool(s): return s.astype(str).str.lower().isin(["true","1","1.0"])

def lowest_row(q,start_abs,end_abs):
    z=q[pd.to_numeric(q["video_time_ms"],errors="coerce").between(start_abs,end_abs)].copy()
    if z.empty:return None
    # Representative image must correspond to the raw EAR valley, not smoothed EAR.
    z["_raw_avg"]=z[["left_ear_raw","right_ear_raw"]].mean(axis=1)
    v=pd.to_numeric(z["_raw_avg"],errors="coerce")
    return z.loc[v.idxmin()] if v.notna().any() else z.iloc[len(z)//2]

def nearest_row(q, target_abs_ms):
    if q.empty:
        return None
    t = pd.to_numeric(q["video_time_ms"], errors="coerce")
    valid = t.notna()
    if not valid.any():
        return None
    idx = (t[valid] - float(target_abs_ms)).abs().idxmin()
    return q.loc[idx]


def representatives(q,pred,truth,matches,start,tol,max_n):
    """
    MATCH / WEBCAM ONLY:
        use the minimum raw-EAR frame inside the webcam event.

    EYELINK ONLY:
        use the video frame nearest the exact midpoint of the EyeLink EBLINK
        interval. This deliberately does NOT search for a low-EAR frame.
    """
    pm={i:j for i,j in matches}
    tm={j:i for i,j in matches}
    out=[]

    for i,p in enumerate(pred):
        j=pm.get(i)
        r=lowest_row(q,start+p[0],start+p[1])
        if r is not None:
            out.append((
                "MATCH" if j is not None else "WEBCAM ONLY",
                float(r["video_time_ms"])/1000.0,
                (p[0]+p[1])/2000.0,
                p[2],
                int(r["frame_number"]),
                float(r["left_ear_raw"]),
                float(r["right_ear_raw"]),
                "min raw EAR",
            ))

    for j,t in enumerate(truth):
        if j in tm:
            continue
        midpoint_rel_ms=(t[0]+t[1])/2.0
        r=nearest_row(q,start+midpoint_rel_ms)
        if r is not None:
            out.append((
                "EYELINK ONLY",
                float(r["video_time_ms"])/1000.0,
                midpoint_rel_ms/1000.0,
                "eyelink",
                int(r["frame_number"]),
                float(r["left_ear_raw"]),
                float(r["right_ear_raw"]),
                "EBLINK midpoint",
            ))

    return out[:max_n]


def extract_frame_at_pts(video: Path, pts_s: float):
    """Decode one frame at a real video timestamp; avoids OpenCV frame-index seek on VFR."""
    cmd=[
        "ffmpeg","-v","error","-ss",f"{max(0.0,pts_s):.6f}",
        "-i",str(video),"-frames:v","1","-f","image2pipe","-vcodec","png","-"
    ]
    r=subprocess.run(cmd,capture_output=True)
    if r.returncode!=0 or not r.stdout:return None
    arr=np.frombuffer(r.stdout,dtype=np.uint8)
    return cv2.imdecode(arr,cv2.IMREAD_COLOR)

def ear(p):
    h=np.linalg.norm(p[0]-p[3])
    return float((np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2*h)) if h else np.nan

def eye_roi(mesh,frame):
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); res=mesh.process(rgb)
    if not res.multi_face_landmarks:return rgb,None
    h,w=frame.shape[:2]; lm=res.multi_face_landmarks[0]
    lp=np.array([[lm.landmark[i].x*w,lm.landmark[i].y*h] for i in LEFT])
    rp=np.array([[lm.landmark[i].x*w,lm.landmark[i].y*h] for i in RIGHT])
    p=np.vstack([lp,rp]); x0=max(0,int(p[:,0].min()-55));x1=min(w,int(p[:,0].max()+55));y0=max(0,int(p[:,1].min()-45));y1=min(h,int(p[:,1].max()+45))
    crop=rgb[y0:y1,x0:x1].copy()
    for pts in (lp,rp):
        for k,(x,y) in enumerate(pts):
            cv2.circle(crop,(int(x-x0),int(y-y0)),5,(255,0,0),-1)
            cv2.putText(crop,str(k),(int(x-x0)+4,int(y-y0)-4),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,0),1)
    return crop,(lp,rp)

def true_regions(q, condition):
    """Merge consecutive True frames into real PTS intervals."""
    if q.empty:
        return []
    cond = pd.Series(condition, index=q.index).fillna(False).astype(bool)
    active = q.loc[cond].copy()
    if active.empty:
        return []

    active = active.sort_values("frame_number")
    groups = active["frame_number"].diff().ne(1).cumsum()
    regions = []

    for _, g in active.groupby(groups, sort=False):
        first=g.iloc[0]
        last=g.iloc[-1]
        start_s=float(first["rel_s"])

        # Prefer the PTS-derived frame end written by export_blink_filter_logs.
        if "frame_end_time_ms" in g.columns and pd.notna(last["frame_end_time_ms"]):
            trial_start_ms=float(first["video_time_ms"])-float(first["rel_s"])*1000.0
            end_s=(float(last["frame_end_time_ms"])-trial_start_ms)/1000.0
        else:
            times=pd.to_numeric(q["rel_s"],errors="coerce").dropna().to_numpy()
            dt=float(np.nanmedian(np.diff(times))) if len(times)>1 else .02
            end_s=float(last["rel_s"])+dt

        regions.append((start_s,max(end_s-start_s,.001)))

    return regions


def trial_page(pdf,vp,tr,q,pred,truth,reps,video,tol):
    fig=plt.figure(figsize=(15,13))
    outer=fig.add_gridspec(5,1,height_ratios=[1.0,1.35,1.22,.16,2.35],hspace=.48)

    a=fig.add_subplot(outer[0])
    for x,y,g,*_ in pred:
        a.axvspan(x/1000,y/1000,facecolor="tab:blue",edgecolor="tab:blue",alpha=.34)
    for x,y in truth:
        a.axvspan(x/1000,y/1000,facecolor="tab:orange",edgecolor="tab:orange",
                  alpha=.28,hatch="//",linewidth=.8)
    a.set_yticks([]);a.set_title("1. Raw 60 Hz webcam regions vs EyeLink EBLINK",loc="left")
    a.legend(handles=[
        Patch(facecolor="tab:blue",edgecolor="tab:blue",alpha=.34,label="Webcam"),
        Patch(facecolor="tab:orange",edgecolor="tab:orange",alpha=.28,hatch="//",label="EyeLink")
    ],ncol=2,fontsize=8)
    a.grid(True,axis="x",alpha=.25)

    b=fig.add_subplot(outer[1],sharex=a)
    b.plot(q.rel_s,q.left_ear_raw,label="L raw",lw=.75,alpha=.65)
    b.plot(q.rel_s,q.right_ear_raw,label="R raw",lw=.75,alpha=.65)
    b.plot(q.rel_s,q.left_ear,label="L smooth",lw=1)
    b.plot(q.rel_s,q.right_ear,label="R smooth",lw=1)
    b.plot(q.rel_s,q.left_threshold,"--",label="L threshold",lw=.8)
    b.plot(q.rel_s,q.right_threshold,"--",label="R threshold",lw=.8)
    for x,y in truth:b.axvspan(x/1000,y/1000,alpha=.07)
    b.set_ylabel("EAR");b.set_title("2. Left/right EAR + adaptive thresholds",loc="left")
    b.legend(ncol=6,fontsize=7);b.grid(True,alpha=.25)

    # One band per detector condition. Consecutive True frames are merged.
    c=fig.add_subplot(outer[2],sharex=a)
    state_defs=[
        ("Final invalid frame", ~as_bool(q["valid_eye_frame"])),
        ("Left raw EAR < threshold", as_bool(q["left_closed_raw"])),
        ("Right raw EAR < threshold", as_bool(q["right_closed_raw"])),
        ("Left smoothed EAR < threshold", as_bool(q["left_closed_smooth"])),
        ("Right smoothed EAR < threshold", as_bool(q["right_closed_smooth"])),
    ]
    n=len(state_defs)
    for idx,(label,condition) in enumerate(state_defs):
        y=n-1-idx
        spans=true_regions(q,condition)
        if spans:
            c.broken_barh(spans,(y-.34,.68),alpha=.78)

    c.set_ylim(-.6,n-.4)
    c.set_yticks(range(n))
    c.set_yticklabels([
        "Right smoothed EAR < threshold",
        "Left smoothed EAR < threshold",
        "Right raw EAR < threshold",
        "Left raw EAR < threshold",
        "Final invalid frame",
    ],fontsize=8)
    c.set_title("3. Eye-closure decisions and final invalid frames",loc="left",pad=15)
    c.text(
        0.0,1.015,
        'Filled band = condition is True. "Final invalid" = frame excluded from gaze/pupil analysis.',
        transform=c.transAxes,fontsize=7.2,color="0.35",va="bottom"
    )
    c.grid(True,axis="x",alpha=.25)
    c.set_xlabel("Time from fixation start (s)")

    title_ax=fig.add_subplot(outer[3]);title_ax.axis("off")
    title_ax.text(0,0.5,"4. Representative eye ROI + MediaPipe EAR landmarks",
                  ha="left",va="center",fontsize=10,fontweight="bold")

    sg=outer[4].subgridspec(2,4,wspace=.15,hspace=.38)
    mesh=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,
        min_detection_confidence=.5,min_tracking_confidence=.5)
    for k in range(8):
        ax=fig.add_subplot(sg[k//4,k%4]);ax.axis("off")
        if k>=len(reps):continue
        typ,pts_s,ts,g,fn,logged_l,logged_r,selection=reps[k]
        im=extract_frame_at_pts(video,pts_s)
        if im is None:continue
        crop,pts=eye_roi(mesh,im);ax.imshow(crop)
        recomputed=""
        if pts is not None:
            recomputed=f"\nROI EAR L={ear(pts[0]):.3f} R={ear(pts[1]):.3f}"
        ax.set_title(
            (
                f"{typ} | "
                + (f"EBLINK midpoint t={ts:.2f}s" if typ=="EYELINK ONLY"
                   else f"event t≈{ts:.2f}s")
                + f"\n{selection} frame {fn}: L={logged_l:.3f} R={logged_r:.3f}"
                + recomputed
            ),
            fontsize=7.6
        );ax.axis("off")
    mesh.close()

    xmax=max(9.5,float(q.rel_s.max()) if len(q) else 9.5)
    for ax in (a,b,c):ax.set_xlim(0,xmax)
    fig.suptitle(
        f"{vp} — Trial {tr} — PTS-corrected Raw 60 Hz blink diagnostic | tolerance={tol:.0f} ms",
        fontsize=14,fontweight="bold",y=.995
    )
    fig.subplots_adjust(top=.955,bottom=.035,left=.065,right=.985)
    pdf.savefig(fig,bbox_inches="tight");plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results-root",type=Path,default=Path("/Volumes/Empra10/Ergebnisse60"))
    p.add_argument("--video-root",type=Path,default=Path("/Volumes/Empra9/Videos_60"))
    p.add_argument("--vp",action="append",default=None)
    p.add_argument("--trial",type=int,action="append",default=None)
    p.add_argument("--tolerance-ms",type=float,default=100.0)
    p.add_argument("--blink-only",action="store_true")
    p.add_argument("--max-roi",type=int,default=8)
    args=p.parse_args()
    vps=args.vp or DEFAULT_VPS
    groups={"blink"} if args.blink_only else {"blink","suspected","invalid"}

    for vp in vps:
        try:
            vp_metrics=[];vp_matches=[]
            vd=args.results_root/vp
            fs=paths_for(vd)
            frames=pd.read_csv(fs["frame"])
            regions=pd.read_csv(fs["regions"])
            eye=pd.read_csv(fs["eye"])
            trials=trials_from_json(fs["phases"])
            wanted=args.trial or sorted(trials)
            video=args.video_root/f"{vp}.mp4"
            if not video.is_file():raise FileNotFoundError(video)

            # One folder contains every artifact generated by this script for this VP.
            out=vd/"blink_diagnostics"
            out.mkdir(parents=True,exist_ok=True)
            pdfpath=out/f"{vp}_raw60_blink_diagnostics.pdf"

            with PdfPages(pdfpath) as pdf:
                for tr in wanted:
                    if tr not in trials:continue
                    z=trials[tr]
                    q=trial_frames(frames,z["start"],z["end"])
                    pred=pred_events(regions,z["start"],z["end"],groups)
                    truth=truth_events(eye,tr)
                    mat=match_events(pred,truth,args.tolerance_ms)

                    tp=len(mat);fp=len(pred)-tp;fn=len(truth)-tp
                    pr=tp/(tp+fp) if tp+fp else np.nan
                    rc=tp/(tp+fn) if tp+fn else np.nan
                    f1=2*pr*rc/(pr+rc) if np.isfinite(pr) and np.isfinite(rc) and pr+rc else 0.0
                    vp_metrics.append(dict(
                        vp=vp,trial=tr,tp=tp,fp=fp,fn=fn,precision=pr,recall=rc,f1=f1,
                        n_webcam=len(pred),n_eyelink=len(truth),tolerance_ms=args.tolerance_ms,
                        included_groups=",".join(sorted(groups))
                    ))

                    mm={i:j for i,j in mat}
                    for i,x in enumerate(pred):
                        j=mm.get(i)
                        vp_matches.append(dict(
                            vp=vp,trial=tr,pred_index=i,pred_start_ms=x[0],pred_end_ms=x[1],
                            reason_group=x[2],matched=j is not None,eyelink_index=j,
                            eyelink_start_ms=truth[j][0] if j is not None else np.nan,
                            eyelink_end_ms=truth[j][1] if j is not None else np.nan
                        ))

                    reps=representatives(
                        q,pred,truth,mat,z["start"],args.tolerance_ms,args.max_roi
                    )
                    trial_page(
                        pdf,vp,tr,q,pred,truth,reps,video,args.tolerance_ms
                    )

            metrics_df=pd.DataFrame(vp_metrics)
            matches_df=pd.DataFrame(vp_matches)
            metrics_df.to_csv(out/"blink_diagnostic_metrics.csv",index=False)
            matches_df.to_csv(out/"blink_event_matches.csv",index=False)

            if not metrics_df.empty:
                tp=int(metrics_df.tp.sum());fp=int(metrics_df.fp.sum());fn=int(metrics_df.fn.sum())
                precision=tp/(tp+fp) if tp+fp else np.nan
                recall=tp/(tp+fn) if tp+fn else np.nan
                f1=2*precision*recall/(precision+recall) if np.isfinite(precision) and np.isfinite(recall) and precision+recall else 0.0
                pd.DataFrame([dict(
                    vp=vp,n_trials=len(metrics_df),tp=tp,fp=fp,fn=fn,
                    precision=precision,recall=recall,f1=f1,
                    tolerance_ms=args.tolerance_ms,
                    included_groups=",".join(sorted(groups))
                )]).to_csv(out/"blink_diagnostic_summary.csv",index=False)

            print(f"[OK] {vp}")
            print(f"     PDF:     {pdfpath}")
            print(f"     Metrics: {out/'blink_diagnostic_metrics.csv'}")
            print(f"     Matches: {out/'blink_event_matches.csv'}")
            print(f"     Summary: {out/'blink_diagnostic_summary.csv'}")
        except Exception as e:
            print(f"[ERROR] {vp}: {type(e).__name__}: {e}",file=sys.stderr)

if __name__=="__main__":
    main()
