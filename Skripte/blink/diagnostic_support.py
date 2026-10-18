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

def truth_events(df,trial):
    x=df[pd.to_numeric(df["trial"],errors="coerce").eq(trial)]
    a=pd.to_numeric(x["start_from_fixation_ms"],errors="coerce")
    b=pd.to_numeric(x["end_from_fixation_ms"],errors="coerce")
    return [(float(i),float(j)) for i,j in zip(a,b) if np.isfinite(i) and np.isfinite(j)]

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

def as_bool(s): return s.astype(str).str.lower().isin(["true","1","1.0"])

def lowest_row(q,start_abs,end_abs):
    z=q[pd.to_numeric(q["timestamp_ms_synced"],errors="coerce").between(start_abs,end_abs)].copy()
    if z.empty:return None
    # Representative image must correspond to the raw EAR valley, not smoothed EAR.
    z["_raw_avg"]=z[["left_ear_raw","right_ear_raw"]].mean(axis=1)
    v=pd.to_numeric(z["_raw_avg"],errors="coerce")
    return z.loc[v.idxmin()] if v.notna().any() else z.iloc[len(z)//2]

def nearest_row(q, target_abs_ms):
    if q.empty:
        return None
    t = pd.to_numeric(q["timestamp_ms_synced"], errors="coerce")
    valid = t.notna()
    if not valid.any():
        return None
    idx = (t[valid] - float(target_abs_ms)).abs().idxmin()
    return q.loc[idx]


def extract_frame_at_pts(video: Path, pts_s: float):
    """Decode one frame at a real video timestamp; avoids OpenCV frame-index seek on VFR."""
    cmd=[
        "ffmpeg","-v","error","-ss",f"{max(0.0,pts_s):.6f}",
        "-i",str(video),"-frames:v","1","-f","image2pipe","-vcodec","png","-"
    ]
    r=subprocess.run(cmd,capture_output=True)
    if r.returncode!=0 or not r.stdout:return None
    import cv2
    arr=np.frombuffer(r.stdout,dtype=np.uint8)
    return cv2.imdecode(arr,cv2.IMREAD_COLOR)

def ear(p):
    h=np.linalg.norm(p[0]-p[3])
    return float((np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2*h)) if h else np.nan

def eye_roi(mesh,frame):
    import cv2
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

        # Synchronized sample end; PTS is reserved for ROI image extraction.
        if "sample_end_ms" in g.columns and pd.notna(last["sample_end_ms"]):
            trial_start_ms=float(first["timestamp_ms_synced"])-float(first["rel_s"])*1000.0
            end_s=(float(last["sample_end_ms"])-trial_start_ms)/1000.0
        else:
            times=pd.to_numeric(q["rel_s"],errors="coerce").dropna().to_numpy()
            dt=float(np.nanmedian(np.diff(times))) if len(times)>1 else .02
            end_s=float(last["rel_s"])+dt

        regions.append((start_s,max(end_s-start_s,.001)))

    return regions
