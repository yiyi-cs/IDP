#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,mediapipe as mp,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R=[33,159,158,133,153,145]; L=[362,380,374,263,386,385]
def ear(p):
 h=np.linalg.norm(p[0]-p[3])
 return float((np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2*h)) if h else np.nan
def win(p,t):
 d=json.load(open(p)); xs=d["phases"].get("experiment_block1",{}).get("trials",[])+d["phases"].get("experiment_block2",{}).get("trials",[])
 x=[x for x in xs if int(x.get("trial_number",-1))==t and not x.get("is_practice",False)][0]
 return float(x["fixation"]["start_video_s"])*1000,float(x["stimulus"]["end_video_s"])*1000
def main():
 p=argparse.ArgumentParser();p.add_argument("--video",type=Path,required=True);p.add_argument("--frame-log",type=Path,required=True);p.add_argument("--phases",type=Path,required=True);p.add_argument("--trial",type=int,default=1);p.add_argument("--output-dir",type=Path,default=Path("ear_consistency"));a=p.parse_args()
 st,en=win(a.phases,a.trial); d=pd.read_csv(a.frame_log); tm=pd.to_numeric(d.video_time_ms,errors="coerce"); d=d[tm.between(st,en)].copy()
 cap=cv2.VideoCapture(str(a.video)); fps=cap.get(cv2.CAP_PROP_FPS); mesh=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=.5,min_tracking_confidence=.5)
 rows=[]
 for q in d.itertuples():
  fi = int(q.frame_number); cap.set(cv2.CAP_PROP_POS_FRAMES,fi); ok,im=cap.read()
  if not ok: continue
  h,w=im.shape[:2]; res=mesh.process(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
  le=re=np.nan
  if res.multi_face_landmarks:
   lm=res.multi_face_landmarks[0]; le=ear(np.array([[lm.landmark[i].x*w,lm.landmark[i].y*h] for i in L])); re=ear(np.array([[lm.landmark[i].x*w,lm.landmark[i].y*h] for i in R]))
  ll=float(q.left_ear_raw) if pd.notna(q.left_ear_raw) else np.nan; lr=float(q.right_ear_raw) if pd.notna(q.right_ear_raw) else np.nan
  rows.append(dict(frame=fi,logged_time_ms=float(q.video_time_ms),frame_time_ms=fi/fps*1000,logged_left=ll,recomputed_left=le,logged_right=lr,recomputed_right=re,logged_avg=np.nanmean([ll,lr]),recomputed_avg=np.nanmean([le,re])))
 cap.release();mesh.close();o=pd.DataFrame(rows);o["time_error_ms"]=o.logged_time_ms-o.frame_time_ms;o["ear_abs_error"]=abs(o.logged_avg-o.recomputed_avg);a.output_dir.mkdir(parents=True,exist_ok=True);csv=a.output_dir/f"trial_{a.trial:02d}_ear_frame_consistency.csv";o.to_csv(csv,index=False)
 fig,ax=plt.subplots(figsize=(13,5));x=(o.logged_time_ms-st)/1000;ax.plot(x,o.logged_avg,label="Logged raw EAR",linewidth=1);ax.plot(x,o.recomputed_avg,label="Recomputed from same frame",linewidth=1,alpha=.8);ax.set_xlabel("Time from fixation start (s)");ax.set_ylabel("Average EAR");ax.grid(True,linewidth=.35);ax.legend();fig.tight_layout();pdf=a.output_dir/f"trial_{a.trial:02d}_ear_frame_consistency.pdf";fig.savefig(pdf);plt.close(fig)
 print("Video FPS:",fps);print("Rows:",len(o));print("Median time error ms:",o.time_error_ms.median());print("Time error range ms:",o.time_error_ms.min(),o.time_error_ms.max());print("Median EAR abs error:",o.ear_abs_error.median());print("Max EAR abs error:",o.ear_abs_error.max());print("PDF:",pdf);print("CSV:",csv)
if __name__=="__main__":main()
