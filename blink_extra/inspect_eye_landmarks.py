#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import cv2, mediapipe as mp, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RIGHT=[33,159,158,133,153,145]
LEFT=[362,380,374,263,386,385]

def ear(pts):
 v1=np.linalg.norm(pts[1]-pts[5]); v2=np.linalg.norm(pts[2]-pts[4]); h=np.linalg.norm(pts[0]-pts[3])
 return float((v1+v2)/(2*h)) if h else 0.0

def trial_start(phases,trial):
 d=json.load(open(phases))
 xs=d["phases"].get("experiment_block1",{}).get("trials",[])+d["phases"].get("experiment_block2",{}).get("trials",[])
 x=[x for x in xs if int(x.get("trial_number",-1))==trial and not x.get("is_practice",False)][0]
 return float(x["fixation"]["start_video_s"])

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--video",type=Path,required=True); p.add_argument("--phases",type=Path,required=True)
 p.add_argument("--trial",type=int,default=1); p.add_argument("--times",type=float,nargs="+",default=[1,4,8,3,5.8])
 p.add_argument("--half-window",type=float,default=.5); p.add_argument("--sample-step",type=int,default=3)
 p.add_argument("--output",type=Path,default=Path("eye_landmark_diagnostic.pdf"))
 a=p.parse_args(); zero=trial_start(a.phases,a.trial)
 cap=cv2.VideoCapture(str(a.video)); fps=cap.get(cv2.CAP_PROP_FPS)
 mesh=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=.5,min_tracking_confidence=.5)
 with PdfPages(a.output) as pdf:
  for target in a.times:
   start=zero+target-a.half_window; end=zero+target+a.half_window
   cap.set(cv2.CAP_PROP_POS_MSEC,start*1000); rows=[]; i=0
   while True:
    ok,frame=cap.read()
    if not ok: break
    t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
    if t>end: break
    if i%a.sample_step: i+=1; continue
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); res=mesh.process(rgb); h,w=frame.shape[:2]
    if not res.multi_face_landmarks: i+=1; continue
    lm=res.multi_face_landmarks[0]
    lp=np.array([[lm.landmark[j].x*w,lm.landmark[j].y*h] for j in LEFT])
    rp=np.array([[lm.landmark[j].x*w,lm.landmark[j].y*h] for j in RIGHT])
    allp=np.vstack([lp,rp]); x0=max(0,int(allp[:,0].min()-50)); x1=min(w,int(allp[:,0].max()+50)); y0=max(0,int(allp[:,1].min()-40)); y1=min(h,int(allp[:,1].max()+40))
    crop=rgb[y0:y1,x0:x1].copy()
    for pts in (lp,rp):
     for k,(x,y) in enumerate(pts):
      cv2.circle(crop,(int(x-x0),int(y-y0)),5,(255,0,0),-1); cv2.putText(crop,str(k),(int(x-x0)+5,int(y-y0)-5),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,0),1)
    rows.append((t-zero,crop,ear(lp),ear(rp))); i+=1
   if not rows: continue
   cols=4; pagesize=12
   for off in range(0,len(rows),pagesize):
    chunk=rows[off:off+pagesize]; nr=math.ceil(len(chunk)/cols)
    fig,axs=plt.subplots(nr,cols,figsize=(16,3.3*nr),squeeze=False)
    for ax in axs.flat: ax.axis("off")
    for ax,(tt,img,le,re) in zip(axs.flat,chunk):
     ax.imshow(img); ax.axis("off"); ax.set_title(f"t={tt:.3f}s | L={le:.3f} R={re:.3f} avg={(le+re)/2:.3f}",fontsize=9)
    fig.suptitle(f"Trial {a.trial} — target {target:.1f}s ±{a.half_window}s — every {a.sample_step} frames",fontsize=13)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
 cap.release(); mesh.close()
 print("Saved:",a.output)
if __name__=="__main__": main()
