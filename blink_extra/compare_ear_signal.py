#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def window(p,t):
 d=json.load(open(p)); xs=d["phases"].get("experiment_block1",{}).get("trials",[])+d["phases"].get("experiment_block2",{}).get("trials",[])
 x=[x for x in xs if int(x.get("trial_number",-1))==t and not x.get("is_practice",False)][0]
 return float(x["fixation"]["start_video_s"])*1000,float(x["stimulus"]["end_video_s"])*1000

def log(p,a,b):
 d=pd.read_csv(p); tm=pd.to_numeric(d.video_time_ms,errors="coerce"); d=d[tm.between(a,b)].copy(); d["time_s"]=(pd.to_numeric(d.video_time_ms)-a)/1000
 for c in ["left_ear_raw","right_ear_raw","left_ear","right_ear","left_threshold","right_threshold"]: d[c]=pd.to_numeric(d[c],errors="coerce")
 d["raw"]=d[["left_ear_raw","right_ear_raw"]].mean(axis=1); d["smooth"]=d[["left_ear","right_ear"]].mean(axis=1); d["threshold"]=d[["left_threshold","right_threshold"]].mean(axis=1)
 return d

def events(p,t):
 d=pd.read_csv(p); d=d[pd.to_numeric(d.trial,errors="coerce").eq(t)]
 return [(float(a)/1000,float(b)/1000,float(z)) for a,b,z in zip(d.start_from_fixation_ms,d.end_from_fixation_ms,d.duration_ms)]

def main():
 p=argparse.ArgumentParser()
 for x in ["raw25-log","raw60-log","lanczos60-log","raw25-phases","raw60-phases","lanczos60-phases","eyelink"]: p.add_argument("--"+x,type=Path,required=True)
 p.add_argument("--trial",type=int,default=1); p.add_argument("--output-dir",type=Path,default=Path("ear_signal_comparison")); p.add_argument("--event-window-ms",type=float,default=250)
 a=p.parse_args(); specs=[("25 Hz Raw",a.raw25_log,a.raw25_phases),("60 Hz Raw",a.raw60_log,a.raw60_phases),("60 Hz Lanczos 2x",a.lanczos60_log,a.lanczos60_phases)]
 ds=[]
 for n,l,ph in specs:
  x,y=window(ph,a.trial); ds.append((n,log(l,x,y)))
 es=events(a.eyelink,a.trial); a.output_dir.mkdir(parents=True,exist_ok=True)
 fig,axs=plt.subplots(3,1,figsize=(13,10),sharex=True)
 for ax,(n,d) in zip(axs,ds):
  ax.plot(d.time_s,d.raw,label="EAR raw",linewidth=.8,alpha=.65); ax.plot(d.time_s,d.smooth,label="EAR smoothed",linewidth=1.1); ax.plot(d.time_s,d.threshold,label="Adaptive threshold",linewidth=.9,linestyle="--")
  for x,y,_ in es: ax.axvspan(x,y,alpha=.13)
  bad=d[d.valid_eye_frame.astype(str).str.lower().eq("false")]; ax.scatter(bad.time_s,bad.smooth,s=10,label="Invalid frame")
  ax.set_ylabel("EAR"); ax.set_title(n,loc="left"); ax.grid(True,linewidth=.35); ax.legend(fontsize=8,ncol=4)
 axs[-1].set_xlabel("Time from fixation start (s)"); fig.suptitle(f"EAR signal comparison — Trial {a.trial} — EyeLink EBLINK shaded"); fig.tight_layout()
 pdf=a.output_dir/f"trial_{a.trial:02d}_ear_signal_comparison.pdf"; fig.savefig(pdf); plt.close(fig)
 rows=[]
 for n,d in ds:
  for i,(x,y,dur) in enumerate(es,1):
   c=(x+y)/2; q=d[d.time_s.between(c-a.event_window_ms/1000,c+a.event_window_ms/1000)]
   rows.append(dict(source=n,event=i,start_s=x,duration_ms=dur,min_raw=q.raw.min(),min_smooth=q.smooth.min(),median_threshold=q.threshold.median(),raw_below_threshold=bool((q.raw<q.threshold).any()),smooth_below_threshold=bool((q.smooth<q.threshold).any()),invalid_frames=int(q.valid_eye_frame.astype(str).str.lower().eq("false").sum())))
 out=pd.DataFrame(rows); csv=a.output_dir/f"trial_{a.trial:02d}_ear_event_diagnostics.csv"; out.to_csv(csv,index=False)
 print("PDF:",pdf); print("CSV:",csv); print(out.to_string(index=False))
if __name__=="__main__": main()
