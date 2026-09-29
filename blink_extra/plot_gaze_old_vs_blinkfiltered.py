#!/usr/bin/env python3
"""OLD vs BlinkFiltered MediaPipe gaze, with unchanged original sync/timing."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr

W,CM,DIST=1920.,53.2,70.; CENTER=(W-1)/2

def eye_deg(x):
    cm=(pd.to_numeric(x,errors="coerce")-CENTER)*(CM/W)
    return np.degrees(np.arctan2(cm,DIST))

def trial(ph,n):
    for bk in ["experiment_block1","experiment_block2"]:
        for t in ph.get("phases",{}).get(bk,{}).get("trials",[]):
            if int(t["trial_number"])==n and not t.get("is_practice",False): return t
    raise ValueError(f"Trial {n} not found")

def load_mp(path,n):
    d=pd.read_csv(path)
    d=d[pd.to_numeric(d["trial_number"],errors="coerce").eq(n)].copy()
    d["t"]=pd.to_numeric(d["timestamp_ms_synced"],errors="coerce")
    d["g"]=pd.to_numeric(d["gaze_deg_x_calib"],errors="coerce")
    return d.sort_values("t")

def load_et(path,start,end):
    d=pd.read_csv(path,usecols=lambda c:c in {"timestamp_ms","x_pos"})
    d["t"]=pd.to_numeric(d["timestamp_ms"],errors="coerce"); d["g"]=eye_deg(d["x_pos"])
    d=d[d["t"].between(start,end)].dropna(subset=["t"])
    return d.groupby("t",as_index=False)["g"].mean().sort_values("t")

def validmask(d):
    if "valid_eye_frame" not in d:return pd.Series(True,index=d.index)
    return d["valid_eye_frame"].astype(str).str.strip().str.lower().isin(["true","1","1.0","yes"])

def interp_et(e,t):
    e=e.dropna(subset=["t","g"])
    if len(e)<2:return np.full(len(t),np.nan)
    x=e["t"].to_numpy(float); y=e["g"].to_numpy(float); q=np.asarray(t,float)
    z=np.interp(q,x,y); z[(q<x[0])|(q>x[-1])]=np.nan
    return z

def metric(d,e,filtered):
    q=d[d["phase_type"].eq("stimulus")].copy() if "phase_type" in d else d.copy()
    if filtered:q=q[validmask(q)]
    q=q.dropna(subset=["t","g"]); b=interp_et(e,q["t"].to_numpy()); a=q["g"].to_numpy()
    ok=np.isfinite(a)&np.isfinite(b); a=a[ok];b=b[ok]
    return {"n":len(a),"r":pearsonr(a,b)[0] if len(a)>2 else np.nan,
            "mae":np.mean(np.abs(a-b)) if len(a) else np.nan,
            "rmse":np.sqrt(np.mean((a-b)**2)) if len(a) else np.nan}

def invalid_spans(d,zero):
    q=d[~validmask(d)].sort_values("t")
    if q.empty:return []
    # split using original camera timeline; >80 ms safely denotes a gap between invalid runs
    groups=q["t"].diff().gt(80).cumsum()
    return [((g["t"].min()-zero)/1000,(g["t"].max()-zero)/1000) for _,g in q.groupby(groups)]

def draw(ax,d,e,zero,title,filtered):
    q=d.copy()
    if filtered:q.loc[~validmask(q),"g"]=np.nan
    ax.plot((e["t"]-zero)/1000,e["g"],label="EyeLink",color="black",linewidth=.9)
    ax.plot((q["t"]-zero)/1000,q["g"],label="MediaPipe",color="tab:red",linewidth=.9)
    if filtered:
        for x,y in invalid_spans(d,zero):
            ax.axvspan(x,y,color="tab:blue",alpha=.18)
    ax.set_title(title,loc="left",fontweight="bold")
    ax.set_ylabel("Horizontal gaze angle (°)");ax.grid(True,alpha=.25);ax.legend(fontsize=8)
    ax.set_xlim(0,10)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--vp",default="beo7");p.add_argument("--trial",type=int,default=1)
    p.add_argument("--old-run",type=Path,required=True);p.add_argument("--new-run",type=Path)
    a=p.parse_args();old=a.old_run.resolve()
    new=a.new_run.resolve() if a.new_run else Path(str(old)+"_BlinkFiltered")
    ph=json.load(open(old/"phases_detected.json",encoding="utf-8"));tr=trial(ph,a.trial)
    zero=float(tr["fixation"]["start_eyelink_ms"]); end=float(tr["stimulus"]["end_eyelink_ms"])
    o=load_mp(old/"debug_5_pupil_data_calibrated.csv",a.trial)
    n=load_mp(new/"debug_5_pupil_data_calibrated.csv",a.trial)

    # Critical guard: OLD and NEW must have identical frame/timing/gaze before filtering.
    chk=o[["frame","timestamp_ms_synced","gaze_deg_x_calib"]].merge(
        n[["frame","timestamp_ms_synced","gaze_deg_x_calib"]],on="frame",suffixes=("_old","_new"))
    if len(chk)!=len(o) or len(chk)!=len(n):raise RuntimeError("OLD/NEW frame sets differ")
    for c in ["timestamp_ms_synced","gaze_deg_x_calib"]:
        a1=pd.to_numeric(chk[c+"_old"],errors="coerce").to_numpy()
        a2=pd.to_numeric(chk[c+"_new"],errors="coerce").to_numpy()
        if not np.allclose(a1,a2,equal_nan=True,rtol=0,atol=1e-10):
            raise RuntimeError(f"OLD/NEW {c} differs")

    e=load_et(old/"debug_3_eyetracker_data.csv",zero,end)
    mo=metric(o,e,False); mn=metric(n,e,True)
    out=new/"gaze_old_vs_blinkfiltered";out.mkdir(exist_ok=True)
    pdf=out/f"{a.vp}_trial_{a.trial:02d}_old_vs_blinkfiltered_gaze.pdf"
    with PdfPages(pdf) as pp:
        f,ax=plt.subplots(2,1,figsize=(14,8),sharex=True)
        draw(ax[0],o,e,zero,f"A — OLD | n={mo['n']}  r={mo['r']:.3f}  MAE={mo['mae']:.2f}°  RMSE={mo['rmse']:.2f}°",False)
        draw(ax[1],n,e,zero,f"B — BlinkFiltered | n={mn['n']}  r={mn['r']:.3f}  MAE={mn['mae']:.2f}°  RMSE={mn['rmse']:.2f}°",True)
        ax[1].set_xlabel("Time from fixation start (s)")
        f.suptitle(f"{a.vp} — Trial {a.trial}: MediaPipe vs EyeLink gaze",fontweight="bold")
        f.tight_layout();pp.savefig(f,bbox_inches="tight");plt.close(f)
    rows=[{"condition":"OLD",**mo},{"condition":"BlinkFiltered",**mn}]
    pd.DataFrame(rows).to_csv(out/f"{a.vp}_trial_{a.trial:02d}_old_vs_blinkfiltered_metrics.csv",index=False)
    print("OLD/NEW timing + gaze guard: PASSED")
    print(f"OLD:      n={mo['n']} r={mo['r']:.4f} MAE={mo['mae']:.4f} RMSE={mo['rmse']:.4f}")
    print(f"FILTERED: n={mn['n']} r={mn['r']:.4f} MAE={mn['mae']:.4f} RMSE={mn['rmse']:.4f}")
    print("PDF:",pdf)
if __name__=="__main__":main()
