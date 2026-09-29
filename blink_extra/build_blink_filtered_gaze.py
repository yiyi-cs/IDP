#!/usr/bin/env python3
"""
Build a BlinkFiltered gaze run without changing gaze values or timing.

Only change:
    eyes_closed = NOT valid_eye_frame
using frame == frame_number to merge the new test60 blink result.

All original gaze timestamps, synchronized timestamps, trial assignments,
phase assignments, calibrated gaze values, and calibration fields remain intact.
"""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
import pandas as pd

VPS=["beo7","bjs4","egf5","fbn6","fgt6","jkl7","kdn8","kro3","mhe9","oem4","ogt7"]

def as_bool(s):
    if pd.api.types.is_bool_dtype(s): return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin(["true","1","1.0","yes"])

def choose_old_run(vp_dir, explicit=None):
    if explicit:
        p=explicit.resolve()
        if not p.is_dir(): raise FileNotFoundError(p)
        return p
    xs=[p for p in (vp_dir/"Analyse").glob("Run_60hz_FullCalib_mode2_*")
        if "_BlinkFiltered" not in p.name and "_PTS_BlinkFiltered" not in p.name
        and (p/"debug_5_pupil_data_calibrated.csv").is_file()
        and (p/"debug_6_three_way_metrics.json").is_file()]
    if not xs: raise FileNotFoundError(f"No evaluated OLD 60 Hz run under {vp_dir/'Analyse'}")
    # Prefer original January baseline if present.
    jan=[p for p in xs if "20260119_" in p.name]
    return max(jan or xs,key=lambda p:p.stat().st_mtime)

def process(vp,root,explicit=None):
    vd=root/vp; old=choose_old_run(vd,explicit)
    gaze_path=old/"debug_5_pupil_data_calibrated.csv"
    blink_path=vd/"test60"/"blink_filter_frame_log.csv"
    if not blink_path.is_file(): raise FileNotFoundError(blink_path)

    g=pd.read_csv(gaze_path); b=pd.read_csv(blink_path)
    if "frame" not in g: raise KeyError("OLD debug_5 missing frame")
    for c in ["frame_number","valid_eye_frame"]:
        if c not in b: raise KeyError(f"blink frame log missing {c}")

    g["frame"]=pd.to_numeric(g["frame"],errors="raise").astype(int)
    b["frame_number"]=pd.to_numeric(b["frame_number"],errors="raise").astype(int)
    if g["frame"].duplicated().any(): raise ValueError("duplicate frame in OLD gaze")
    if b["frame_number"].duplicated().any(): raise ValueError("duplicate frame_number in blink log")

    # Keep useful diagnostic columns, but do not overwrite timing/gaze fields.
    cols=["frame_number","valid_eye_frame","invalid_reason","reason_group","is_blink",
          "is_suspected_blink","is_transition_frame","left_ear_raw","right_ear_raw",
          "left_ear","right_ear","avg_ear","left_threshold","right_threshold"]
    cols=[c for c in cols if c in b]

    # Replace only detector-related columns if they already existed.
    repl=[c for c in cols if c!="frame_number"]
    x=g.drop(columns=[c for c in repl if c in g.columns],errors="ignore").merge(
        b[cols],left_on="frame",right_on="frame_number",
        how="left",validate="one_to_one",indicator=True)

    miss=x["_merge"].ne("both")
    if miss.any():
        raise ValueError(f"{miss.sum()} gaze rows lack blink match; first frames: "
                         f"{x.loc[miss,'frame'].head(10).tolist()}")
    x=x.drop(columns=["_merge","frame_number"])

    valid=as_bool(x["valid_eye_frame"])
    x["valid_eye_frame"]=valid
    # Intentional compatibility mapping for the unchanged debug_6 evaluator:
    x["eyes_closed"]=~valid
    if "is_blink" in x: x["is_blink"]=as_bool(x["is_blink"])

    # Hard guard: timing/gaze must be bit-for-bit/numerically unchanged after merge.
    orig=g.set_index("frame").loc[x["frame"]]
    for c in ["timestamp_ms","timestamp_ms_synced","trial_number","phase_type",
              "gaze_deg_x_calib","gaze_deg_y_calib"]:
        if c in g.columns and c in x.columns:
            a=orig[c].reset_index(drop=True)
            z=x[c].reset_index(drop=True)
            if pd.api.types.is_numeric_dtype(a):
                if not a.equals(z) and not ((a-z).abs().fillna(0)<1e-12).all():
                    raise RuntimeError(f"Guard failed: {c} changed")
            else:
                if not a.fillna("").astype(str).equals(z.fillna("").astype(str)):
                    raise RuntimeError(f"Guard failed: {c} changed")

    new=vd/"Analyse"/f"{old.name}_BlinkFiltered"
    new.mkdir(parents=True,exist_ok=True)

    # Inputs needed by original debug_6.
    required=["debug_5_ptgaze_calibrated.csv","debug_3_eyetracker_data.csv","phases_detected.json"]
    optional=["debug_5_calibration_info.json","debug_3_blocks_overview.csv"]
    for name in required+optional:
        s=old/name
        if s.is_file(): shutil.copy2(s,new/name)
    missing=[n for n in required if not (new/n).is_file()]
    if missing: raise FileNotFoundError("OLD run missing debug_6 inputs: "+", ".join(missing))

    out=new/"debug_5_pupil_data_calibrated.csv"; x.to_csv(out,index=False)
    info={"vp":vp,"old_run":str(old),"new_run":str(new),"rows":len(x),
          "valid_eye":int(valid.sum()),"invalid_eye":int((~valid).sum()),
          "changed_fields":["eyes_closed","valid_eye_frame","invalid_reason","reason_group",
                            "is_blink","is_suspected_blink","is_transition_frame","EAR diagnostics"],
          "unchanged_fields":["timestamp_ms","timestamp_ms_synced","trial_number","phase_type",
                              "gaze_deg_x_calib","gaze_deg_y_calib"],
          "mapping":"eyes_closed := NOT valid_eye_frame"}
    with open(new/"blinkfiltered_build_info.json","w",encoding="utf-8") as f:
        json.dump(info,f,indent=2,ensure_ascii=False)

    print(f"\\n[OK] {vp}")
    print(f" OLD: {old}")
    print(f" NEW: {new}")
    print(f" rows={len(x)}, valid={valid.sum()}, invalid={(~valid).sum()}")
    print(" timing/gaze guard: PASSED")
    return new

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results-root",type=Path,default=Path("/Volumes/Empra10/Ergebnisse60"))
    p.add_argument("--vp",action="append")
    p.add_argument("--old-run",type=Path)
    a=p.parse_args();v=a.vp or VPS
    if a.old_run and len(v)!=1:p.error("--old-run requires exactly one --vp")
    ok=bad=0
    for vp in v:
        try: process(vp,a.results_root,a.old_run);ok+=1
        except Exception as e:
            bad+=1;print(f"[ERROR] {vp}: {type(e).__name__}: {e}",file=sys.stderr)
    print(f"\\n=== Build finished ===\\nSuccessful: {ok}\\nFailed:     {bad}")
    if bad:sys.exit(1)
if __name__=="__main__":main()
