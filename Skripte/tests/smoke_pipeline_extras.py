"""Synthetic smoke test; does not measure real participant accuracy.

Run from the repository root:
python Skripte/tests/smoke_pipeline_extras.py --output-dir /tmp/idp-smoke/Run_synthetic
"""
import json, os, sys, types
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blink.run_pipeline_extras import run_stage
from blink import diagnostic_support
import argparse
parser=argparse.ArgumentParser(description='Synthetic integration smoke test; inference and ROIs are fixtures')
parser.add_argument('--output-dir',type=Path,required=True)
args=parser.parse_args()
root=args.output_dir.resolve()
root.mkdir(parents=True,exist_ok=True)
t=np.arange(0.,4000.,20.)
gaze=pd.DataFrame(dict(frame=np.arange(len(t)),timestamp_ms_synced=50000+t,
    trial_assignment=1,phase_type=np.where(t<1500,'fixation','stimulus'),
    gaze_deg_x_calib=np.sin(t/70)*0.1,valid_eye_frame=True,detected=True))
for filename in ['debug_5_ptgaze_calibrated.csv','debug_5_pupil_data_calibrated.csv']:
    gaze.to_csv(root/filename,index=False)
eye=pd.DataFrame(dict(timestamp_ms=50000+t,x_pos=960+np.sin(t/70)*.1,y_pos=540.,pupil_size=100.))
eye.to_csv(root/'debug_3_eyetracker_data.csv',index=False)
pd.DataFrame([dict(trial_number=1,start_time=50000,end_time=51400,duration=1400,is_practice=False)]).to_csv(root/'debug_3_fixations.csv',index=False)
phases={'phases': {'experiment_block1': {'trials': [dict(trial_number=1,is_practice=False,
    fixation=dict(start_video_s=0.,end_video_s=1.5,start_eyelink_ms=50000,end_eyelink_ms=51500),
    stimulus=dict(start_video_s=1.5,end_video_s=4.,start_eyelink_ms=51500,end_eyelink_ms=54000))]}}}
(root/'phases_detected.json').write_text(json.dumps(phases))
video=root/'synthetic.mp4';video.touch();asc=root/'synthetic.asc';asc.touch()

def export(video_path, output_dir, **kwargs):
    log=pd.DataFrame(dict(frame_number=gaze.frame, video_time_ms=t+90000,
        frame_end_time_ms=t+90020,valid_eye_frame=True,reason_group='valid'))
    log.loc[25:29,'valid_eye_frame']=False
    log.loc[25:29,'reason_group']='blink'
    for col in ['left_ear_raw','right_ear_raw','left_ear','right_ear','avg_ear']:
        log[col]=np.where(log.valid_eye_frame,.3,.1)
    for col in ['left_threshold','right_threshold']:log[col]=.16
    for col in ['left_closed_raw','right_closed_raw','left_closed_smooth','right_closed_smooth']:
        log[col]=~log.valid_eye_frame
    log.to_csv(Path(output_dir)/'blink_filter_frame_log.csv',index=False)
    pd.DataFrame([dict(start_frame=25,end_frame=29,reason_group='blink')]).to_csv(Path(output_dir)/'blink_filter_regions.csv',index=False)

def benchmark(**kwargs):
    extra=kwargs['output_dir']
    pd.DataFrame([dict(trial=1,start_from_fixation_ms=510,end_from_fixation_ms=590)]).to_csv(extra/'eyelink_blink_benchmark.csv',index=False)
    pd.DataFrame([dict(trial=1,fixation_start_ms=50000)]).to_csv(extra/'eyelink_trial_markers.csv',index=False)
exporter=types.ModuleType('blink.export_blink_filter_logs');exporter.export_blink_filter_logs=export
benchmark_module=types.ModuleType('blink.extract_eyelink_blink_benchmark');benchmark_module.process_one_asc=benchmark
class Mesh:
    def __init__(self,**kw):pass
    def close(self):pass
fake_mp=types.SimpleNamespace(solutions=types.SimpleNamespace(face_mesh=types.SimpleNamespace(FaceMesh=Mesh)))
with patch.dict(sys.modules, {'blink.export_blink_filter_logs':exporter,
        'blink.extract_eyelink_blink_benchmark':benchmark_module,'mediapipe':fake_mp}), \
     patch.object(diagnostic_support,'extract_frame_at_pts',return_value=np.full((120,240,3),180,dtype=np.uint8)), \
     patch.object(diagnostic_support,'eye_roi',side_effect=lambda mesh,im:(im,None)):
    run_stage(root,video,asc,'synthetic',methods='both')
print(root)
