"""Run-local pipeline stage: blink exports -> fixation -> sanity check -> PDFs.

Manual acceptance on one newly calibrated Run:
python -m blink.run_pipeline_extras --run-dir /path/to/Run --video /path/vp.mp4
    --asc /path/vp.asc --vp vp --methods both --preprocess-mode none
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.frame_timing import TIMEBASE_VERSION, detector_input, load_frame_clock, sync_offset
from blink.run_data import load_validated_gaze, validate_gaze_clock
from shared.shared_image_preprocessing import validate_preprocess_mode

EXPECTED_OUTPUTS = (
    'blink_filter_frame_log.csv', 'blink_filter_regions.csv',
    'eyelink_blink_benchmark.csv', 'eyelink_trial_markers.csv',
    'blink_diagnostic_metrics.csv', 'blink_event_matches.csv',
    'blink_diagnostic_summary.csv', 'eyelink_fixations_sanity.csv',
    'blink_diagnostics.pdf', 'fixation_four_row_timelines.pdf',
)


def run_stage(run_dir, video, asc, vp, methods='both', preprocess_mode='none'):
    validate_preprocess_mode(preprocess_mode)
    run, video, asc = (Path(p).resolve() for p in (run_dir, video, asc))
    if (run / 'config_used.json').exists() and json.loads((run / 'config_used.json').read_text()).get('diagnostic_only'):
        raise ValueError('Temporal-only diagnostic copies cannot be used as production Runs')
    if methods not in ('both', 'mediapipe', 'ptgaze'):
        raise ValueError(f'Unknown gaze methods: {methods}')
    for p in [run / 'phases_detected.json', run / 'debug_3_eyetracker_data.csv',
              run / 'debug_3_fixations.csv', video, asc]:
        if not p.is_file():
            raise FileNotFoundError(p)
    extra = run / 'extra'
    extra.mkdir(exist_ok=True)
    marker = extra / 'integration_complete.json'
    marker.unlink(missing_ok=True)
    filenames = {'mediapipe': 'debug_5_pupil_data_calibrated.csv',
                 'ptgaze': 'debug_5_ptgaze_calibrated.csv'}
    selected = list(filenames) if methods == 'both' else [methods]
    # Fail cheaply before rerunning blink inference if calibration is unsuitable.
    import pandas as pd
    import numpy as np
    clock = load_frame_clock(video, run)
    reference = pd.Series(clock.times_ms)
    offset = sync_offset(run)
    for method in selected:
        columns = pd.read_csv(run / filenames[method], nrows=0).columns
        required = {'frame', 'timestamp_ms_synced', 'phase_type', 'gaze_deg_x_calib'}
        if not required.issubset(columns):
            raise ValueError(f'{method}: missing calibrated synchronized columns: {required-set(columns)}')
        source = pd.read_csv(run / filenames[method], usecols=['timestamp_source']) if 'timestamp_source' in columns else None
        if source is None or not source['timestamp_source'].eq(TIMEBASE_VERSION).all():
            raise ValueError(f'{method}: old frame/FPS timebase; regenerate or use diagnostic repair')
        if not {'trial_number', 'trial_assignment'}.intersection(columns):
            raise ValueError(f'{method}: missing trial assignment')
        data = pd.read_csv(run / filenames[method])
        validate_gaze_clock(data, reference, offset)
        if not np.isfinite(pd.to_numeric(data['gaze_deg_x_calib'], errors='coerce')).any():
            raise ValueError(f'{method}: no finite calibrated gaze samples')
    from blink.export_blink_filter_logs import export_blink_filter_logs
    from blink.extract_eyelink_blink_benchmark import process_one_asc
    export_blink_filter_logs(str(video), str(extra), ap_code=vp,
                            run_name=run.name, preprocess_mode=preprocess_mode)
    process_one_asc(asc_file=asc, vp_code=vp, output_dir=extra)
    # Validate EVERY selected method before writing fixation outputs.
    validated = {m: load_validated_gaze(run, filenames[m]) for m in selected}
    env = os.environ.copy()
    env.update(PIPELINE_OUTPUT_BASE_DIR=str(run), PIPELINE_METHODS=methods)
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] /
                    'fixation' / 'run_fixation_detection.py')], env=env, check=True)
    preferred = 'ptgaze' if 'ptgaze' in selected else 'mediapipe'
    from fixation.eyelink_sanity import run_sanity
    run_sanity(run, detector_input(validated[preferred], pd.read_csv(extra / 'blink_filter_frame_log.csv'), run), preferred)
    from blink.generate_blink_diagnostics import render_run as render_blink
    from fixation.generate_fixation_timeline_pdf import render_run as render_fixation
    render_blink(run, video, vp, preferred)
    render_fixation(run, vp, preferred)
    for name in EXPECTED_OUTPUTS:
        if not (extra / name).is_file() or (extra / name).stat().st_size == 0:
            raise RuntimeError(f'Missing or empty integration output: {name}')
    marker.write_text(json.dumps(dict(run_dir=str(run), video=str(video), asc=str(asc),
        methods=methods, preprocess_mode=preprocess_mode, timebase_version=TIMEBASE_VERSION, outputs=EXPECTED_OUTPUTS), indent=2))
    print(f'[OK] Integrated blink/fixation artifacts: {extra}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--asc', type=Path, required=True)
    parser.add_argument('--vp', required=True)
    parser.add_argument('--methods', choices=['both', 'mediapipe', 'ptgaze'], default='both')
    parser.add_argument('--preprocess-mode', choices=['none'], default='none')
    args = parser.parse_args()
    run_stage(args.run_dir, args.video, args.asc, args.vp, args.methods, args.preprocess_mode)


if __name__ == '__main__':
    main()
