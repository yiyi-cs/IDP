"""Rebuild a legacy Run's BLINK clock diagnostic without rerunning models.

The output is a NEW diagnostic directory. Legacy gaze is shown unchanged as
context: its frame identifiers are unverified after OpenCV VFR seeking. It is
unsafe to relabel that gaze or recompute fixation from these CSVs alone.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from shared.frame_timing import sync_offset, TIMEBASE_VERSION


def repair_run(source, output, vp, include_rois=False, video=None):
    started = time.perf_counter()
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise ValueError('Output must be a new separate directory; original Run is never overwritten')
    root_files = ['phases_detected.json', 'debug_4_sync_info.json', 'config_used.json',
                  'debug_3_eyetracker_data.csv']
    extra_files = ['blink_filter_frame_log.csv', 'blink_filter_regions.csv',
                   'eyelink_blink_benchmark.csv', 'eyelink_trial_markers.csv']
    required = [source / name for name in root_files]
    required += [source / 'extra' / name for name in extra_files]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    methods = {m: name for m, name in {
        'mediapipe':'debug_5_pupil_data_calibrated.csv',
        'ptgaze':'debug_5_ptgaze_calibrated.csv'}.items() if (source / name).exists()}
    if not methods:
        raise ValueError('No gaze input for diagnostic context')
    if include_rois and (video is None or not Path(video).is_file()):
        raise ValueError('A real video path is required for ROI extraction')
    output.mkdir(parents=True)
    extra = output / 'extra'
    extra.mkdir()
    for path in required:
        dest = extra / path.name if path.parent.name == 'extra' else output / path.name
        shutil.copy2(path, dest)
    source_config = json.loads((source / 'config_used.json').read_text())
    config = dict(source_config, diagnostic_only=True, legacy_gaze_context=True,
                  timebase_version='legacy_blink_clock_diagnostic', source_run=str(source),
                  model_outputs_recomputed=False, requires_fresh_inference=True)
    (output / 'config_used.json').write_text(json.dumps(config, indent=2))
    frames = pd.read_csv(extra / 'blink_filter_frame_log.csv').set_index('frame_number')
    audit = []
    for method, name in methods.items():
        shutil.copy2(source / name, output / name)
        gaze = pd.read_csv(source / name)
        declared_pts = gaze['frame'].map(frames['video_time_ms'])
        discrepancy = declared_pts + sync_offset(output) - gaze['timestamp_ms_synced']
        audit.append(dict(method=method, n_gaze_rows=len(gaze),
            median_declared_frame_discrepancy_ms=float(discrepancy.median()),
            max_abs_declared_frame_discrepancy_ms=float(discrepancy.abs().max()),
            frame_identity_verified=False, gaze_timestamps_changed=False))
    pd.DataFrame(audit).to_csv(extra / 'timebase_audit.csv', index=False)
    from blink.generate_blink_diagnostics import render_run
    preferred = 'ptgaze' if 'ptgaze' in methods else 'mediapipe'
    render_run(output, video, vp, preferred, include_rois=include_rois)
    summary = dict(source_run=str(source), diagnostic_only=True,
        duration_seconds=time.perf_counter()-started,
        source_preprocess_mode=source_config.get('preprocess_mode', 'unknown'),
        model_outputs_recomputed=False, gaze_timestamps_changed=False,
        fixation_recomputed=False, offset_used_ms=sync_offset(output),
        blink_synchronized_timebase=TIMEBASE_VERSION,
        requires_fresh_inference=True, requires_fresh_calibration=True,
        roi_images_included=include_rois,
        warning='Legacy gaze frame identities are unverified. Blink clock diagnostic only; not raw-frame acceptance.')
    (extra / 'blink_clock_diagnostic_complete.json').write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(audit).to_string(index=False))
    print(f'[OK] Blink clock diagnostic only: {output}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--vp', required=True)
    parser.add_argument('--include-rois', action='store_true')
    parser.add_argument('--video', type=Path)
    args = parser.parse_args()
    repair_run(args.run_dir, args.output_dir, args.vp, args.include_rois, args.video)


if __name__ == '__main__':
    main()
