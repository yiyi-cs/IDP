"""Recheck observed trial anchors using audio only; never modify an existing Run.

Supports the V2 experiment log used by beo7. A partial recording produces a
partial diagnostic, not a replacement synchronization for the whole recording.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.audio_sync import refine_880_centers, summarize_anchor_offsets


def audit(phases_path, log_path, audio_path, audio_start_s=0.):
    phases = json.loads(Path(phases_path).read_text())
    text = Path(log_path).read_text(encoding='utf-8-sig')
    try:
        log = json.loads(text)
    except json.JSONDecodeError:
        # Historical log writer emitted unescaped Windows paths.
        log = json.loads(text.replace('\\', '/'))
    events = {}
    for event in log.get('events', []):
        name = event.get('event_type', '')
        if name.startswith('start_fixation'):
            number = event.get('trial_number')
            if number not in events or len(event) > len(events[number]):
                events[number] = event
    sr, audio = wavfile.read(audio_path)
    if audio.ndim != 1:
        raise ValueError('Use mono audio (ffmpeg -ac 1)')
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(float) / max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
    audio = audio.astype(float)
    rows = []
    expected_markers = 0
    for block in ('experiment_block1', 'experiment_block2'):
        for trial in phases['phases'].get(block, {}).get('trials', []):
            event = events.get(trial['trial_number'], {})
            required = ('audio_play_start_eyelink_ms', 'audio_play_actual_delay_ms', 'audio_duration_ms')
            if any(key not in event for key in required):
                raise ValueError(f'Trial {trial["trial_number"]}: V2 audio timestamps required; no guessed latency')
            start_ms = event['audio_play_start_eyelink_ms'] * 1000 + event['audio_play_actual_delay_ms']
            fix = trial['fixation']
            for frequency, time_key, found_key in (
                    (440, 'start_video_s', 'audio_found'), (880, 'end_video_s', 'end_audio_found')):
                expected_markers += 1
                if not fix.get(found_key, False):
                    continue
                old_time = float(fix[time_key])
                relative = old_time - audio_start_s
                if not .4 <= relative <= len(audio)/sr - .4:
                    continue
                if frequency == 880:
                    lo = max(0, int((relative - 1.) * sr))
                    hi = min(len(audio), int((relative + 1.) * sr))
                    centers = refine_880_centers(audio[lo:hi], sr, [relative - lo/sr])
                    if not centers:
                        continue
                    new_time = centers[0] + lo/sr + audio_start_s
                else:
                    # Retain the observed narrow-band center; do not infer a
                    # new 440 Hz detection from the experiment's predicted time.
                    new_time = old_time
                reference = start_ms + (100 if frequency == 440 else event['audio_duration_ms'] - 100)
                rows.append(dict(trial_number=trial['trial_number'], frequency=frequency,
                    old_video_time_s=old_time, new_video_time_s=new_time,
                    marker_shift_ms=(new_time-old_time)*1000,
                    old_offset_ms=reference-old_time*1000,
                    new_offset_ms=reference-new_time*1000,
                    audio_rechecked=frequency == 880))
    quality = summarize_anchor_offsets([r['new_offset_ms'] for r in rows], [r['frequency'] for r in rows])
    old_offset = float(phases['sync_info']['offset_ms'])
    new_offset = quality['selected_trial_offset_ms']
    quality.update(source_offset_ms=old_offset,
        diagnostic_offset_change_ms=new_offset-old_offset if new_offset is not None else None,
        expected_trial_marker_count=expected_markers,
        inspected_trial_marker_count=len(rows),
        whole_recording_checked=len(rows) == expected_markers,
        model_inference_rerun=False, existing_run_modified=False,
        note='440 Hz uses saved observations; 880 Hz centers rechecked against supplied audio. '
             'No absolute A/V validation, gaze-lag compensation or drift correction.')
    if len(rows) != expected_markers:
        quality['warnings'].append('partial_anchor_coverage_not_a_full_run_replacement')
    return rows, quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phases', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audio', type=Path)
    group.add_argument('--video', type=Path)
    parser.add_argument('--audio-start-s', type=float, default=0.)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.video and args.audio_start_s != 0:
        parser.error('--audio-start-s is only for an extracted audio clip')
    if args.output_dir.exists():
        parser.error('--output-dir must be a new directory')
    with tempfile.TemporaryDirectory() as tmp:
        audio = args.audio
        if args.video:
            audio = Path(tmp) / 'audio.wav'
            subprocess.run(['ffmpeg','-v','error','-i',str(args.video),'-map','0:a:0',
                            '-ac','1','-c:a','pcm_s16le',str(audio)], check=True)
        rows, quality = audit(args.phases, args.log, audio, args.audio_start_s)
    if not rows:
        raise ValueError('No supported observed markers in supplied audio')
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / 'audio_anchor_audit.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / 'audio_sync_quality.json').write_text(json.dumps(quality, indent=2, allow_nan=False))
    print(json.dumps(quality, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
