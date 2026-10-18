# Raw-frame / exact-frame-time pipeline acceptance

Base remote branch: yiyi-cs/IDP Blink-Detection, 2068bae.

## Corrected contract

Production frames are never enlarged. Only preprocessing mode `none` is allowed.
Frame selection uses the ffprobe presentation-time table. MediaPipe and PTGaze
trial extraction and offline calibration decode exact windows with ffmpeg accurate
PTS seeking and output frame-rate conversion disabled. OpenCV frame-number seeks
are not used for these windows: on VFR material their returned images need not
match the ordinal written into the CSV.

`timestamp_ms` is the frame's actual presentation time. debug_4 applies the same
existing synchronization offset to form `timestamp_ms_synced`. Detectors and
interval comparisons consume synchronized time only; they never compare raw PTS
with EyeLink events. No sync offset, blink model parameter or I-DT parameter was
retuned to improve matches.

This corrects the initial plan's assumption that upstream gaze timestamps and
frame identities were already reliable. PTS is now also used UPSTREAM to construct
that reliable synchronized clock, in addition to its ROI localization role.

Frame timestamp tables and provenance JSONs are written to the current Run/extra/.
Both gaze streams use the same mapping. Trial labels are reassigned from the
synchronized clock. Every measured gaze frame must still match the current Run's
blink mask one-to-one. Missing calibrated samples stay invalid; no interpolation
or invented gaze values are used. The unchanged fixation detector therefore sees
explicit gaps instead of accidentally joining separated valid segments.

Blink diagrams cover the FULL synchronized frame log, including intervals where
no calibrated gaze was recorded. Gaze lines break across missing frame ranges.
The core fixation detector remains unchanged and is used for both Webcam and
EyeLink. The original Step 11 comparison calculation and cache controls remain.

## Cache rules

Image-dependent cached data and calibrations must explicitly record both
`preprocess_mode: none` and `timebase_version: frame_pts_v1`. This is checked
again during copying, including explicitly selected cache sources. Old or unknown
clock/image provenance is not relabelled as corrected. EyeLink import and audio
phase detection remain independent of the image clock and may be reused.
Custom calibration PKLs also require matching provenance from their source Run.

## Checks completed

From repository root:

```bash
PYTHONPATH=Skripte python -m unittest discover -s Skripte/tests -v
python -m compileall -q Skripte
```

20 tests pass. One test generates a genuine VFR video and checks every selected
frame pixel-for-pixel against full sequential ffmpeg decoding. Other tests cover
frame-mask coverage, legacy-clock rejection, raw-frame cache isolation, event gaps,
clock conversion, and use of the same detector with independent P75 estimation.
A synthetic Step 10 smoke run also completes with inference/ROI fixtures.

Actual uploaded beo7 CSVs were used to rebuild a 24-page blink clock diagnostic.
The pre-existing offset and 100 ms match tolerance were preserved. Matches changed
from TP=3, FP=20, FN=73 to TP=24, FP=0, FN=52. This is an alignment diagnostic;
it is not a new raw-frame model accuracy measurement.

## Fast check with existing files (no model inference)

Run from Skripte/, use a NEW output directory:

```bash
python -m blink.repair_run_timestamps \
  --run-dir /Volumes/Empra10/Ergebnisse60/beo7/Analyse/Run_60hz_FullCalib_mode2_20261018_220853 \
  --output-dir /Volumes/Empra10/Ergebnisse60/beo7/Analyse/beo7_blink_clock_check \
  --vp beo7
```

This only rebuilds blink diagnostic PDF and statistics. Gaze is copied unchanged
and labelled unverified legacy context. Fixation is NOT recomputed from ambiguous
frame IDs. No original file is overwritten. ROI images are omitted by default;
add `--include-rois --video /Volumes/Empra9/Videos_60/beo7.mp4` to extract real ROIs.
The diagnostic copy is marked non-production and cannot enter image caches.

## Required next acceptance: one fresh beo7 Run

From Skripte/ in the existing environment:

```bash
python master_cli.py --data-root /Volumes/Empra10/Ergebnisse60 --video-root /Volumes/Empra9/Videos_60
```

Select analysis, only beo7, 60 Hz, FullCalib, mode 2, both methods. Regenerate image
inference and calibration. Do not reuse old gaze/calibration results: their image
identity and/or calibration extraction windows were wrong. Check config_used.json
records none + frame_pts_v1, inspect both PDFs, and confirm Step 11 completion.
Only then batch the remaining VPs.

The development environment lacks the actual source video, OpenCV/MediaPipe and
the production gaze models. Full real-beo7 inference and calibration therefore
remain unverified here. ffmpeg and ffprobe are required for the repaired decoder.
The previous Run's 102.56 min includes both gaze models, both calibrations and a
full blink pass; no reliable per-stage timing profile was available. No fixed
runtime improvement is claimed for this correction.
