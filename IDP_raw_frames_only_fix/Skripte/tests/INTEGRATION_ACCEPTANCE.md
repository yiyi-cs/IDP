# Blink / fixation integration acceptance

Base: `yiyi-cs/IDP`, `Blink-Detection`, commit `fb767fb73e26697cdafba1732d1933b40c2d38c7`.

## Implemented

Step 10 now exports blink logs and ASC benchmark/markers into the current Run's
`extra/`, joins the final mask by frame, runs the existing fixation detector, runs
that same detector on EyeLink samples at calibrated timestamps with an independent
P75, and renders both PDFs. Step 11's comparison logic and cache switches remain
unchanged. Cached upstream artifacts must first be copied into the current Run by
the existing cache machinery. New diagnostics never search other Runs or `test60`.

The ten filenames in the integration plan are produced unchanged. Reproduction
information is additionally saved as `extra/eyelink_sanity_samples.csv`,
`extra/eyelink_sanity_metadata.json` and `extra/integration_complete.json`.
Existing `debug_6_*fixations.csv` and fixation metadata remain at Run root.
The completion marker is written only after all ten required outputs exist.

The new stage requires synchronized calibrated gaze and EyeLink inputs. It fails
explicitly if these are unavailable; there is no fallback to raw video time.
PTGaze supplies the timeline when selected; MediaPipe supplies it in MediaPipe-only
runs. Requested gaze methods cannot silently skip a missing calibrated CSV.
Core blink and fixation detectors and their parameter values are unchanged.
`blink_extra/` is unchanged. The CLI entrypoint also no longer starts twice.

## Local checks completed

```bash
PYTHONPATH=Skripte python -m unittest discover -s Skripte/tests -v
python -m compileall -q Skripte/blink Skripte/fixation Skripte/utils/analysis_pipeline.py Skripte/master_cli.py
```

Eight boundary tests cover missing/duplicate/null masks, string booleans, frame
order, synchronized timestamps, PTS isolation, invalid/unmatched EyeLink samples,
the shared detector and independent P75, and empty-event CSV schemas.
A synthetic single-Run smoke check executed both fixation methods, EyeLink
sampling, statistics, output validation and both PDF renderers. Only video
inference, benchmark export and ROI image extraction were replaced with fixtures.
The two synthetic PDF pages were rendered to PNG and visually inspected.
Reproduce the fixture check (outside your real participant results) with:

```bash
python Skripte/tests/smoke_pipeline_extras.py --output-dir /tmp/idp-smoke/Run_synthetic
```

This smoke script deliberately replaces inference, benchmark export and ROI images.
It does not validate MediaPipe/OpenCV inference or actual ASC parsing.

## Required next check: ONE real VP, new Run

Use the existing Python environment with MediaPipe, OpenCV, NumPy, pandas,
Matplotlib and the current gaze-model dependencies. `ffprobe` supplies video PTS;
`ffmpeg` extracts ROI images.

From `Skripte/`:

```bash
python master_cli.py --data-root /Volumes/Empra10/Ergebnisse60 --video-root /Volumes/Empra9/Videos_60
```

Choose one participant (for example `bjs4`), 60 Hz, FullCalib, synchronized
comparison mode (mode 2), and the desired method/preprocessing configuration.
Create a new Run. First use no cache for acceptance. After inspecting this Run,
repeat with the intended upstream cache controls to confirm equivalent integration.
Do not launch all 11 participants until the single-VP check passes.

If a fresh Run already completed calibration, rerun just Step 10 explicitly:

```bash
python -m blink.run_pipeline_extras \
  --run-dir '/Volumes/Empra10/Ergebnisse60/bjs4/Analyse/REPLACE_WITH_NEW_RUN' \
  --video /Volumes/Empra9/Videos_60/bjs4.mp4 \
  --asc '/Volumes/Empra10/Ergebnisse60/bjs4/REPLACE_WITH_ACTUAL_ASC.asc' \
  --vp bjs4 --methods both --preprocess-mode none
```

Only `none` is supported. Image-derived caches must explicitly record `preprocess_mode: none`.
Verify 100% frame-mask coverage, sensible independent thresholds, no shifted
blink shading, the third/fourth timeline rows using saved events, every expected
file in `extra/`, and successful Step 11. Inspect real eye ROIs separately.

Real video inference, actual ASC parsing on participant files, and a complete
new-VP main-pipeline run could not be executed in the development environment:
the external-drive data and OpenCV/MediaPipe dependencies were unavailable.
