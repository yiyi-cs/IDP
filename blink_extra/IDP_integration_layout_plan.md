# IDP pipeline integration — branch Blink-Detection

Source of truth: `yiyi-cs/IDP`, branch `Blink-Detection`.

## Directory contract

- `Skripte/blink/`: blink frame-log exporter, EyeLink blink benchmark exporter, blink diagnostic PDF generator, blink diagnostic statistics, and pipeline orchestration.
- `Skripte/fixation/`: detector, runner, EyeLink sanity-check adapter (calling the same `detect_fixations` function), and four-row PDF generator.
- `blink_extra/`: read-only reference scripts. No pipeline call or output shall depend on their location.

## Run output contract

`<results>/<vp>/Analyse/<Run>/extra/`:

- `blink_filter_frame_log.csv`
- `blink_filter_regions.csv`
- `eyelink_blink_benchmark.csv`
- `eyelink_trial_markers.csv`
- `blink_diagnostic_metrics.csv`
- `blink_event_matches.csv`
- `blink_diagnostic_summary.csv`
- `eyelink_fixations_sanity.csv`
- `blink_diagnostics.pdf`
- `fixation_four_row_timelines.pdf`

Existing `debug_*.csv` and other core pipeline artifacts remain at Run root.

## Execution order

1. Existing gaze estimation and calibration (including `debug_5`).
2. Blink frame-log export and EyeLink EBLINK/marker export directly into `extra/`.
3. Join current Run's blink validity mask by `frame` with `debug_5_*` gaze data; assert complete one-to-one coverage and fail on mismatch.
4. Apply existing `fixation.fixation_detector.detect_fixations` to validated calibrated gaze.
5. Apply that **same** detector to EyeLink gaze sampled at the calibrated gaze timestamps; estimate an independent P75 and save events into `extra/`.
6. Render Blink Diagnostic and Fixation Timeline PDFs to `extra/`.
7. Maintain Step 11 existing comparison logic and cache controls; avoid pulling results from other runs.

## Invariants / tests

- Video time (`PTS`) is used for eye ROI image localization only; synchronized gaze timestamps for fixation and interval comparison.
- The frame join must cover all calibrated gaze frames. Null mask values are errors, not silently treated as valid.
- Both third and fourth timeline rows use same detector implementation.
- A Run's two PDF files and intermediate CSVs must only read data from that exact Run and its `extra/`.
- `blink_extra/` files remain unchanged.
- Do not change the model or I-DT thresholds when integrating.
- Test new Run for one VP before batching 11 VPs. 
