#!/bin/zsh
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.10 || command -v python3)}"

VIDEO_ROOT="${VIDEO_ROOT:-/Volumes/Empra9/Videos_60}"
RESULTS_ROOT="${RESULTS_ROOT:-/Volumes/Empra10/Ergebnisse60}"

# Final validation output location
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-test60}"

# Final setting: preprocessing did not materially improve Blink Detection.
PREPROCESS_MODE="${PREPROCESS_MODE:-none}"

# true: delete old two CSVs before rerunning
# false: skip VP if both final CSVs already exist
OVERWRITE="${OVERWRITE:-true}"

EXPORT_SCRIPT="$SCRIPT_DIR/export_blink_filter_logs.py"

VPS=(
  beo7
  bjs4
  egf5
  fbn6
  fgt6
  jkl7
  kdn8
  kro3
  mhe9
  oem4
  ogt7
)

[[ -n "$PYTHON_BIN" ]] || {
  echo "[FATAL] No python3.10/python3 found."
  exit 1
}

[[ -f "$EXPORT_SCRIPT" ]] || {
  echo "[FATAL] Missing export script: $EXPORT_SCRIPT"
  exit 1
}

[[ -d "$VIDEO_ROOT" ]] || {
  echo "[FATAL] Missing video root: $VIDEO_ROOT"
  exit 1
}

mkdir -p "$RESULTS_ROOT"

echo "============================================================"
echo "60 Hz blink-filter batch export"
echo "============================================================"
echo "Video root:      $VIDEO_ROOT"
echo "Results root:    $RESULTS_ROOT"
echo "Output subdir:   $OUTPUT_SUBDIR"
echo "Preprocess mode: $PREPROCESS_MODE"
echo "Overwrite:       $OVERWRITE"
echo "============================================================"

success=0
failed=0
skipped=0

for vp in "${VPS[@]}"; do

  video="$VIDEO_ROOT/$vp.mp4"
  output_dir="$RESULTS_ROOT/$vp/$OUTPUT_SUBDIR"

  frame_log="$output_dir/blink_filter_frame_log.csv"
  regions="$output_dir/blink_filter_regions.csv"

  echo ""
  echo "------------------------------------------------------------"
  echo "VP: $vp"
  echo "Video:  $video"
  echo "Output: $output_dir"
  echo "------------------------------------------------------------"

  if [[ ! -f "$video" ]]; then
    echo "[ERROR] Missing video: $video"
    (( failed += 1 ))
    continue
  fi

  if [[ "$OVERWRITE" != "true" \
        && -f "$frame_log" \
        && -f "$regions" ]]; then

    echo "[SKIP] Final outputs already exist."
    (( skipped += 1 ))
    continue
  fi

  mkdir -p "$output_dir"

  # Avoid mixing stale and newly generated results.
  if [[ "$OVERWRITE" == "true" ]]; then
    rm -f "$frame_log" "$regions"
  fi

  if "$PYTHON_BIN" "$EXPORT_SCRIPT" \
      --video "$video" \
      --output-dir "$output_dir" \
      --preprocess-mode "$PREPROCESS_MODE"; then

    if [[ -f "$frame_log" && -f "$regions" ]]; then
      echo "[OK] $vp"
      (( success += 1 ))
    else
      echo "[ERROR] Expected CSV outputs missing after run."
      (( failed += 1 ))
    fi

  else
    echo "[ERROR] Export failed."
    (( failed += 1 ))
  fi

done

echo ""
echo "============================================================"
echo "Processing finished"
echo "Successful: $success"
echo "Skipped:    $skipped"
echo "Failed:     $failed"
echo "============================================================"

(( failed == 0 )) || exit 1