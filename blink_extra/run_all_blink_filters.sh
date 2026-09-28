#!/bin/zsh
set -u
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.10 || command -v python3)}"
VIDEO_DIR="${VIDEO_DIR:-/Volumes/Empra9/Videos_60}"
RESULTS_ROOT="${RESULTS_ROOT:-/Volumes/Empra10/Ergebnisse60}"
PREPROCESS_MODE="${PREPROCESS_MODE:-lanczos_2x}"
OVERWRITE="${OVERWRITE:-true}"
VPS=(beo7 bjs4 egf5 fbn6 fgt6 jkl7 kdn8 kro3 mhe9 oem4 ogt7)
PYTHON_SCRIPT="$PROJECT_ROOT/export_blink_filter_logs.py"

for vp in "${VPS[@]}"; do
  run_dir=$(find "$RESULTS_ROOT/$vp/Analyse" -maxdepth 1 -type d -name "Run_60hz_FullCalib_*" -print 2>/dev/null | sort | tail -1)
  if [[ -z "$run_dir" ]]; then
    echo "[ERROR] No 60Hz FullCalib run for $vp"
    continue
  fi
  regions_file="$run_dir/blink_filter_regions.csv"
  if [[ "$OVERWRITE" == false && -f "$regions_file" ]]; then
    echo "[SKIP] $vp"
    continue
  fi
  "$PYTHON_BIN" "$PYTHON_SCRIPT" --vp "$vp" --video-root "$VIDEO_DIR" --run-dir "$run_dir" --preprocess-mode "$PREPROCESS_MODE"
done
