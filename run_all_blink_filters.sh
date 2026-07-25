#!/bin/zsh

# ==============================================================================
# Run export_blink_filter_logs.py for all VPs
# ==============================================================================

# Stop only for unset variables; individual VP failures are handled manually.

set -u

# PYTHON_BIN="/opt/homebrew/bin/python3.10"
PYTHON_BIN="$(command -v python3.10)"

PROJECT_ROOT="/Users/yiyi_mac/IDP_Code"
VIDEO_DIR="/Volumes/Empra/Videos_25"
PYTHON_SCRIPT="$PROJECT_ROOT/Skripte/debug/export_blink_filter_logs.py"


# true  = rerun and overwrite existing CSV files
# false = skip a VP when blink_filter_regions.csv already exists
OVERWRITE=true

VPS=(
  beo7
  bjs4
  egf5
  fbn6
  fgt6
  jkl7
  kdn8
  kly9
  kro3
  ldj9
  mhe9
  oem4
  ogt7
)

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "[ERROR] Python script not found:"
  echo "        $PYTHON_SCRIPT"
  exit 1
fi

if [[ ! -d "$VIDEO_DIR" ]]; then
  echo "[ERROR] Video directory not found:"
  echo "        $VIDEO_DIR"
  echo "Make sure /Volumes/Empra is mounted."
  exit 1
fi

cd "$PROJECT_ROOT" || exit 1

successful_vps=()
failed_vps=()
skipped_vps=()

echo "============================================================"
echo "Blink-filter batch export"
echo "Project:   $PROJECT_ROOT"
echo "Videos:    $VIDEO_DIR"
echo "Overwrite: $OVERWRITE"
echo "============================================================"

for vp in "${VPS[@]}"; do
  video_path="$VIDEO_DIR/${vp}.MP4"
  output_dir="$PROJECT_ROOT/Ergebnisse/${vp}/test"
  regions_file="$output_dir/blink_filter_regions.csv"

  echo
  echo "------------------------------------------------------------"
  echo "[VP] $vp"
  echo "Video:  $video_path"
  echo "Output: $output_dir"
  echo "------------------------------------------------------------"

  if [[ ! -f "$video_path" ]]; then
    echo "[ERROR] Video not found. Skipping $vp."
    failed_vps+=("$vp")
    continue
  fi

  if [[ "$OVERWRITE" == false && -f "$regions_file" ]]; then
    echo "[SKIP] Existing result found:"
    echo "       $regions_file"
    skipped_vps+=("$vp")
    continue
  fi

  mkdir -p "$output_dir"

  PYTHONPATH="$PROJECT_ROOT/Skripte" \
  "$PYTHON_BIN" "$PYTHON_SCRIPT" \
    --video "$video_path" \
    --output-dir "$output_dir" \
    --ap-code "$vp"
  
  exit_code=$?



  if [[ $exit_code -eq 0 ]]; then
    echo "[OK] Finished $vp"
    successful_vps+=("$vp")
  else
    echo "[ERROR] $vp failed with exit code $exit_code"
    failed_vps+=("$vp")
  fi
done

echo
echo "============================================================"
echo "Batch finished"
echo "============================================================"
echo "Successful (${#successful_vps[@]}): ${successful_vps[*]:-none}"
echo "Skipped    (${#skipped_vps[@]}): ${skipped_vps[*]:-none}"
echo "Failed     (${#failed_vps[@]}): ${failed_vps[*]:-none}"

if (( ${#failed_vps[@]} > 0 )); then
  exit 1
fi

exit 0
