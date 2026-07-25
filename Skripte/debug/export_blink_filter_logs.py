#!/usr/bin/env python3
"""
Export blink / eye-frame filter logs for a full video or a selected trial segment.

Outputs
-------
1. blink_filter_frame_log.csv
   One row per exported video frame, including raw/smoothed EAR values,
   thresholds, eye-closure flags, validity, and invalid reasons.

2. blink_filter_regions.csv
   Consecutive invalid frames merged into regions for Figure 7 overlays in R.

Important
---------
- Run this script with the same video version (25 Hz or 60 Hz) that was used
  to create the gaze data shown in R.
- When processing only a trial segment, use:
      --process-start-ms <trial start in full video>
      --process-end-ms   <trial end in full video>
      --time-zero-ms     <trial start in full video>
  The default --warmup-ms 2000 processes two seconds before the selected
  segment to initialize the adaptive blink baseline, but does not export
  those warm-up frames.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd


# =============================================================================
# PATH SETUP
# =============================================================================

# Expected location:
#   IDP_Code/Skripte/debug/export_blink_filter_logs.py
#
# parents[0] = debug
# parents[1] = Skripte
# parents[2] = IDP_Code
SCRIPT_PATH = Path(__file__).resolve()
SKRIPTE_DIR = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]

# "shared" is located in IDP_Code/Skripte/shared.
# PROJECT_ROOT is also added because other project imports may depend on it.
for import_root in (SKRIPTE_DIR, PROJECT_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

from shared.shared_pupil_detection import RobustPupilDetector


# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================

FRAME_LOG_COLUMNS = [
    "ap_code",
    "run_name",
    "vp_label",
    "trial_nr",
    "video_path",
    "video_fps",
    "frame_duration_ms",
    "frame_number",
    "video_time_ms",
    "time_ms",
    "valid_eye_frame",
    "invalid_reason",
    "reason_group",
    "is_blink",
    "blink_count",
    "frames_below_threshold",
    "eyes_closed",
    "is_suspected_blink",
    "is_transition_frame",
    "left_ear_raw",
    "right_ear_raw",
    "left_ear",
    "right_ear",
    "avg_ear",
    "left_threshold",
    "right_threshold",
    "left_closed",
    "right_closed",
    "left_closed_raw",
    "right_closed_raw",
    "left_closed_smooth",
    "right_closed_smooth",
    "ear_asymmetry",
    "baseline_samples",
    "position_x",
    "position_y",
    "confidence",
    "plausibility_passed",
]

REGION_COLUMNS = [
    "ap_code",
    "run_name",
    "vp_label",
    "trial_nr",
    "video_path",
    "video_fps",
    "start_frame",
    "end_frame",
    "start_video_time_ms",
    "end_video_time_ms",
    "start_ms",
    "end_ms",
    "reason_group",
    "invalid_reason",
    "n_frames",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def map_reason_group(invalid_reason: Optional[str]) -> str:
    """
    Map detailed invalid reasons to visualization groups.

    Returns one of:
    - blink
    - suspected
    - invalid

    This function is only called for frames already marked invalid.
    Therefore, a missing reason must remain "invalid", not "valid".
    """
    if invalid_reason is None or pd.isna(invalid_reason):
        return "invalid"

    if invalid_reason == "blink_or_eyes_closed":
        return "blink"

    if invalid_reason in {
        "blink_transition",
        "suspected_blink_or_landmark_error",
    }:
        return "suspected"

    return "invalid"


def infer_metadata_from_output_dir(output_dir: Path):
    """
    Infer AP/VP code and run name from common project output paths.

    Examples
    --------
    Ergebnisse/bjs4/Analyse/Run_25hz_...
        -> ap_code = bjs4
        -> run_name = Run_25hz_...

    Ergebnisse/beo7/test
        -> ap_code = beo7
        -> run_name = test
    """
    output_dir = output_dir.resolve()
    run_name = output_dir.name
    ap_code = None

    parts = output_dir.parts

    # Preferred structure: .../Ergebnisse/<ap_code>/Analyse/<run_name>
    if "Analyse" in parts:
        analyse_index = len(parts) - 1 - list(reversed(parts)).index("Analyse")
        if analyse_index >= 1:
            ap_code = parts[analyse_index - 1]

    # Fallback: .../Ergebnisse/<ap_code>/<some output folder>
    if ap_code is None and "Ergebnisse" in parts:
        ergebnisse_index = len(parts) - 1 - list(reversed(parts)).index("Ergebnisse")
        if ergebnisse_index + 1 < len(parts):
            ap_code = parts[ergebnisse_index + 1]

    return ap_code, run_name


def reset_detector_state(detector: RobustPupilDetector) -> None:
    """
    Reset temporal state before processing a new video.

    Blink detection keeps state for:
    - adaptive open-eye baseline
    - closed-frame counter
    - transition padding
    - blink counter
    """
    if hasattr(detector, "reset"):
        detector.reset()
        return

    blink_detector = getattr(detector, "blink_detector", None)
    if blink_detector is not None and hasattr(blink_detector, "reset"):
        blink_detector.reset()

    head_pose_estimator = getattr(detector, "head_pose_estimator", None)
    if head_pose_estimator is not None and hasattr(head_pose_estimator, "reset"):
        head_pose_estimator.reset()


def safe_get_position_component(position, index: int):
    if position is None:
        return None

    try:
        return float(position[index])
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def safe_bool(value: Any) -> bool:
    """Convert detector output to bool without treating None as valid."""
    return bool(value) if value is not None else False


def validate_arguments(
    process_start_ms: float,
    process_end_ms: Optional[float],
    time_zero_ms: float,
    warmup_ms: float,
    max_frames: Optional[int],
) -> None:
    if process_start_ms < 0:
        raise ValueError("--process-start-ms must be >= 0.")

    if process_end_ms is not None and process_end_ms <= process_start_ms:
        raise ValueError(
            "--process-end-ms must be greater than --process-start-ms."
        )

    if time_zero_ms < 0:
        raise ValueError("--time-zero-ms must be >= 0.")

    if warmup_ms < 0:
        raise ValueError("--warmup-ms must be >= 0.")

    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be a positive integer.")


def choose_region_reason_group(region_df: pd.DataFrame) -> str:
    """
    Assign one visualization group to a merged invalid region.

    Priority:
        blink > suspected > invalid

    This keeps one physical blink/unstable-eye episode as one rectangle even
    when its individual frames change from "blink" to "suspected".
    """
    groups = set(region_df["reason_group"].dropna().astype(str))

    if "blink" in groups:
        return "blink"

    if "suspected" in groups:
        return "suspected"

    return "invalid"


def combine_invalid_reasons(region_df: pd.DataFrame) -> str:
    """Join distinct frame-level invalid reasons in first-occurrence order."""
    reasons: List[str] = []

    for reason in region_df["invalid_reason"]:
        if reason is None or pd.isna(reason):
            reason_text = "unknown_invalid_reason"
        else:
            reason_text = str(reason)

        if reason_text not in reasons:
            reasons.append(reason_text)

    return "|".join(reasons)


def build_invalid_regions(
    frame_log: pd.DataFrame,
    frame_duration_ms: float,
) -> pd.DataFrame:
    """
    Merge consecutive invalid frames into regions.

    A new region starts only when frame numbers are no longer consecutive.
    Changes between blink/suspected/transition reasons do not split a single
    physical event into multiple rectangles.
    """
    if frame_log.empty:
        return pd.DataFrame(columns=REGION_COLUMNS)

    required_columns = {
        "valid_eye_frame",
        "frame_number",
        "video_time_ms",
        "time_ms",
        "reason_group",
        "invalid_reason",
    }
    missing_columns = required_columns.difference(frame_log.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise KeyError(
            f"Cannot build invalid regions. Missing frame-log columns: "
            f"{missing_text}"
        )

    invalid_df = frame_log.loc[
        frame_log["valid_eye_frame"].eq(False)
    ].copy()

    if invalid_df.empty:
        return pd.DataFrame(columns=REGION_COLUMNS)

    invalid_df = invalid_df.sort_values("frame_number").reset_index(drop=True)

    # A new region starts after every gap in frame numbering.
    region_id = (
        invalid_df["frame_number"]
        .diff()
        .ne(1)
        .cumsum()
    )

    regions: List[Dict[str, Any]] = []

    for _, region_df in invalid_df.groupby(region_id, sort=False):
        region_df = region_df.reset_index(drop=True)
        start_row = region_df.iloc[0]
        end_row = region_df.iloc[-1]

        regions.append({
            "ap_code": start_row.get("ap_code"),
            "run_name": start_row.get("run_name"),
            "vp_label": start_row.get("vp_label"),
            "trial_nr": start_row.get("trial_nr"),
            "video_path": start_row.get("video_path"),
            "video_fps": start_row.get("video_fps"),

            "start_frame": int(start_row["frame_number"]),
            "end_frame": int(end_row["frame_number"]),

            # Absolute time inside the original video.
            "start_video_time_ms": float(start_row["video_time_ms"]),
            "end_video_time_ms": float(
                end_row["video_time_ms"] + frame_duration_ms
            ),

            # Time aligned to the requested zero point, normally trial-relative.
            "start_ms": float(start_row["time_ms"]),
            "end_ms": float(end_row["time_ms"] + frame_duration_ms),

            "reason_group": choose_region_reason_group(region_df),
            "invalid_reason": combine_invalid_reasons(region_df),
            "n_frames": int(len(region_df)),
        })

    return pd.DataFrame(regions, columns=REGION_COLUMNS)


# =============================================================================
# MAIN EXPORT FUNCTION
# =============================================================================

def export_blink_filter_logs(
    video_path: str,
    output_dir: str,
    process_start_ms: float = 0.0,
    process_end_ms: Optional[float] = None,
    time_zero_ms: float = 0.0,
    warmup_ms: float = 2000.0,
    ap_code: Optional[str] = None,
    run_name: Optional[str] = None,
    vp_label: Optional[str] = None,
    trial_nr: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> None:
    """
    Process a video frame by frame and export blink-filter logs.

    Parameters
    ----------
    video_path:
        Input video path. It must match the camera/fps used by the R gaze data.

    output_dir:
        Folder in which the two CSV files are written.

    process_start_ms:
        First video time to export. Default: 0.

    process_end_ms:
        Stop exporting at this video time. Default: end of video.

    time_zero_ms:
        Zero point for exported ``time_ms``:
            time_ms = video_time_ms - time_zero_ms

        For a full-video log:
            time_zero_ms = 0

        For trial-relative output:
            time_zero_ms = trial start time in the full video

    warmup_ms:
        Number of milliseconds processed before ``process_start_ms`` to
        initialize adaptive EAR baselines. Warm-up frames are not exported.
        Default: 2000 ms.

    ap_code, run_name, vp_label, trial_nr:
        Optional metadata copied into the CSV files.

    max_frames:
        Optional number of exported frames for quick testing.
    """
    validate_arguments(
        process_start_ms=process_start_ms,
        process_end_ms=process_end_ms,
        time_zero_ms=time_zero_ms,
        warmup_ms=warmup_ms,
        max_frames=max_frames,
    )

    video_path_obj = Path(video_path).expanduser().resolve()
    output_dir_obj = Path(output_dir).expanduser().resolve()
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    inferred_ap_code, inferred_run_name = infer_metadata_from_output_dir(
        output_dir_obj
    )

    if ap_code is None:
        ap_code = inferred_ap_code

    if run_name is None:
        run_name = inferred_run_name

    cap = cv2.VideoCapture(str(video_path_obj))
    detector = None

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path_obj}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Invalid FPS. Cannot compute time columns.")

        if total_frames <= 0:
            raise RuntimeError("Video reports zero frames.")

        frame_duration_ms = 1000.0 / fps
        video_duration_ms = total_frames * frame_duration_ms

        if process_start_ms >= video_duration_ms:
            raise ValueError(
                f"--process-start-ms ({process_start_ms:.1f} ms) is outside "
                f"the video duration ({video_duration_ms:.1f} ms)."
            )

        export_start_frame = int(
            math.floor(process_start_ms / frame_duration_ms)
        )

        if process_end_ms is None:
            export_end_frame = total_frames
        else:
            export_end_frame = int(
                math.ceil(process_end_ms / frame_duration_ms)
            )
            export_end_frame = min(export_end_frame, total_frames)

        if max_frames is not None:
            export_end_frame = min(
                export_end_frame,
                export_start_frame + max_frames,
            )

        if export_end_frame <= export_start_frame:
            raise ValueError("The selected frame range is empty.")

        warmup_start_ms = max(0.0, process_start_ms - warmup_ms)
        warmup_start_frame = int(
            math.floor(warmup_start_ms / frame_duration_ms)
        )

        cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start_frame)

        detector = RobustPupilDetector()
        reset_detector_state(detector)

        rows: List[Dict[str, Any]] = []

        print("\n=== Export blink filter logs ===")
        print(f"Video:          {video_path_obj}")
        print(f"Output dir:     {output_dir_obj}")
        print(f"FPS:            {fps:.3f}")
        print(
            f"Warm-up frames: {warmup_start_frame} "
            f"to {export_start_frame - 1}"
        )
        print(
            f"Export frames:  {export_start_frame} "
            f"to {export_end_frame - 1}"
        )
        print(f"time_zero:      {time_zero_ms:.1f} ms")
        print("Processing...\n")

        for frame_number in range(warmup_start_frame, export_end_frame):
            ret, frame = cap.read()

            if not ret:
                print(
                    f"[WARN] Video read stopped at frame {frame_number}.",
                    file=sys.stderr,
                )
                break

            result = detector.extract_from_frame(frame)

            # Warm-up frames update temporal detector state, but are not exported.
            if frame_number < export_start_frame:
                continue

            if "valid_eye_frame" not in result:
                raise KeyError(
                    "Detector result is missing required field "
                    f"'valid_eye_frame' at frame {frame_number}. "
                    "Check the shared_pupil_detection interface."
                )

            video_time_ms = frame_number * frame_duration_ms
            time_ms = video_time_ms - time_zero_ms

            valid_eye_frame = safe_bool(result["valid_eye_frame"])
            invalid_reason = result.get("invalid_reason")

            reason_group = (
                "valid"
                if valid_eye_frame
                else map_reason_group(invalid_reason)
            )

            position = result.get("position")

            rows.append({
                "ap_code": ap_code,
                "run_name": run_name,
                "vp_label": vp_label,
                "trial_nr": trial_nr,
                "video_path": str(video_path_obj),
                "video_fps": fps,
                "frame_duration_ms": frame_duration_ms,

                "frame_number": int(frame_number),
                "video_time_ms": float(video_time_ms),
                "time_ms": float(time_ms),

                "valid_eye_frame": valid_eye_frame,
                "invalid_reason": invalid_reason,
                "reason_group": reason_group,

                "is_blink": safe_bool(result.get("is_blink")),
                "blink_count": result.get("blink_count"),
                "frames_below_threshold": result.get(
                    "frames_below_threshold"
                ),
                "eyes_closed": safe_bool(result.get("eyes_closed")),
                "is_suspected_blink": safe_bool(
                    result.get("is_suspected_blink")
                ),
                "is_transition_frame": safe_bool(
                    result.get("is_transition_frame")
                ),

                # Raw EAR explains rapid closure decisions.
                "left_ear_raw": result.get("left_ear_raw"),
                "right_ear_raw": result.get("right_ear_raw"),

                # left_ear/right_ear are the temporally smoothed values.
                "left_ear": result.get("left_ear"),
                "right_ear": result.get("right_ear"),
                "avg_ear": result.get("avg_ear"),

                "left_threshold": result.get("left_threshold"),
                "right_threshold": result.get("right_threshold"),

                "left_closed": safe_bool(result.get("left_closed")),
                "right_closed": safe_bool(result.get("right_closed")),
                "left_closed_raw": safe_bool(
                    result.get("left_closed_raw")
                ),
                "right_closed_raw": safe_bool(
                    result.get("right_closed_raw")
                ),
                "left_closed_smooth": safe_bool(
                    result.get("left_closed_smooth")
                ),
                "right_closed_smooth": safe_bool(
                    result.get("right_closed_smooth")
                ),

                "ear_asymmetry": result.get("ear_asymmetry"),
                "baseline_samples": result.get("baseline_samples"),

                "position_x": safe_get_position_component(position, 0),
                "position_y": safe_get_position_component(position, 1),
                "confidence": result.get("confidence"),
                "plausibility_passed": safe_bool(
                    result.get("plausibility_passed")
                ),
            })

            if len(rows) % 500 == 0:
                print(f"  exported {len(rows)} frames...")

        frame_log = pd.DataFrame(rows, columns=FRAME_LOG_COLUMNS)

        frame_log_path = (
            output_dir_obj / "blink_filter_frame_log.csv"
        )
        frame_log.to_csv(frame_log_path, index=False)

        regions = build_invalid_regions(
            frame_log=frame_log,
            frame_duration_ms=frame_duration_ms,
        )

        regions_path = output_dir_obj / "blink_filter_regions.csv"
        regions.to_csv(regions_path, index=False)

        print("\n=== Done ===")
        print(f"Frame log: {frame_log_path}")
        print(f"Regions:   {regions_path}")

        if frame_log.empty:
            print("\n[WARN] No frames were exported.")
        else:
            print("\nFrame summary:")
            print(
                frame_log["valid_eye_frame"].value_counts(
                    dropna=False
                )
            )

            print("\nInvalid reason summary:")
            invalid_only = frame_log.loc[
                frame_log["valid_eye_frame"].eq(False)
            ]

            if invalid_only.empty:
                print("No invalid frames.")
            else:
                print(
                    invalid_only["invalid_reason"].value_counts(
                        dropna=False
                    )
                )

        print("\nRegion summary:")
        if regions.empty:
            print("No invalid regions.")
        else:
            print(
                regions["reason_group"].value_counts(dropna=False)
            )

    finally:
        cap.release()

        if detector is not None:
            close_method = getattr(detector, "close", None)
            if callable(close_method):
                close_method()


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export blink-filter frame logs and merged invalid regions "
            "for Figure 7 overlays."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which the two CSV files are written.",
    )

    parser.add_argument(
        "--process-start-ms",
        type=float,
        default=0.0,
        help=(
            "First video time to export, in ms. "
            "Default: 0."
        ),
    )

    parser.add_argument(
        "--process-end-ms",
        type=float,
        default=None,
        help=(
            "Stop exporting at this video time, in ms. "
            "Default: end of video."
        ),
    )

    parser.add_argument(
        "--time-zero-ms",
        type=float,
        default=0.0,
        help=(
            "Zero point for exported time_ms. "
            "For trial-relative output, use the trial start time. "
            "Default: 0."
        ),
    )

    parser.add_argument(
        "--warmup-ms",
        type=float,
        default=2000.0,
        help=(
            "Process this much video before process-start-ms to "
            "initialize the adaptive EAR baseline. Warm-up frames are "
            "not exported. Default: 2000."
        ),
    )

    parser.add_argument(
        "--ap-code",
        default=None,
        help=(
            "Optional AP/VP code, e.g. beo7. "
            "Inferred from output-dir when possible."
        ),
    )

    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Optional run name. "
            "Inferred from output-dir when omitted."
        ),
    )

    parser.add_argument(
        "--vp-label",
        default=None,
        help='Optional R label, e.g. "VP 8".',
    )

    parser.add_argument(
        "--trial-nr",
        default=None,
        help="Optional experiment trial number.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional exported-frame limit for quick testing.",
    )

    args = parser.parse_args()

    export_blink_filter_logs(
        video_path=args.video,
        output_dir=args.output_dir,
        process_start_ms=args.process_start_ms,
        process_end_ms=args.process_end_ms,
        time_zero_ms=args.time_zero_ms,
        warmup_ms=args.warmup_ms,
        ap_code=args.ap_code,
        run_name=args.run_name,
        vp_label=args.vp_label,
        trial_nr=args.trial_nr,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()