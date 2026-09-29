"""
Integrated fixation detection runner.

Reads calibrated gaze directly from the current pipeline run.

Supported inputs:
    MediaPipe:
        debug_5_pupil_data_calibrated.csv

    PTGaze:
        debug_5_ptgaze_calibrated.csv

Both use the same fixation detector:
    - X-only
    - 100 ms centered time median
    - participant/method-specific P75 threshold
    - 100 ms minimum fixation duration
    - 50 ms maximum invalid gap

No external blink CSV is required.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT IMPORT
# =============================================================================

_PROJECT_ROOT = Path(__file__).parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from fixation.fixation_detector import (
    FixationConfig,
    detect_fixations,
)


# =============================================================================
# PIPELINE PATH
# =============================================================================

if "PIPELINE_OUTPUT_BASE_DIR" in os.environ:
    OUTPUT_BASE_DIR = Path(
        os.environ["PIPELINE_OUTPUT_BASE_DIR"]
    )
else:
    # Manual execution:
    # python fixation/run_fixation_detection.py /path/to/run
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage:\n"
            "  python fixation/run_fixation_detection.py "
            "/path/to/Run_xxx"
        )

    OUTPUT_BASE_DIR = Path(
        sys.argv[1]
    )


# =============================================================================
# FINAL FROZEN CONFIG
# =============================================================================

CONFIG = FixationConfig(
    smoothing_window_ms=100.0,
    dispersion_window_ms=100.0,
    min_fixation_duration_ms=100.0,
    adaptive_percentile=75.0,
    max_gap_ms=50.0,
)


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_input(
    df: pd.DataFrame,
    method: str,
):

    required = [
        "timestamp_ms_synced",
        "phase_type",
        "gaze_deg_x_calib",
        "valid_eye_frame",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if (
        "trial_number" not in df.columns
        and
        "trial_assignment" not in df.columns
    ):
        missing.append(
            "trial_number/trial_assignment"
        )

    if missing:

        raise ValueError(
            f"{method}: missing required columns: "
            f"{missing}"
        )


# =============================================================================
# RUN ONE GAZE METHOD
# =============================================================================

def run_method(
    method: str,
    input_path: Path,
    output_prefix: str,
):

    print()
    print("=" * 70)
    print(
        f"FIXATION DETECTION: "
        f"{method.upper()}"
    )
    print("=" * 70)

    print(
        f"Input: {input_path.name}"
    )

    df = pd.read_csv(
        input_path
    )

    validate_input(
        df,
        method,
    )

    # -----------------------------------------------------------------
    # Input quality
    # -----------------------------------------------------------------

    valid_eye = (
        df["valid_eye_frame"]
        .fillna(False)
        .astype(bool)
    )

    print(
        f"Samples: {len(df)}"
    )

    print(
        f"Valid eye frames: "
        f"{valid_eye.sum()}/{len(df)} "
        f"({100 * valid_eye.mean():.1f}%)"
    )

    # -----------------------------------------------------------------
    # Detection
    # -----------------------------------------------------------------

    result = detect_fixations(
        df,
        config=CONFIG,
    )

    fixations = (
        result.fixations
    )

    noise = (
        result.noise_distribution
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    print(
        f"Adaptive threshold: "
        f"{result.threshold_deg:.3f} deg"
    )

    print(
        f"Fixations detected: "
        f"{len(fixations)}"
    )

    if len(fixations):

        print(
            f"Mean duration: "
            f"{fixations['duration'].mean():.1f} ms"
        )

        print(
            f"Median duration: "
            f"{fixations['duration'].median():.1f} ms"
        )

    print(
        f"Noise P50/P75/P90: "
        f"{np.percentile(noise, 50):.3f} / "
        f"{np.percentile(noise, 75):.3f} / "
        f"{np.percentile(noise, 90):.3f} deg"
    )

    # -----------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------

    fixation_path = (
        OUTPUT_BASE_DIR
        / f"{output_prefix}_fixations.csv"
    )

    metadata_path = (
        OUTPUT_BASE_DIR
        / f"{output_prefix}_fixation_metadata.json"
    )

    fixations.to_csv(
        fixation_path,
        index=False,
    )

    metadata = {

        "gaze_method":
            method,

        "input_file":
            input_path.name,

        "method":
            "I-DT",

        "axis":
            "horizontal_x_only",

        "smoothing":
            "centered_time_median",

        "smoothing_window_ms":
            CONFIG.smoothing_window_ms,

        "dispersion_window_ms":
            CONFIG.dispersion_window_ms,

        "minimum_fixation_duration_ms":
            CONFIG.min_fixation_duration_ms,

        "adaptive_threshold_method":
            f"P{CONFIG.adaptive_percentile:g}_known_fixation_phase",

        "adaptive_threshold_deg":
            result.threshold_deg,

        "max_invalid_gap_ms":
            CONFIG.max_gap_ms,

        "timestamp_source":
            "timestamp_ms_synced",

        "eye_validity_source":
            (
                "mediapipe"
                if method == "mediapipe"
                else "mediapipe_propagated"
            ),

        "n_samples":
            len(df),

        "n_valid_eye_frames":
            int(
                valid_eye.sum()
            ),

        "valid_eye_frame_pct":
            float(
                100
                * valid_eye.mean()
            ),

        "n_noise_windows":
            len(noise),

        "noise_p50_deg":
            float(
                np.percentile(
                    noise,
                    50,
                )
            ),

        "noise_p75_deg":
            float(
                np.percentile(
                    noise,
                    75,
                )
            ),

        "noise_p90_deg":
            float(
                np.percentile(
                    noise,
                    90,
                )
            ),

        "n_fixations":
            len(fixations),

        "mean_fixation_duration_ms":
            (
                float(
                    fixations[
                        "duration"
                    ].mean()
                )
                if len(fixations)
                else None
            ),

        "median_fixation_duration_ms":
            (
                float(
                    fixations[
                        "duration"
                    ].median()
                )
                if len(fixations)
                else None
            ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"Saved: {fixation_path.name}"
    )

    print(
        f"Saved: {metadata_path.name}"
    )

    return True


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 70)
    print("INTEGRATED FIXATION DETECTION")
    print("=" * 70)

    print(
        f"Run folder: "
        f"{OUTPUT_BASE_DIR}"
    )

    # -----------------------------------------------------------------
    # Available calibrated gaze methods
    # -----------------------------------------------------------------

    methods = [

        (
            "mediapipe",
            OUTPUT_BASE_DIR
            / "debug_5_pupil_data_calibrated.csv",
            "debug_6_mediapipe",
        ),

        (
            "ptgaze",
            OUTPUT_BASE_DIR
            / "debug_5_ptgaze_calibrated.csv",
            "debug_6_ptgaze",
        ),
    ]

    completed = []
    skipped = []
    failed = []

    for (
        method,
        input_path,
        output_prefix,
    ) in methods:

        if not input_path.exists():

            print()
            print(
                f"[SKIP] {method}: "
                f"{input_path.name} not found"
            )

            skipped.append(
                method
            )

            continue

        try:

            run_method(
                method=method,
                input_path=input_path,
                output_prefix=output_prefix,
            )

            completed.append(
                method
            )

        except Exception as e:

            print()
            print(
                f"[FAILED] {method}: {e}"
            )

            failed.append(
                (
                    method,
                    str(e),
                )
            )

    # -----------------------------------------------------------------
    # Final status
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("STATUS")
    print("=" * 70)

    print(
        "Completed: "
        + (
            ", ".join(completed)
            if completed
            else "none"
        )
    )

    print(
        "Skipped: "
        + (
            ", ".join(skipped)
            if skipped
            else "none"
        )
    )

    print(
        "Failed: "
        + (
            ", ".join(
                method
                for method, _
                in failed
            )
            if failed
            else "none"
        )
    )

    if failed:
        raise RuntimeError(
            "One or more fixation "
            "detection methods failed."
        )

    if not completed:
        raise RuntimeError(
            "No calibrated gaze input found."
        )


if __name__ == "__main__":
    main()