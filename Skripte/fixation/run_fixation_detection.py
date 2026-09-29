"""
Run the integrated fixation detector on one existing analysis run.

Inputs:
    1. debug_5_ptgaze_calibrated.csv
    2. blink_filter_frame_log.csv

Outputs:
    1. debug_6_ptgaze_fixations.csv
    2. debug_6_fixation_metadata.json

Important:
    - Blink validity is merged by frame == frame_number.
    - timestamp_ms_synced is never modified.
"""

import argparse
import json
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
# INPUT PREPARATION
# =============================================================================

def prepare_gaze_data(
    ptgaze_path: Path,
    blink_path: Path,
) -> pd.DataFrame:

    pt = pd.read_csv(ptgaze_path)
    blink = pd.read_csv(blink_path)

    # -------------------------------------------------------------------------
    # Required columns
    # -------------------------------------------------------------------------

    required_pt = [
        "frame",
        "timestamp_ms_synced",
        "trial_number",
        "phase_type",
        "gaze_deg_x_calib",
    ]

    required_blink = [
        "frame_number",
        "valid_eye_frame",
    ]

    missing_pt = [
        col for col in required_pt
        if col not in pt.columns
    ]

    missing_blink = [
        col for col in required_blink
        if col not in blink.columns
    ]

    if missing_pt:
        raise ValueError(
            f"PTGaze missing columns: {missing_pt}"
        )

    if missing_blink:
        raise ValueError(
            f"Blink log missing columns: {missing_blink}"
        )

    # -------------------------------------------------------------------------
    # Merge final blink / invalid-eye-frame mask
    #
    # IMPORTANT:
    #   gaze.frame == blink.frame_number
    #
    # Do NOT modify timestamp_ms_synced.
    # -------------------------------------------------------------------------

    blink_small = blink[
        [
            "frame_number",
            "valid_eye_frame",
        ]
    ].copy()

    merged = pt.merge(
        blink_small,
        left_on="frame",
        right_on="frame_number",
        how="left",
        validate="one_to_one",
    )

    matched = (
        merged["valid_eye_frame"]
        .notna()
        .sum()
    )

    match_ratio = (
        matched / len(merged)
        if len(merged)
        else 0.0
    )

    print(
        f"Blink mask matched: "
        f"{matched}/{len(merged)} "
        f"({100 * match_ratio:.1f}%)"
    )

    # Missing blink information is NOT silently treated as valid.
    merged["valid_eye_frame"] = (
        merged["valid_eye_frame"]
        .astype("boolean")
    )

    return merged


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "ptgaze_csv",
        help="Path to debug_5_ptgaze_calibrated.csv",
    )

    parser.add_argument(
        "blink_log_csv",
        help="Path to blink_filter_frame_log.csv",
    )

    args = parser.parse_args()

    ptgaze_path = Path(args.ptgaze_csv)
    blink_path = Path(args.blink_log_csv)

    if not ptgaze_path.exists():
        raise FileNotFoundError(ptgaze_path)

    if not blink_path.exists():
        raise FileNotFoundError(blink_path)

    print("=" * 70)
    print("INTEGRATED FIXATION DETECTION")
    print("=" * 70)

    print(f"\nPTGaze: {ptgaze_path}")
    print(f"Blink:   {blink_path}")

    # -------------------------------------------------------------------------
    # Prepare input
    # -------------------------------------------------------------------------

    gaze = prepare_gaze_data(
        ptgaze_path,
        blink_path,
    )

    # -------------------------------------------------------------------------
    # Final configuration
    # -------------------------------------------------------------------------

    config = FixationConfig(
        smoothing_window_ms=100.0,
        dispersion_window_ms=100.0,
        min_fixation_duration_ms=100.0,
        adaptive_percentile=75.0,
        max_gap_ms=50.0,
    )

    # -------------------------------------------------------------------------
    # Detect
    # -------------------------------------------------------------------------

    result = detect_fixations(
        gaze,
        config=config,
    )

    fixations = result.fixations

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

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

    noise = result.noise_distribution

    print(
        f"Noise windows: "
        f"{len(noise)}"
    )

    print(
        f"Noise P50/P75/P90: "
        f"{np.percentile(noise, 50):.3f} / "
        f"{np.percentile(noise, 75):.3f} / "
        f"{np.percentile(noise, 90):.3f} deg"
    )

    # -------------------------------------------------------------------------
    # Save fixation events
    # -------------------------------------------------------------------------

    output_dir = ptgaze_path.parent

    fixation_path = (
        output_dir
        / "debug_6_ptgaze_fixations.csv"
    )

    fixations.to_csv(
        fixation_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------------------

    metadata = {

        "method":
            "I-DT",

        "axis":
            "horizontal_x_only",

        "smoothing":
            "centered_time_median",

        "smoothing_window_ms":
            config.smoothing_window_ms,

        "dispersion_window_ms":
            config.dispersion_window_ms,

        "minimum_fixation_duration_ms":
            config.min_fixation_duration_ms,

        "adaptive_threshold_method":
            f"P{config.adaptive_percentile:g}_known_fixation_phase",

        "adaptive_threshold_deg":
            result.threshold_deg,

        "max_invalid_gap_ms":
            config.max_gap_ms,

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

        "timestamp_source":
            "timestamp_ms_synced",

        "blink_merge":
            "frame == frame_number",
    }

    metadata_path = (
        output_dir
        / "debug_6_fixation_metadata.json"
    )

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

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(fixation_path)
    print(metadata_path)


if __name__ == "__main__":
    main()