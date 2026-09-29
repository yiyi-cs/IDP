"""
EyeLink X-only I-DT validation.

Input:
    debug_3_eyetracker_data.csv   -> raw EyeLink gaze samples
    debug_3_fixations.csv         -> EyeLink native fixation events

Purpose:
    Validate our X-only I-DT implementation on high-quality EyeLink gaze
    before applying the same detector to webcam/PTGaze gaze.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sys
# Allow imports from Skripte/
_PROJECT_ROOT = Path(__file__).parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# =============================================================================
# PARAMETERS
# =============================================================================

MIN_FIXATION_DURATION_MS = 100.0

# Smaller thresholds than PTGaze are included because EyeLink should be
# considerably less noisy.
THRESHOLDS_DEG = [
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
]


# =============================================================================
# SCREEN GEOMETRY
# =============================================================================
#
# IMPORTANT:
# Set these to the SAME values used in your existing config.py.
#
# Do not guess new screen geometry here.
#

from config import (
    SCREEN_WIDTH_PX,
    SCREEN_WIDTH_CM,
    VIEWING_DISTANCE_CM,
)


# =============================================================================
# PIXEL -> VISUAL ANGLE
# =============================================================================

def pixel_x_to_degree(x_px):

    screen_center_px = SCREEN_WIDTH_PX / 2.0

    pixels_per_cm = (
        SCREEN_WIDTH_PX
        / SCREEN_WIDTH_CM
    )

    offset_cm = (
        x_px - screen_center_px
    ) / pixels_per_cm

    degree = (
        np.arctan2(
            offset_cm,
            VIEWING_DISTANCE_CM
        )
        * 180.0
        / np.pi
    )

    return degree


# =============================================================================
# I-DT
# =============================================================================

def x_dispersion(x):
    return np.max(x) - np.min(x)


def detect_idt(
    samples,
    threshold_deg,
):

    samples = (
        samples
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
    )

    t = samples[
        "timestamp_ms"
    ].to_numpy()

    x = samples[
        "gaze_x_deg"
    ].to_numpy()

    n = len(samples)

    fixations = []

    i = 0
    fixation_id = 0

    while i < n:

        # -------------------------------------------------------------
        # Minimum 100 ms window
        # -------------------------------------------------------------

        target_time = (
            t[i]
            + MIN_FIXATION_DURATION_MS
        )

        j = np.searchsorted(
            t,
            target_time,
            side="left",
        )

        if j >= n:
            break

        # -------------------------------------------------------------
        # Test initial dispersion
        # -------------------------------------------------------------

        dispersion = x_dispersion(
            x[i:j + 1]
        )

        if dispersion > threshold_deg:
            i += 1
            continue

        # -------------------------------------------------------------
        # Extend fixation
        # -------------------------------------------------------------

        k = j

        while k + 1 < n:

            new_dispersion = x_dispersion(
                x[i:k + 2]
            )

            if new_dispersion > threshold_deg:
                break

            k += 1

        fixation_id += 1

        fixations.append({

            "fixation_id":
                fixation_id,

            "start_time":
                t[i],

            "end_time":
                t[k],

            "duration":
                t[k] - t[i],

            "x_pos_deg":
                np.mean(
                    x[i:k + 1]
                ),

            "dispersion_x_deg":
                x_dispersion(
                    x[i:k + 1]
                ),

            "n_samples":
                k - i + 1,
        })

        i = k + 1

    return pd.DataFrame(
        fixations
    )


# =============================================================================
# OVERLAP
# =============================================================================

def interval_overlap(
    a_start,
    a_end,
    b_start,
    b_end,
):

    return max(
        0.0,
        min(a_end, b_end)
        - max(a_start, b_start)
    )


def interval_iou(
    a_start,
    a_end,
    b_start,
    b_end,
):

    intersection = interval_overlap(
        a_start,
        a_end,
        b_start,
        b_end,
    )

    if intersection <= 0:
        return 0.0

    union = (
        max(a_end, b_end)
        - min(a_start, b_start)
    )

    if union <= 0:
        return 0.0

    return intersection / union


def compare_events(
    native,
    detected,
):

    """
    For every native EyeLink fixation, find the detected I-DT fixation
    with the highest temporal IoU.
    """

    rows = []

    for _, gt in native.iterrows():

        best_iou = 0.0
        best_overlap = 0.0

        n_overlapping = 0

        for _, pred in detected.iterrows():

            overlap = interval_overlap(
                gt["start_time"],
                gt["end_time"],
                pred["start_time"],
                pred["end_time"],
            )

            if overlap <= 0:
                continue

            n_overlapping += 1

            iou = interval_iou(
                gt["start_time"],
                gt["end_time"],
                pred["start_time"],
                pred["end_time"],
            )

            if iou > best_iou:
                best_iou = iou
                best_overlap = overlap

        rows.append({

            "native_start":
                gt["start_time"],

            "native_end":
                gt["end_time"],

            "native_duration":
                gt["duration"],

            "best_iou":
                best_iou,

            "best_overlap_ms":
                best_overlap,

            "n_overlapping_idt":
                n_overlapping,
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "eyelink_samples_csv",
        help="debug_3_eyetracker_data.csv",
    )

    parser.add_argument(
        "eyelink_fixations_csv",
        help="debug_3_fixations.csv",
    )

    args = parser.parse_args()

    sample_path = Path(
        args.eyelink_samples_csv
    )

    fixation_path = Path(
        args.eyelink_fixations_csv
    )

    if not sample_path.exists():
        raise FileNotFoundError(
            sample_path
        )

    if not fixation_path.exists():
        raise FileNotFoundError(
            fixation_path
        )

    print("=" * 70)
    print("EYELINK X-ONLY I-DT VALIDATION")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    samples = pd.read_csv(
        sample_path
    )

    native = pd.read_csv(
        fixation_path
    )

    # Remove practice native fixations
    if "is_practice" in native.columns:

        native = native[
            ~native[
                "is_practice"
            ]
            .fillna(False)
            .astype(bool)
        ].copy()

    # -----------------------------------------------------------------
    # Remove invalid EyeLink samples
    # -----------------------------------------------------------------

    samples = samples.dropna(
        subset=[
            "timestamp_ms",
            "x_pos",
        ]
    ).copy()

    # -----------------------------------------------------------------
    # Pixel -> degree
    # -----------------------------------------------------------------

    samples["gaze_x_deg"] = (
        pixel_x_to_degree(
            samples["x_pos"].to_numpy()
        )
    )

    print(
        f"\nEyeLink samples: "
        f"{len(samples)}"
    )

    print(
        f"Native fixations: "
        f"{len(native)}"
    )

    print(
        f"\nScreen geometry:"
        f"\n  width = {SCREEN_WIDTH_PX} px"
        f"\n  width = {SCREEN_WIDTH_CM} cm"
        f"\n  viewing distance = {VIEWING_DISTANCE_CM} cm"
    )

    # -----------------------------------------------------------------
    # Sweep
    # -----------------------------------------------------------------

    output_dir = (
        sample_path.parent
    )

    summary = []

    for threshold in THRESHOLDS_DEG:

        detected = detect_idt(
            samples,
            threshold_deg=threshold,
        )

        comparison = compare_events(
            native,
            detected,
        )

        threshold_name = (
            str(threshold)
            .replace(".", "p")
        )

        detected.to_csv(
            output_dir
            / (
                "debug_6_eyelink_idt_"
                f"{threshold_name}deg.csv"
            ),
            index=False,
        )

        comparison.to_csv(
            output_dir
            / (
                "debug_6_eyelink_idt_"
                f"{threshold_name}deg_comparison.csv"
            ),
            index=False,
        )

        matched = (
            comparison[
                "best_iou"
            ] > 0
        )

        summary.append({

            "threshold_deg":
                threshold,

            "native_fixations":
                len(native),

            "idt_fixations":
                len(detected),

            "idt_mean_duration_ms":
                (
                    detected["duration"].mean()
                    if len(detected)
                    else np.nan
                ),

            "idt_median_duration_ms":
                (
                    detected["duration"].median()
                    if len(detected)
                    else np.nan
                ),

            "native_with_overlap_pct":
                100 * matched.mean(),

            "mean_best_iou":
                comparison[
                    "best_iou"
                ].mean(),

            "median_best_iou":
                comparison[
                    "best_iou"
                ].median(),

            "iou_ge_0_3_pct":
                100 * (
                    comparison[
                        "best_iou"
                    ] >= 0.3
                ).mean(),

            "iou_ge_0_5_pct":
                100 * (
                    comparison[
                        "best_iou"
                    ] >= 0.5
                ).mean(),

            "iou_ge_0_7_pct":
                100 * (
                    comparison[
                        "best_iou"
                    ] >= 0.7
                ).mean(),
        })

    summary_df = pd.DataFrame(
        summary
    )

    print()
    print("=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)
    print()

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    output_path = (
        output_dir
        / "debug_6_eyelink_idt_validation_summary.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()