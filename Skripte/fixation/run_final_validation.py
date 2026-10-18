"""
Final validation for integrated webcam fixation detection.

For each VP:
    1. Load original calibrated PTGaze gaze.
    2. Merge final valid_eye_frame mask by frame.
    3. Run the integrated fixation detector:
         - X only
         - 100 ms centered time median
         - adaptive P75 threshold
         - minimum fixation duration = 100 ms
         - max invalid gap = 50 ms
    4. Save final fixation events + metadata.
    5. Compare against native EyeLink fixation events.

IMPORTANT:
    This is final validation, not parameter tuning.
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
# CONFIG
# =============================================================================

VPS = [
    "beo7",
    "bjs4",
    "egf5",
    "fbn6",
    "fgt6",
    "jkl7",
    "kdn8",
    "kro3",
    "mhe9",
    "oem4",
    "ogt7",
]


FINAL_CONFIG = FixationConfig(
    smoothing_window_ms=100.0,
    dispersion_window_ms=100.0,
    min_fixation_duration_ms=100.0,
    adaptive_percentile=75.0,
    max_gap_ms=50.0,
)


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_files(vp_dir):

    analyse_dir = vp_dir / "Analyse"

    candidates = sorted(
        analyse_dir.glob(
            "Run_60hz_FullCalib_mode2_*"
        )
    )

    # Do not use derived experimental runs.
    candidates = [
        p for p in candidates
        if (
            "BlinkFiltered" not in p.name
            and "PTS" not in p.name
        )
    ]

    valid_runs = []

    for run in candidates:

        ptgaze = (
            run
            / "debug_5_ptgaze_calibrated.csv"
        )

        eyelink = (
            run
            / "debug_3_fixations.csv"
        )

        if (
            ptgaze.exists()
            and eyelink.exists()
        ):
            valid_runs.append(
                (
                    run,
                    ptgaze,
                    eyelink,
                )
            )

    if not valid_runs:
        raise FileNotFoundError(
            f"No valid original run in {analyse_dir}"
        )

    # Same selection principle as previous batch experiment.
    run, ptgaze, eyelink = valid_runs[0]

    blink = (
        vp_dir
        / "test60"
        / "blink_filter_frame_log.csv"
    )

    if not blink.exists():
        raise FileNotFoundError(
            blink
        )

    return (
        run,
        ptgaze,
        eyelink,
        blink,
    )


# =============================================================================
# INPUT PREPARATION
# =============================================================================

def prepare_gaze(
    ptgaze_path,
    blink_path,
):

    pt = pd.read_csv(
        ptgaze_path
    )

    blink = pd.read_csv(
        blink_path
    )

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
        merged[
            "valid_eye_frame"
        ]
        .notna()
        .sum()
    )

    match_pct = (
        100 * matched / len(merged)
        if len(merged)
        else 0.0
    )

    merged["valid_eye_frame"] = (
        merged[
            "valid_eye_frame"
        ]
        .astype("boolean")
    )

    return (
        merged,
        match_pct,
    )


# =============================================================================
# EYELINK
# =============================================================================

def load_eyelink(
    path,
):

    df = pd.read_csv(
        path
    )

    if "is_practice" in df.columns:

        df = df[
            ~df[
                "is_practice"
            ]
            .fillna(False)
            .astype(bool)
        ].copy()

    return df


# =============================================================================
# TEMPORAL IoU
# =============================================================================

def interval_iou(
    a_start,
    a_end,
    b_start,
    b_end,
):

    intersection = max(
        0.0,
        min(a_end, b_end)
        - max(a_start, b_start)
    )

    if intersection <= 0:
        return 0.0

    union = (
        max(a_end, b_end)
        - min(a_start, b_start)
    )

    if union <= 0:
        return 0.0

    return (
        intersection / union
    )


def evaluate_against_eyelink(
    eyelink,
    detected,
):

    """
    For every native EyeLink fixation, find the temporally
    best-matching detected fixation in the same trial.
    """

    best_ious = []

    for _, gt in (
        eyelink.iterrows()
    ):

        candidates = detected[
            detected[
                "trial_number"
            ]
            == gt["trial_number"]
        ]

        best = 0.0

        for _, pred in (
            candidates.iterrows()
        ):

            # No temporal overlap
            if (
                pred["end_time"]
                <= gt["start_time"]
                or
                pred["start_time"]
                >= gt["end_time"]
            ):
                continue

            iou = interval_iou(
                gt["start_time"],
                gt["end_time"],
                pred["start_time"],
                pred["end_time"],
            )

            if iou > best:
                best = iou

        best_ious.append(
            best
        )

    best_ious = np.asarray(
        best_ious,
        dtype=float,
    )

    return {

        "eyelink_overlap_pct":
            100
            * (
                best_ious > 0
            ).mean(),

        "mean_best_iou":
            best_ious.mean(),

        "median_best_iou":
            np.median(
                best_ious
            ),

        "iou_ge_0_3_pct":
            100
            * (
                best_ious >= 0.3
            ).mean(),

        "iou_ge_0_5_pct":
            100
            * (
                best_ious >= 0.5
            ).mean(),

        "iou_ge_0_7_pct":
            100
            * (
                best_ious >= 0.7
            ).mean(),
    }


# =============================================================================
# SAVE METADATA
# =============================================================================

def save_metadata(
    path,
    result,
    fixations,
    blink_match_pct,
):

    noise = (
        result.noise_distribution
    )

    config = result.config

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

        "timestamp_source":
            "timestamp_ms_synced",

        "blink_merge":
            "frame == frame_number",

        "blink_match_pct":
            blink_match_pct,

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
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


# =============================================================================
# ONE VP
# =============================================================================

def process_vp(
    vp,
    root,
):

    vp_dir = (
        root / vp
    )

    (
        run,
        ptgaze_path,
        eyelink_path,
        blink_path,
    ) = find_files(
        vp_dir
    )

    # -----------------------------------------------------------------
    # Prepare final detector input
    # -----------------------------------------------------------------

    gaze, blink_match_pct = (
        prepare_gaze(
            ptgaze_path,
            blink_path,
        )
    )

    # -----------------------------------------------------------------
    # Run the actual integrated detector
    # -----------------------------------------------------------------

    result = detect_fixations(
        gaze,
        config=FINAL_CONFIG,
    )

    detected = (
        result.fixations
    )

    # -----------------------------------------------------------------
    # Save final integrated outputs
    # -----------------------------------------------------------------

    fixation_path = (
        run
        / "debug_6_ptgaze_fixations.csv"
    )

    metadata_path = (
        run
        / "debug_6_fixation_metadata.json"
    )

    detected.to_csv(
        fixation_path,
        index=False,
    )

    save_metadata(
        metadata_path,
        result,
        detected,
        blink_match_pct,
    )

    # -----------------------------------------------------------------
    # EyeLink reference
    # -----------------------------------------------------------------

    eyelink = load_eyelink(
        eyelink_path
    )

    metrics = (
        evaluate_against_eyelink(
            eyelink,
            detected,
        )
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    return {

        "vp":
            vp,

        "run":
            run.name,

        "blink_match_pct":
            blink_match_pct,

        "adaptive_threshold_deg":
            result.threshold_deg,

        "noise_p50_deg":
            float(
                np.percentile(
                    result.noise_distribution,
                    50,
                )
            ),

        "noise_p75_deg":
            float(
                np.percentile(
                    result.noise_distribution,
                    75,
                )
            ),

        "noise_p90_deg":
            float(
                np.percentile(
                    result.noise_distribution,
                    90,
                )
            ),

        "eyelink_fixations":
            len(eyelink),

        "detected_fixations":
            len(detected),

        "detected_mean_duration_ms":
            (
                detected[
                    "duration"
                ].mean()
                if len(detected)
                else np.nan
            ),

        "detected_median_duration_ms":
            (
                detected[
                    "duration"
                ].median()
                if len(detected)
                else np.nan
            ),

        **metrics,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "results_root",
        help=(
            "Example: "
            "/Volumes/Empra10/Ergebnisse60"
        ),
    )

    args = parser.parse_args()

    root = Path(
        args.results_root
    )

    if not root.exists():
        raise FileNotFoundError(
            root
        )

    print("=" * 70)
    print("FINAL FIXATION VALIDATION")
    print("=" * 70)

    print()
    print("Frozen method:")
    print("  Axis               : X only")
    print("  Smoothing          : centered median 100 ms")
    print("  Adaptive threshold : fixation-phase P75")
    print("  Min duration       : 100 ms")
    print("  Max invalid gap    : 50 ms")

    rows = []
    failures = []

    # =================================================================
    # Run all participants
    # =================================================================

    for vp in VPS:

        print()
        print(f"[{vp}]")

        try:

            row = process_vp(
                vp,
                root,
            )

            rows.append(
                row
            )

            print(
                f"  threshold = "
                f"{row['adaptive_threshold_deg']:.3f} deg"
            )

            print(
                f"  fixations = "
                f"{row['detected_fixations']} "
                f"(EyeLink {row['eyelink_fixations']})"
            )

            print(
                f"  median duration = "
                f"{row['detected_median_duration_ms']:.1f} ms"
            )

            print(
                f"  overlap = "
                f"{row['eyelink_overlap_pct']:.1f}%"
            )

            print(
                f"  mean best IoU = "
                f"{row['mean_best_iou']:.3f}"
            )

        except Exception as e:

            failures.append(
                (
                    vp,
                    str(e),
                )
            )

            print(
                f"  FAILED: {e}"
            )

    if not rows:

        raise RuntimeError(
            "No VP completed successfully."
        )

    results = pd.DataFrame(
        rows
    )

    # =================================================================
    # Per-VP table
    # =================================================================

    print()
    print("=" * 70)
    print("PER-VP FINAL RESULTS")
    print("=" * 70)
    print()

    display_columns = [
        "vp",
        "adaptive_threshold_deg",
        "eyelink_fixations",
        "detected_fixations",
        "detected_median_duration_ms",
        "eyelink_overlap_pct",
        "mean_best_iou",
        "median_best_iou",
        "iou_ge_0_3_pct",
        "iou_ge_0_5_pct",
        "iou_ge_0_7_pct",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # =================================================================
    # Overall aggregate
    # =================================================================

    aggregate = {

        "n_vps":
            results[
                "vp"
            ].nunique(),

        "mean_threshold_deg":
            results[
                "adaptive_threshold_deg"
            ].mean(),

        "median_threshold_deg":
            results[
                "adaptive_threshold_deg"
            ].median(),

        "mean_eyelink_fixations":
            results[
                "eyelink_fixations"
            ].mean(),

        "mean_detected_fixations":
            results[
                "detected_fixations"
            ].mean(),

        "mean_detected_duration_ms":
            results[
                "detected_mean_duration_ms"
            ].mean(),

        "mean_overlap_pct":
            results[
                "eyelink_overlap_pct"
            ].mean(),

        "mean_best_iou":
            results[
                "mean_best_iou"
            ].mean(),

        "mean_iou_ge_0_3_pct":
            results[
                "iou_ge_0_3_pct"
            ].mean(),

        "mean_iou_ge_0_5_pct":
            results[
                "iou_ge_0_5_pct"
            ].mean(),

        "mean_iou_ge_0_7_pct":
            results[
                "iou_ge_0_7_pct"
            ].mean(),
    }

    print()
    print("=" * 70)
    print("OVERALL FINAL RESULT")
    print("=" * 70)
    print()

    for key, value in (
        aggregate.items()
    ):

        if isinstance(
            value,
            (float, np.floating),
        ):
            print(
                f"{key}: {value:.3f}"
            )

        else:
            print(
                f"{key}: {value}"
            )

    # =================================================================
    # Save
    # =================================================================

    result_path = (
        root
        / "final_fixation_validation.csv"
    )

    aggregate_path = (
        root
        / "final_fixation_validation_summary.json"
    )

    results.to_csv(
        result_path,
        index=False,
    )

    with open(
        aggregate_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            aggregate,
            f,
            indent=2,
        )

    # =================================================================
    # Status
    # =================================================================

    print()
    print("=" * 70)
    print("STATUS")
    print("=" * 70)

    print(
        f"Successful: "
        f"{len(rows)}/{len(VPS)}"
    )

    print(
        f"Failed: "
        f"{len(failures)}"
    )

    for vp, error in failures:

        print(
            f"  {vp}: {error}"
        )

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(result_path)
    print(aggregate_path)


if __name__ == "__main__":
    main()