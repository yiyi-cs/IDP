"""
Batch validation of participant-adaptive I-DT thresholds.

For each VP:
    1. Load calibrated PTGaze gaze.
    2. Merge final valid_eye_frame mask by frame.
    3. Use known fixation phases to estimate horizontal gaze noise.
    4. Apply 100 ms centered time-based median.
    5. Compute 100 ms X-dispersion distribution.
    6. Use P75 / P80 / P90 as participant-specific I-DT thresholds.
    7. Run X-only I-DT on the complete gaze trajectory.
    8. Compare detected fixations with EyeLink native fixations using temporal IoU.

No participant-specific manual threshold tuning.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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

PERCENTILES = [75, 80, 90]

SMOOTHING_WINDOW_MS = 100.0
DISPERSION_WINDOW_MS = 100.0
MIN_FIXATION_DURATION_MS = 100.0
MAX_GAP_MS = 50.0
MIN_WINDOW_DURATION_MS = 90.0


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_run_files(vp_dir):

    analyse_dir = vp_dir / "Analyse"

    candidates = sorted(
        analyse_dir.glob(
            "Run_60hz_FullCalib_mode2_*"
        )
    )

    # Ignore experimental derived runs
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
            f"No valid FullCalib run in {analyse_dir}"
        )

    # Prefer earliest/original run if several exist.
    run, ptgaze, eyelink = valid_runs[0]

    blink = (
        vp_dir
        / "test60"
        / "blink_filter_frame_log.csv"
    )

    if not blink.exists():
        raise FileNotFoundError(
            f"Missing blink log: {blink}"
        )

    return (
        run,
        ptgaze,
        eyelink,
        blink,
    )


# =============================================================================
# LOAD
# =============================================================================

def load_data(
    ptgaze_path,
    eyelink_path,
    blink_path,
):

    pt = pd.read_csv(
        ptgaze_path
    )

    eyelink = pd.read_csv(
        eyelink_path
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

    pt = pt.merge(
        blink_small,
        left_on="frame",
        right_on="frame_number",
        how="left",
        validate="one_to_one",
    )

    match_ratio = (
        pt["valid_eye_frame"]
        .notna()
        .mean()
    )

    pt["valid_eye_frame"] = (
        pt["valid_eye_frame"]
        .astype("boolean")
    )

    if "is_practice" in eyelink.columns:

        eyelink = eyelink[
            ~eyelink[
                "is_practice"
            ]
            .fillna(False)
            .astype(bool)
        ].copy()

    return (
        pt,
        eyelink,
        match_ratio,
    )


# =============================================================================
# VALIDITY
# =============================================================================

def get_valid_mask(df):

    valid = (
        df["valid_eye_frame"]
        .fillna(False)
        &
        df["gaze_deg_x_calib"]
        .notna()
    )

    if "detected" in df.columns:

        valid &= (
            df["detected"]
            .fillna(False)
            .astype(bool)
        )

    if "plausibility_check" in df.columns:

        valid &= (
            df["plausibility_check"]
            .fillna(False)
            .astype(bool)
        )

    return valid


# =============================================================================
# CENTERED TIME MEDIAN
# =============================================================================

def centered_time_median(
    timestamps,
    values,
    window_ms,
):

    t = np.asarray(
        timestamps,
        dtype=float,
    )

    x = np.asarray(
        values,
        dtype=float,
    )

    result = np.empty(
        len(x),
        dtype=float,
    )

    half = (
        window_ms / 2.0
    )

    left = 0
    right = 0

    for i in range(len(x)):

        lower = t[i] - half
        upper = t[i] + half

        while (
            left < len(x)
            and t[left] < lower
        ):
            left += 1

        if right < i:
            right = i

        while (
            right + 1 < len(x)
            and t[right + 1] <= upper
        ):
            right += 1

        result[i] = np.median(
            x[left:right + 1]
        )

    return result


# =============================================================================
# SEGMENTS
# =============================================================================

def build_segments(
    trial_df,
):

    trial_df = (
        trial_df
        .sort_values(
            "timestamp_ms_synced"
        )
        .reset_index(drop=True)
    )

    valid = (
        get_valid_mask(
            trial_df
        )
        .to_numpy()
    )

    segments = []

    current = []
    last_valid = None

    for idx in range(
        len(trial_df)
    ):

        if not valid[idx]:
            continue

        if last_valid is None:

            current = [idx]
            last_valid = idx
            continue

        current_time = (
            trial_df.loc[
                idx,
                "timestamp_ms_synced",
            ]
        )

        previous_time = (
            trial_df.loc[
                last_valid,
                "timestamp_ms_synced",
            ]
        )

        gap = (
            current_time
            - previous_time
        )

        invalid_between = np.any(
            ~valid[
                last_valid + 1:idx
            ]
        )

        split = (
            invalid_between
            and gap > MAX_GAP_MS
        )

        if split:

            if current:

                segments.append(
                    trial_df
                    .loc[current]
                    .copy()
                )

            current = [idx]

        else:

            current.append(idx)

        last_valid = idx

    if current:

        segments.append(
            trial_df
            .loc[current]
            .copy()
        )

    return segments


# =============================================================================
# ADAPTIVE THRESHOLD ESTIMATION
# =============================================================================

def estimate_noise_distribution(
    pt,
):

    fixation = pt[
        pt["phase_type"]
        == "fixation"
    ].copy()

    fixation = fixation[
        get_valid_mask(
            fixation
        )
    ].copy()

    dispersions = []

    # Keep trials separate.
    for _, trial in fixation.groupby(
        "trial_number",
        sort=True,
    ):

        trial = (
            trial
            .sort_values(
                "timestamp_ms_synced"
            )
            .reset_index(drop=True)
        )

        if len(trial) < 2:
            continue

        t = trial[
            "timestamp_ms_synced"
        ].to_numpy()

        x = trial[
            "gaze_deg_x_calib"
        ].to_numpy()

        x = centered_time_median(
            t,
            x,
            SMOOTHING_WINDOW_MS,
        )


        for i in range(len(t)):

            target = (
                t[i]
                + DISPERSION_WINDOW_MS
            )

            # First sample at or after start + 100 ms
            j = np.searchsorted(
                t,
                target,
                side="left",
            )

            # No complete 100 ms window left
            if j >= len(t):
                continue

            actual_duration = (
                t[j]
                - t[i]
            )

            # Safety check
            if (
                actual_duration
                < MIN_WINDOW_DURATION_MS
            ):
                continue

            # Include endpoint j
            window_x = x[i:j + 1]

            dispersion = (
                np.max(window_x)
                - np.min(window_x)
            )

            dispersions.append(
                dispersion
            )

    return np.asarray(
        dispersions,
        dtype=float,
    )


# =============================================================================
# I-DT
# =============================================================================

def detect_segment(
    segment,
    threshold_deg,
):

    segment = (
        segment
        .sort_values(
            "timestamp_ms_synced"
        )
        .reset_index(drop=True)
    )

    t = segment[
        "timestamp_ms_synced"
    ].to_numpy()

    raw_x = segment[
        "gaze_deg_x_calib"
    ].to_numpy()

    # Same smoothing used during
    # adaptive noise estimation.
    x = centered_time_median(
        t,
        raw_x,
        SMOOTHING_WINDOW_MS,
    )

    detected = []

    i = 0
    n = len(segment)

    while i < n:

        target = (
            t[i]
            + MIN_FIXATION_DURATION_MS
        )

        j = np.searchsorted(
            t,
            target,
            side="left",
        )

        if j >= n:
            break

        dispersion = (
            np.max(x[i:j + 1])
            - np.min(x[i:j + 1])
        )

        if (
            dispersion
            > threshold_deg
        ):

            i += 1
            continue

        k = j

        while k + 1 < n:

            new_dispersion = (
                np.max(
                    x[i:k + 2]
                )
                -
                np.min(
                    x[i:k + 2]
                )
            )

            if (
                new_dispersion
                > threshold_deg
            ):
                break

            k += 1

        detected.append({

            "start_time":
                t[i],

            "end_time":
                t[k],

            "duration":
                t[k] - t[i],

            "x_pos":
                np.mean(
                    x[i:k + 1]
                ),
        })

        i = k + 1

    return detected


def detect_fixations(
    pt,
    threshold_deg,
):

    rows = []

    for trial_number, trial in (
        pt.groupby(
            "trial_number",
            sort=True,
        )
    ):

        segments = (
            build_segments(
                trial
            )
        )

        for segment in segments:

            events = (
                detect_segment(
                    segment,
                    threshold_deg,
                )
            )

            for event in events:

                event[
                    "trial_number"
                ] = trial_number

                rows.append(
                    event
                )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# IoU
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


def evaluate(
    eyelink,
    detected,
):

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

            best = max(
                best,
                iou,
            )

        best_ious.append(
            best
        )

    best_ious = np.asarray(
        best_ious
    )

    return {

        "native_overlap_pct":
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
# VP
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
    ) = find_run_files(
        vp_dir
    )

    pt, eyelink, match_ratio = (
        load_data(
            ptgaze_path,
            eyelink_path,
            blink_path,
        )
    )

    noise = (
        estimate_noise_distribution(
            pt
        )
    )

    if len(noise) == 0:
        raise RuntimeError(
            f"{vp}: no noise windows"
        )

    thresholds = {
        p: np.percentile(
            noise,
            p,
        )
        for p in PERCENTILES
    }

    rows = []

    for percentile in PERCENTILES:

        threshold = (
            thresholds[
                percentile
            ]
        )

        detected = (
            detect_fixations(
                pt,
                threshold,
            )
        )

        metrics = evaluate(
            eyelink,
            detected,
        )

        rows.append({

            "vp":
                vp,

            "run":
                run.name,

            "blink_match_pct":
                100 * match_ratio,

            "adaptive_percentile":
                percentile,

            "threshold_deg":
                threshold,

            "noise_p50_deg":
                np.percentile(
                    noise,
                    50,
                ),

            "noise_p75_deg":
                np.percentile(
                    noise,
                    75,
                ),

            "noise_p80_deg":
                np.percentile(
                    noise,
                    80,
                ),

            "noise_p90_deg":
                np.percentile(
                    noise,
                    90,
                ),

            "native_fixations":
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
        })

    return rows


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
    print("BATCH ADAPTIVE I-DT VALIDATION")
    print("=" * 70)

    rows = []

    failed = []

    for vp in VPS:

        print(
            f"\n[{vp}]"
        )

        try:

            vp_rows = (
                process_vp(
                    vp,
                    root,
                )
            )

            rows.extend(
                vp_rows
            )

            for r in vp_rows:

                print(
                    f"  P{r['adaptive_percentile']}: "
                    f"T={r['threshold_deg']:.2f} deg | "
                    f"IoU={r['mean_best_iou']:.3f} | "
                    f"IoU>=0.5="
                    f"{r['iou_ge_0_5_pct']:.1f}%"
                )

        except Exception as e:

            failed.append(
                (
                    vp,
                    str(e),
                )
            )

            print(
                f"  FAILED: {e}"
            )

    results = pd.DataFrame(
        rows
    )

    if len(results) == 0:

        raise RuntimeError(
            "No VP completed successfully."
        )

    # =================================================================
    # Overall summary by percentile
    # =================================================================

    summary = (
        results
        .groupby(
            "adaptive_percentile"
        )
        .agg(

            n_vps=(
                "vp",
                "nunique",
            ),

            mean_threshold_deg=(
                "threshold_deg",
                "mean",
            ),

            median_threshold_deg=(
                "threshold_deg",
                "median",
            ),

            mean_best_iou=(
                "mean_best_iou",
                "mean",
            ),

            median_best_iou=(
                "mean_best_iou",
                "median",
            ),

            mean_iou_ge_0_5_pct=(
                "iou_ge_0_5_pct",
                "mean",
            ),

            mean_iou_ge_0_7_pct=(
                "iou_ge_0_7_pct",
                "mean",
            ),

            mean_overlap_pct=(
                "native_overlap_pct",
                "mean",
            ),

        )
        .reset_index()
    )

    # =================================================================
    # Print detailed results
    # =================================================================

    print()
    print("=" * 70)
    print("PER-VP RESULTS")
    print("=" * 70)
    print()

    display_columns = [
        "vp",
        "adaptive_percentile",
        "threshold_deg",
        "native_fixations",
        "detected_fixations",
        "detected_median_duration_ms",
        "native_overlap_pct",
        "mean_best_iou",
        "median_best_iou",
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
    # Print overall summary
    # =================================================================

    print()
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print()

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # =================================================================
    # Save
    # =================================================================

    output_detail = (
        root
        / "debug_6_adaptive_threshold_batch_results.csv"
    )

    output_summary = (
        root
        / "debug_6_adaptive_threshold_batch_summary.csv"
    )

    results.to_csv(
        output_detail,
        index=False,
    )

    summary.to_csv(
        output_summary,
        index=False,
    )

    # =================================================================
    # Failures
    # =================================================================

    print()
    print("=" * 70)
    print("RUN STATUS")
    print("=" * 70)

    print(
        f"Successful VPs: "
        f"{results['vp'].nunique()}/{len(VPS)}"
    )

    if failed:

        print(
            f"Failed VPs: {len(failed)}"
        )

        for vp, error in failed:

            print(
                f"  {vp}: {error}"
            )

    else:

        print(
            "Failed VPs: 0"
        )

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(output_detail)
    print(output_summary)


if __name__ == "__main__":
    main()