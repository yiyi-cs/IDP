"""
PTGaze time-based smoothing experiment for X-only I-DT.

Experiments:
    raw
    centered rolling median 50 ms
    centered rolling median 100 ms

For each signal:
    X-only I-DT threshold sweep 1-6 deg

Inputs:
    debug_5_ptgaze_calibrated.csv
    blink_filter_frame_log.csv
    debug_3_fixations.csv

Important:
    - Keep timestamp_ms_synced unchanged.
    - Blink mask is merged only by frame == frame_number.
    - Smoothing is time-based, not frame-based.
    - Smoothing never crosses invalid temporal segments.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PARAMETERS
# =============================================================================

MIN_FIXATION_DURATION_MS = 100.0
MAX_GAP_MS = 50.0

THRESHOLDS = [
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
]

SMOOTHING_CONFIGS = {
    "raw": None,
    "median_50ms": 50.0,
    "median_100ms": 100.0,
}


# =============================================================================
# LOAD + MERGE
# =============================================================================

def load_data(
    ptgaze_path,
    blink_path,
    eyelink_fixation_path,
):

    pt = pd.read_csv(ptgaze_path)
    blink = pd.read_csv(blink_path)
    eyelink = pd.read_csv(eyelink_fixation_path)

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

    matched = (
        pt["valid_eye_frame"]
        .notna()
        .sum()
    )

    print(
        f"Blink mask matched: "
        f"{matched}/{len(pt)} "
        f"({100 * matched / len(pt):.1f}%)"
    )

    pt["valid_eye_frame"] = (
        pt["valid_eye_frame"]
        .astype("boolean")
    )

    # Remove practice EyeLink fixations
    if "is_practice" in eyelink.columns:

        eyelink = eyelink[
            ~eyelink[
                "is_practice"
            ]
            .fillna(False)
            .astype(bool)
        ].copy()

    return pt, eyelink


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
# VALID TEMPORAL SEGMENTS
# =============================================================================

def build_segments(
    trial_df,
    max_gap_ms=MAX_GAP_MS,
):

    """
    Build valid temporal segments.

    Short invalid gaps <= MAX_GAP_MS may be crossed.
    Longer gaps split the signal.

    Invalid samples themselves are never used for smoothing.
    """

    trial_df = (
        trial_df
        .sort_values("timestamp_ms_synced")
        .reset_index(drop=True)
    )

    valid_mask = (
        get_valid_mask(trial_df)
        .to_numpy()
    )

    segments = []

    current_indices = []
    last_valid_idx = None

    for idx in range(len(trial_df)):

        if not valid_mask[idx]:
            continue

        if last_valid_idx is None:

            current_indices = [idx]
            last_valid_idx = idx
            continue

        current_time = trial_df.loc[
            idx,
            "timestamp_ms_synced",
        ]

        previous_time = trial_df.loc[
            last_valid_idx,
            "timestamp_ms_synced",
        ]

        gap_ms = (
            current_time
            - previous_time
        )

        invalid_between = np.any(
            ~valid_mask[
                last_valid_idx + 1:idx
            ]
        )

        should_split = (
            invalid_between
            and gap_ms > max_gap_ms
        )

        if should_split:

            if current_indices:

                segments.append(
                    trial_df
                    .loc[current_indices]
                    .copy()
                )

            current_indices = [idx]

        else:

            current_indices.append(idx)

        last_valid_idx = idx

    if current_indices:

        segments.append(
            trial_df
            .loc[current_indices]
            .copy()
        )

    return segments


# =============================================================================
# TIME-BASED CENTERED MEDIAN
# =============================================================================

def centered_time_median(
    timestamps,
    values,
    window_ms,
):

    """
    Centered rolling median using actual timestamp_ms_synced.

    Example for 50 ms:
        current time +/- 25 ms

    No assumption about FPS.
    """

    timestamps = np.asarray(timestamps)
    values = np.asarray(values)

    smoothed = np.empty(
        len(values),
        dtype=float,
    )

    half_window = (
        window_ms / 2.0
    )

    left = 0
    right = 0

    for i in range(len(values)):

        lower = (
            timestamps[i]
            - half_window
        )

        upper = (
            timestamps[i]
            + half_window
        )

        while (
            left < len(values)
            and timestamps[left] < lower
        ):
            left += 1

        if right < i:
            right = i

        while (
            right + 1 < len(values)
            and timestamps[right + 1] <= upper
        ):
            right += 1

        smoothed[i] = np.median(
            values[left:right + 1]
        )

    return smoothed


# =============================================================================
# I-DT
# =============================================================================

def x_dispersion(x):

    return (
        np.max(x)
        - np.min(x)
    )


def detect_segment(
    segment,
    threshold_deg,
    smoothing_ms,
):

    segment = (
        segment
        .sort_values("timestamp_ms_synced")
        .reset_index(drop=True)
    )

    t = segment[
        "timestamp_ms_synced"
    ].to_numpy()

    raw_x = segment[
        "gaze_deg_x_calib"
    ].to_numpy()

    if smoothing_ms is None:

        x = raw_x.copy()

    else:

        x = centered_time_median(
            t,
            raw_x,
            smoothing_ms,
        )

    fixations = []

    n = len(segment)
    i = 0

    while i < n:

        minimum_end = (
            t[i]
            + MIN_FIXATION_DURATION_MS
        )

        j = np.searchsorted(
            t,
            minimum_end,
            side="left",
        )

        if j >= n:
            break

        if (
            x_dispersion(
                x[i:j + 1]
            )
            > threshold_deg
        ):

            i += 1
            continue

        k = j

        while k + 1 < n:

            new_dispersion = (
                x_dispersion(
                    x[i:k + 2]
                )
            )

            if new_dispersion > threshold_deg:
                break

            k += 1

        fixations.append({

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

            "dispersion_x_deg":
                x_dispersion(
                    x[i:k + 1]
                ),

            "n_samples":
                k - i + 1,
        })

        i = k + 1

    return fixations


def detect_fixations(
    df,
    threshold_deg,
    smoothing_ms,
):

    all_fixations = []

    fixation_id = 0

    for trial_number, trial in df.groupby(
        "trial_number",
        sort=True,
    ):

        segments = build_segments(
            trial
        )

        for segment in segments:

            detected = detect_segment(
                segment,
                threshold_deg,
                smoothing_ms,
            )

            for fixation in detected:

                fixation_id += 1

                fixation[
                    "fixation_id"
                ] = fixation_id

                fixation[
                    "trial_number"
                ] = trial_number

                all_fixations.append(
                    fixation
                )

    return pd.DataFrame(
        all_fixations
    )


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


def compare_to_eyelink(
    eyelink,
    detected,
):

    best_ious = []

    for _, gt in eyelink.iterrows():

        trial = gt[
            "trial_number"
        ]

        candidates = detected[
            detected[
                "trial_number"
            ] == trial
        ]

        best_iou = 0.0

        for _, pred in candidates.iterrows():

            # Fast skip
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

            if iou > best_iou:
                best_iou = iou

        best_ious.append(
            best_iou
        )

    return np.asarray(
        best_ious
    )


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiment(
    pt,
    eyelink,
    output_dir,
):

    results = []

    for smoothing_name, smoothing_ms in (
        SMOOTHING_CONFIGS.items()
    ):

        print(
            f"\nRunning: "
            f"{smoothing_name}"
        )

        for threshold in THRESHOLDS:

            detected = detect_fixations(
                df=pt,
                threshold_deg=threshold,
                smoothing_ms=smoothing_ms,
            )

            best_ious = (
                compare_to_eyelink(
                    eyelink,
                    detected,
                )
            )

            results.append({

                "smoothing":
                    smoothing_name,

                "window_ms":
                    (
                        0
                        if smoothing_ms is None
                        else smoothing_ms
                    ),

                "threshold_deg":
                    threshold,

                "n_fixations":
                    len(detected),

                "mean_duration_ms":
                    (
                        detected[
                            "duration"
                        ].mean()
                        if len(detected)
                        else np.nan
                    ),

                "median_duration_ms":
                    (
                        detected[
                            "duration"
                        ].median()
                        if len(detected)
                        else np.nan
                    ),

                "native_overlap_pct":
                    100 * (
                        best_ious > 0
                    ).mean(),

                "mean_best_iou":
                    best_ious.mean(),

                "median_best_iou":
                    np.median(
                        best_ious
                    ),

                "iou_ge_0_3_pct":
                    100 * (
                        best_ious >= 0.3
                    ).mean(),

                "iou_ge_0_5_pct":
                    100 * (
                        best_ious >= 0.5
                    ).mean(),

                "iou_ge_0_7_pct":
                    100 * (
                        best_ious >= 0.7
                    ).mean(),
            })

            # Save events as well
            threshold_name = (
                str(threshold)
                .replace(".", "p")
            )

            event_path = (
                output_dir
                / (
                    "debug_6_ptgaze_"
                    f"{smoothing_name}_"
                    f"{threshold_name}deg.csv"
                )
            )

            detected.to_csv(
                event_path,
                index=False,
            )

    return pd.DataFrame(
        results
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "ptgaze_csv"
    )

    parser.add_argument(
        "blink_log_csv"
    )

    parser.add_argument(
        "eyelink_fixations_csv"
    )

    args = parser.parse_args()

    ptgaze_path = Path(
        args.ptgaze_csv
    )

    blink_path = Path(
        args.blink_log_csv
    )

    eyelink_path = Path(
        args.eyelink_fixations_csv
    )

    for path in [
        ptgaze_path,
        blink_path,
        eyelink_path,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                path
            )

    print("=" * 70)
    print("PTGAZE TIME-BASED SMOOTHING EXPERIMENT")
    print("=" * 70)

    pt, eyelink = load_data(
        ptgaze_path,
        blink_path,
        eyelink_path,
    )

    print(
        f"PTGaze samples: "
        f"{len(pt)}"
    )

    print(
        f"EyeLink native fixations: "
        f"{len(eyelink)}"
    )

    print(
        f"Minimum fixation duration: "
        f"{MIN_FIXATION_DURATION_MS:.0f} ms"
    )

    output_dir = (
        ptgaze_path.parent
    )

    results = run_experiment(
        pt,
        eyelink,
        output_dir,
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    output_path = (
        output_dir
        / "debug_6_ptgaze_smoothing_experiment.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()