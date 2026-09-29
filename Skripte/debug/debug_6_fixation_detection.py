"""
DEBUG_6_FIXATION_DETECTION.PY

X-only I-DT fixation detection for calibrated PTGaze gaze.

Inputs:
    1. debug_5_ptgaze_calibrated.csv
    2. blink_filter_frame_log.csv

Important:
    - Fixation detection uses horizontal gaze (X) only.
    - Blink validity is merged by:
          PTGaze.frame == Blink.frame_number
    - Original timestamp_ms_synced is kept unchanged.
    - valid_eye_frame=False is treated as invalid gaze.

Outputs:
    debug_6_ptgaze_fixations_idt_X_*deg.csv
    debug_6_idt_X_threshold_sweep.csv
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

THRESHOLDS_TO_TEST = [
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
]


# =============================================================================
# LOAD + BLINK MASK
# =============================================================================

def load_and_merge(ptgaze_path, blink_path):

    pt = pd.read_csv(ptgaze_path)
    blink = pd.read_csv(blink_path)

    required_pt = [
        "frame",
        "timestamp_ms_synced",
        "trial_number",
        "gaze_deg_x_calib",
    ]

    required_blink = [
        "frame_number",
        "valid_eye_frame",
    ]

    for col in required_pt:
        if col not in pt.columns:
            raise ValueError(
                f"PTGaze missing column: {col}"
            )

    for col in required_blink:
        if col not in blink.columns:
            raise ValueError(
                f"Blink log missing column: {col}"
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
        merged["valid_eye_frame"]
        .notna()
        .sum()
    )

    print(
        f"Blink mask matched: "
        f"{matched}/{len(merged)} "
        f"({100 * matched / len(merged):.1f}%)"
    )

    # Keep unmatched explicitly unknown/invalid
    merged["valid_eye_frame"] = (
        merged["valid_eye_frame"]
        .astype("boolean")
    )

    return merged


# =============================================================================
# VALIDITY
# =============================================================================

def get_valid_mask(df):

    valid = (
        df["valid_eye_frame"].fillna(False)
        &
        df["gaze_deg_x_calib"].notna()
    )

    # Keep existing gaze quality checks
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
# TEMPORAL SEGMENTS
# =============================================================================

def build_segments(
    trial_df,
    mode="tolerant",
    max_gap_ms=MAX_GAP_MS,
):

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

        should_split = False

        if mode == "strict":

            if invalid_between:
                should_split = True

        elif mode == "tolerant":

            if (
                invalid_between
                and gap_ms > max_gap_ms
            ):
                should_split = True

        else:
            raise ValueError(
                f"Unknown mode: {mode}"
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
# X-ONLY I-DT
# =============================================================================

def calculate_x_dispersion(x):

    return (
        np.max(x)
        - np.min(x)
    )


def detect_idt_in_segment(
    segment,
    trial_number,
    fixation_id_start,
    threshold_deg,
):

    segment = (
        segment
        .sort_values("timestamp_ms_synced")
        .reset_index(drop=True)
    )

    timestamps = segment[
        "timestamp_ms_synced"
    ].to_numpy()

    gaze_x = segment[
        "gaze_deg_x_calib"
    ].to_numpy()

    frames = segment[
        "frame"
    ].to_numpy()

    n = len(segment)

    fixations = []
    fixation_id = fixation_id_start

    i = 0

    while i < n:

        # -------------------------------------------------------------
        # Minimum 100 ms window
        # -------------------------------------------------------------

        minimum_end_time = (
            timestamps[i]
            + MIN_FIXATION_DURATION_MS
        )

        j = np.searchsorted(
            timestamps,
            minimum_end_time,
            side="left",
        )

        if j >= n:
            break

        # -------------------------------------------------------------
        # X-only dispersion
        # -------------------------------------------------------------

        dispersion_x = (
            calculate_x_dispersion(
                gaze_x[i:j + 1]
            )
        )

        if dispersion_x > threshold_deg:

            i += 1
            continue

        # -------------------------------------------------------------
        # Valid fixation candidate -> extend
        # -------------------------------------------------------------

        k = j

        while k + 1 < n:

            new_dispersion = (
                calculate_x_dispersion(
                    gaze_x[i:k + 2]
                )
            )

            if new_dispersion > threshold_deg:
                break

            k += 1

        # -------------------------------------------------------------
        # Store event
        # -------------------------------------------------------------

        start_time = timestamps[i]
        end_time = timestamps[k]

        fixation_id += 1

        fixations.append({

            "fixation_id":
                fixation_id,

            "trial_number":
                trial_number,

            "start_time":
                start_time,

            "end_time":
                end_time,

            "duration":
                end_time - start_time,

            # Horizontal fixation position
            "x_pos":
                np.mean(
                    gaze_x[i:k + 1]
                ),

            "dispersion_x_deg":
                calculate_x_dispersion(
                    gaze_x[i:k + 1]
                ),

            "n_samples":
                k - i + 1,

            "start_frame":
                frames[i],

            "end_frame":
                frames[k],
        })

        i = k + 1

    return fixations, fixation_id


# =============================================================================
# COMPLETE DETECTION
# =============================================================================

def detect_fixations(
    df,
    threshold_deg,
    mode="tolerant",
):

    all_fixations = []

    fixation_id = 0
    total_segments = 0

    for trial_number, trial_df in df.groupby(
        "trial_number",
        sort=True,
    ):

        segments = build_segments(
            trial_df,
            mode=mode,
            max_gap_ms=MAX_GAP_MS,
        )

        total_segments += len(segments)

        for segment in segments:

            fixations, fixation_id = (
                detect_idt_in_segment(
                    segment=segment,
                    trial_number=trial_number,
                    fixation_id_start=fixation_id,
                    threshold_deg=threshold_deg,
                )
            )

            all_fixations.extend(
                fixations
            )

    return (
        pd.DataFrame(all_fixations),
        total_segments,
    )


# =============================================================================
# THRESHOLD SWEEP
# =============================================================================

def run_threshold_sweep(
    df,
    output_dir,
    mode,
):

    results = []

    for threshold in THRESHOLDS_TO_TEST:

        fixations, n_segments = (
            detect_fixations(
                df=df,
                threshold_deg=threshold,
                mode=mode,
            )
        )

        threshold_name = (
            str(threshold)
            .replace(".", "p")
        )

        event_path = (
            output_dir
            / (
                "debug_6_ptgaze_fixations_"
                f"idt_X_{threshold_name}deg.csv"
            )
        )

        fixations.to_csv(
            event_path,
            index=False,
        )

        if len(fixations) > 0:

            results.append({

                "threshold_deg":
                    threshold,

                "n_fixations":
                    len(fixations),

                "n_segments":
                    n_segments,

                "mean_duration_ms":
                    fixations[
                        "duration"
                    ].mean(),

                "median_duration_ms":
                    fixations[
                        "duration"
                    ].median(),

                "min_duration_ms":
                    fixations[
                        "duration"
                    ].min(),

                "max_duration_ms":
                    fixations[
                        "duration"
                    ].max(),

                "mean_dispersion_x_deg":
                    fixations[
                        "dispersion_x_deg"
                    ].mean(),
            })

        else:

            results.append({

                "threshold_deg":
                    threshold,

                "n_fixations":
                    0,

                "n_segments":
                    n_segments,

                "mean_duration_ms":
                    np.nan,

                "median_duration_ms":
                    np.nan,

                "min_duration_ms":
                    np.nan,

                "max_duration_ms":
                    np.nan,

                "mean_dispersion_x_deg":
                    np.nan,
            })

    return pd.DataFrame(
        results
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "ptgaze_csv",
        help="debug_5_ptgaze_calibrated.csv",
    )

    parser.add_argument(
        "blink_log_csv",
        help="blink_filter_frame_log.csv",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "strict",
            "tolerant",
        ],
        default="tolerant",
    )

    args = parser.parse_args()

    ptgaze_path = Path(
        args.ptgaze_csv
    )

    blink_path = Path(
        args.blink_log_csv
    )

    if not ptgaze_path.exists():
        raise FileNotFoundError(
            ptgaze_path
        )

    if not blink_path.exists():
        raise FileNotFoundError(
            blink_path
        )

    print("=" * 70)
    print("X-ONLY I-DT FIXATION DETECTION")
    print("=" * 70)

    print(
        f"\nPTGaze: {ptgaze_path}"
    )

    print(
        f"Blink:   {blink_path}"
    )

    print(
        f"\nMode: {args.mode}"
    )

    print(
        f"Minimum duration: "
        f"{MIN_FIXATION_DURATION_MS:.0f} ms"
    )

    print(
        f"Max invalid gap: "
        f"{MAX_GAP_MS:.0f} ms"
    )

    print(
        f"Thresholds: "
        f"{THRESHOLDS_TO_TEST}"
    )

    # -----------------------------------------------------------------
    # Merge
    # -----------------------------------------------------------------

    df = load_and_merge(
        ptgaze_path,
        blink_path,
    )

    print(
        f"Samples: {len(df)}"
    )

    n_invalid = (
        (
            df["valid_eye_frame"]
            == False
        )
        .sum()
    )

    print(
        f"Invalid eye frames "
        f"in calibrated gaze: "
        f"{n_invalid}"
    )

    # -----------------------------------------------------------------
    # Sweep
    # -----------------------------------------------------------------

    output_dir = (
        ptgaze_path.parent
    )

    sweep_df = (
        run_threshold_sweep(
            df=df,
            output_dir=output_dir,
            mode=args.mode,
        )
    )

    print()
    print("=" * 70)
    print("X-ONLY THRESHOLD SWEEP")
    print("=" * 70)
    print()

    print(
        sweep_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.1f}",
        )
    )

    sweep_path = (
        output_dir
        / "debug_6_idt_X_threshold_sweep.csv"
    )

    sweep_df.to_csv(
        sweep_path,
        index=False,
    )

    print()
    print(
        f"Saved: {sweep_path}"
    )


if __name__ == "__main__":
    main()