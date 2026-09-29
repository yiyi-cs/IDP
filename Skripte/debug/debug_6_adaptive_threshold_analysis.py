"""
Adaptive I-DT threshold feasibility analysis.

Goal:
    Estimate PTGaze horizontal gaze noise from known fixation phases.

Pipeline:
    calibrated PTGaze
        + valid_eye_frame
        -> phase_type == fixation
        -> 100 ms centered time-based median
        -> 100 ms sliding-window X dispersion
        -> dispersion distribution

Important:
    - Uses timestamp_ms_synced.
    - Blink mask merged only by frame == frame_number.
    - X only.
    - No threshold is chosen here.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PARAMETERS
# =============================================================================

SMOOTHING_WINDOW_MS = 100.0
DISPERSION_WINDOW_MS = 100.0

# Require enough temporal coverage inside a window.
# Since data are VFR, we do NOT require a fixed number of frames.
MIN_WINDOW_DURATION_MS = 90.0


# =============================================================================
# LOAD
# =============================================================================

def load_and_merge(ptgaze_path, blink_path):

    pt = pd.read_csv(ptgaze_path)
    blink = pd.read_csv(blink_path)

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
        ["frame_number", "valid_eye_frame"]
    ].copy()

    df = pt.merge(
        blink_small,
        left_on="frame",
        right_on="frame_number",
        how="left",
        validate="one_to_one",
    )

    matched = (
        df["valid_eye_frame"]
        .notna()
        .sum()
    )

    print(
        f"Blink mask matched: "
        f"{matched}/{len(df)} "
        f"({100 * matched / len(df):.1f}%)"
    )

    df["valid_eye_frame"] = (
        df["valid_eye_frame"]
        .astype("boolean")
    )

    return df


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
# TIME-BASED CENTERED MEDIAN
# =============================================================================

def centered_time_median(
    timestamps,
    values,
    window_ms,
):

    timestamps = np.asarray(
        timestamps,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    result = np.empty(
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

        result[i] = np.median(
            values[left:right + 1]
        )

    return result


# =============================================================================
# DISPERSION WINDOWS
# =============================================================================

def calculate_window_dispersions(
    trial_df,
):

    """
    Calculate X dispersion in time-based 100 ms windows.

    One window begins at every valid sample.
    """

    trial_df = (
        trial_df
        .sort_values("timestamp_ms_synced")
        .reset_index(drop=True)
    )

    t = trial_df[
        "timestamp_ms_synced"
    ].to_numpy()

    x_raw = trial_df[
        "gaze_deg_x_calib"
    ].to_numpy()

    if len(trial_df) == 0:
        return []

    # -------------------------------------------------------------
    # First smooth using centered 100 ms median
    # -------------------------------------------------------------

    x_smooth = centered_time_median(
        timestamps=t,
        values=x_raw,
        window_ms=SMOOTHING_WINDOW_MS,
    )

    rows = []

    for i in range(len(t)):

        target_end = (
            t[i]
            + DISPERSION_WINDOW_MS
        )

        j = np.searchsorted(
            t,
            target_end,
            side="right",
        )

        # Window is [i, j)
        if j <= i + 1:
            continue

        actual_duration = (
            t[j - 1] - t[i]
        )

        # Do not treat a very short incomplete tail
        # as a full 100 ms dispersion window.
        if (
            actual_duration
            < MIN_WINDOW_DURATION_MS
        ):
            continue

        window_x = (
            x_smooth[i:j]
        )

        dispersion = (
            np.max(window_x)
            - np.min(window_x)
        )

        rows.append({

            "trial_number":
                trial_df.loc[
                    i,
                    "trial_number",
                ],

            "window_start_ms":
                t[i],

            "window_end_ms":
                t[j - 1],

            "window_duration_ms":
                actual_duration,

            "n_samples":
                j - i,

            "dispersion_x_deg":
                dispersion,
        })

    return rows


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
    print("ADAPTIVE I-DT THRESHOLD ANALYSIS")
    print("=" * 70)

    df = load_and_merge(
        ptgaze_path,
        blink_path,
    )

    # =================================================================
    # Known fixation phases only
    # =================================================================

    fixation = df[
        df["phase_type"] == "fixation"
    ].copy()

    valid_mask = get_valid_mask(
        fixation
    )

    fixation = fixation[
        valid_mask
    ].copy()

    print(
        f"\nKnown fixation-phase "
        f"samples: {len(fixation)}"
    )

    print(
        f"Trials represented: "
        f"{fixation['trial_number'].nunique()}"
    )

    print(
        f"Smoothing window: "
        f"{SMOOTHING_WINDOW_MS:.0f} ms"
    )

    print(
        f"Dispersion window: "
        f"{DISPERSION_WINDOW_MS:.0f} ms"
    )

    # =================================================================
    # Per-trial analysis
    # =================================================================

    all_rows = []

    for trial_number, trial in fixation.groupby(
        "trial_number",
        sort=True,
    ):

        rows = calculate_window_dispersions(
            trial
        )

        all_rows.extend(
            rows
        )

    dispersion_df = pd.DataFrame(
        all_rows
    )

    if len(dispersion_df) == 0:

        raise RuntimeError(
            "No valid dispersion windows found."
        )

    # =================================================================
    # Global distribution
    # =================================================================

    d = dispersion_df[
        "dispersion_x_deg"
    ]

    percentiles = [
        50,
        75,
        80,
        90,
        95,
        97.5,
        99,
    ]

    print()
    print("=" * 70)
    print("X DISPERSION DISTRIBUTION")
    print("=" * 70)

    print(
        f"\nValid 100 ms windows: "
        f"{len(d)}"
    )

    print(
        f"Mean   : "
        f"{d.mean():.3f} deg"
    )

    print(
        f"Std    : "
        f"{d.std():.3f} deg"
    )

    print(
        f"Min    : "
        f"{d.min():.3f} deg"
    )

    for p in percentiles:

        value = np.percentile(
            d,
            p,
        )

        print(
            f"P{p:<5}: "
            f"{value:.3f} deg"
        )

    print(
        f"Max    : "
        f"{d.max():.3f} deg"
    )

    # =================================================================
    # Per-trial summary
    # =================================================================

    trial_summary = (
        dispersion_df
        .groupby("trial_number")[
            "dispersion_x_deg"
        ]
        .agg(
            n_windows="count",
            median="median",
            mean="mean",
            max="max",
        )
        .reset_index()
    )

    # Add robust percentiles per trial
    p90 = (
        dispersion_df
        .groupby("trial_number")[
            "dispersion_x_deg"
        ]
        .quantile(0.90)
        .rename("p90")
    )

    p95 = (
        dispersion_df
        .groupby("trial_number")[
            "dispersion_x_deg"
        ]
        .quantile(0.95)
        .rename("p95")
    )

    trial_summary = (
        trial_summary
        .merge(
            p90,
            on="trial_number",
        )
        .merge(
            p95,
            on="trial_number",
        )
    )

    print()
    print("=" * 70)
    print("PER-TRIAL SUMMARY")
    print("=" * 70)
    print()

    print(
        trial_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # =================================================================
    # Save
    # =================================================================

    output_dir = (
        ptgaze_path.parent
    )

    dispersion_path = (
        output_dir
        / "debug_6_fixation_noise_windows.csv"
    )

    summary_path = (
        output_dir
        / "debug_6_fixation_noise_trial_summary.csv"
    )

    dispersion_df.to_csv(
        dispersion_path,
        index=False,
    )

    trial_summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(dispersion_path)
    print(summary_path)


if __name__ == "__main__":
    main()