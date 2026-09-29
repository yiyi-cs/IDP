"""
FIXATION_DIAGNOSTICS.PY

Diagnostic visualization for PTGaze fixation detection.

Purpose:
    Inspect PTGaze gaze behaviour inside EyeLink fixation intervals.

Inputs:
    1. debug_5_ptgaze_calibrated.csv
    2. debug_3_fixations.csv
    3. blink_filter_frame_log.csv

Important:
    - Keep original PTGaze timestamp_ms_synced unchanged.
    - Merge Blink validity only by:
          PTGaze.frame == Blink.frame_number
    - valid_eye_frame is visualized as a quality mask.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data(ptgaze_path, eyelink_path, blink_path):

    pt = pd.read_csv(ptgaze_path)
    et = pd.read_csv(eyelink_path)
    blink = pd.read_csv(blink_path)

    # -------------------------------------------------------------------------
    # Basic checks
    # -------------------------------------------------------------------------

    required_pt = [
        "frame",
        "timestamp_ms_synced",
        "trial_number",
        "gaze_deg_x_calib",
        "gaze_deg_y_calib",
    ]

    for col in required_pt:
        if col not in pt.columns:
            raise ValueError(
                f"PTGaze missing column: {col}"
            )

    required_et = [
        "start_time",
        "end_time",
        "duration",
        "trial_number",
    ]

    for col in required_et:
        if col not in et.columns:
            raise ValueError(
                f"EyeLink fixation file missing column: {col}"
            )

    required_blink = [
        "frame_number",
        "valid_eye_frame",
    ]

    for col in required_blink:
        if col not in blink.columns:
            raise ValueError(
                f"Blink log missing column: {col}"
            )

    # -------------------------------------------------------------------------
    # Remove practice EyeLink fixations
    # -------------------------------------------------------------------------

    if "is_practice" in et.columns:
        et = et[
            ~et["is_practice"].fillna(False).astype(bool)
        ].copy()

    # -------------------------------------------------------------------------
    # Merge NEW blink validity into OLD calibrated gaze
    #
    # IMPORTANT:
    # frame == frame_number
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

    # -------------------------------------------------------------------------
    # Merge diagnostics
    # -------------------------------------------------------------------------

    matched = merged["valid_eye_frame"].notna().sum()

    print(
        f"Blink mask matched: "
        f"{matched}/{len(merged)} "
        f"({100 * matched / len(merged):.1f}%)"
    )

    # Do not silently call unmatched frames valid
    merged["valid_eye_frame"] = (
        merged["valid_eye_frame"]
        .astype("boolean")
    )

    return merged, et


# =============================================================================
# FIXATION DISPERSION
# =============================================================================

def calculate_fixation_diagnostics(pt_trial, eyelink_trial):

    rows = []

    for fixation_id, (_, fixation) in enumerate(
        eyelink_trial.iterrows(),
        start=1,
    ):

        start = fixation["start_time"]
        end = fixation["end_time"]

        samples = pt_trial[
            (pt_trial["timestamp_ms_synced"] >= start)
            &
            (pt_trial["timestamp_ms_synced"] <= end)
        ].copy()

        valid = samples[
            samples["valid_eye_frame"] == True
        ].dropna(
            subset=[
                "gaze_deg_x_calib",
                "gaze_deg_y_calib",
            ]
        )

        if len(valid) > 0:

            x_range = (
                valid["gaze_deg_x_calib"].max()
                - valid["gaze_deg_x_calib"].min()
            )

            y_range = (
                valid["gaze_deg_y_calib"].max()
                - valid["gaze_deg_y_calib"].min()
            )

            dispersion = x_range + y_range

            x_std = valid[
                "gaze_deg_x_calib"
            ].std()

            y_std = valid[
                "gaze_deg_y_calib"
            ].std()

        else:

            x_range = np.nan
            y_range = np.nan
            dispersion = np.nan
            x_std = np.nan
            y_std = np.nan

        rows.append({

            "eyelink_fixation_id":
                fixation_id,

            "start_time":
                start,

            "end_time":
                end,

            "eyelink_duration_ms":
                fixation["duration"],

            "n_ptgaze_samples":
                len(samples),

            "n_valid_ptgaze_samples":
                len(valid),

            "ptgaze_x_range_deg":
                x_range,

            "ptgaze_y_range_deg":
                y_range,

            "ptgaze_dispersion_deg":
                dispersion,

            "ptgaze_x_std_deg":
                x_std,

            "ptgaze_y_std_deg":
                y_std,
        })

    return pd.DataFrame(rows)


# =============================================================================
# PLOT
# =============================================================================

def plot_trial(
    trial_number,
    pt,
    et,
    output_path,
):

    pt_trial = pt[
        pt["trial_number"] == trial_number
    ].copy()

    et_trial = et[
        et["trial_number"] == trial_number
    ].copy()

    if len(pt_trial) == 0:
        print(
            f"Trial {trial_number}: "
            f"no PTGaze data"
        )
        return

    if len(et_trial) == 0:
        print(
            f"Trial {trial_number}: "
            f"no EyeLink fixation data"
        )
        return

    pt_trial = pt_trial.sort_values(
        "timestamp_ms_synced"
    )

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
    )

    ax_x = axes[0]
    ax_y = axes[1]

    t = pt_trial[
        "timestamp_ms_synced"
    ]

    x = pt_trial[
        "gaze_deg_x_calib"
    ]

    y = pt_trial[
        "gaze_deg_y_calib"
    ]

    # -------------------------------------------------------------------------
    # PTGaze trajectory
    # -------------------------------------------------------------------------

    ax_x.plot(
        t,
        x,
        marker=".",
        markersize=3,
        linewidth=1,
        label="PTGaze X",
    )

    ax_y.plot(
        t,
        y,
        marker=".",
        markersize=3,
        linewidth=1,
        label="PTGaze Y",
    )

    # -------------------------------------------------------------------------
    # EyeLink fixation intervals
    # -------------------------------------------------------------------------

    for _, fixation in et_trial.iterrows():

        start = fixation["start_time"]
        end = fixation["end_time"]

        ax_x.axvspan(
            start,
            end,
            alpha=0.12,
        )

        ax_y.axvspan(
            start,
            end,
            alpha=0.12,
        )

    # -------------------------------------------------------------------------
    # Invalid eye frames
    # -------------------------------------------------------------------------

    invalid = pt_trial[
        pt_trial["valid_eye_frame"] == False
    ]

    for timestamp in invalid[
        "timestamp_ms_synced"
    ]:

        ax_x.axvline(
            timestamp,
            alpha=0.18,
            linewidth=1,
        )

        ax_y.axvline(
            timestamp,
            alpha=0.18,
            linewidth=1,
        )

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------

    ax_x.set_ylabel(
        "Gaze X (deg)"
    )

    ax_y.set_ylabel(
        "Gaze Y (deg)"
    )

    ax_y.set_xlabel(
        "Synced timestamp (ms)"
    )

    ax_x.set_title(
        f"Trial {trial_number}: "
        f"PTGaze trajectory inside EyeLink fixations"
    )

    ax_x.legend()
    ax_y.legend()

    ax_x.grid(alpha=0.2)
    ax_y.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------------------------
    # Numeric diagnostics
    # -------------------------------------------------------------------------

    diagnostics = calculate_fixation_diagnostics(
        pt_trial,
        et_trial,
    )

    csv_path = (
        output_path.parent
        / f"trial_{trial_number:02d}_dispersion.csv"
    )

    diagnostics.to_csv(
        csv_path,
        index=False,
    )

    print(
        f"Trial {trial_number}: "
        f"{len(et_trial)} EyeLink fixations, "
        f"saved plot + diagnostics"
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
        "eyelink_fixations_csv",
        help="debug_3_fixations.csv",
    )

    parser.add_argument(
        "blink_log_csv",
        help="blink_filter_frame_log.csv",
    )

    parser.add_argument(
        "--trials",
        nargs="+",
        type=int,
        default=[1, 2, 9],
    )

    args = parser.parse_args()

    ptgaze_path = Path(
        args.ptgaze_csv
    )

    eyelink_path = Path(
        args.eyelink_fixations_csv
    )

    blink_path = Path(
        args.blink_log_csv
    )

    for path in [
        ptgaze_path,
        eyelink_path,
        blink_path,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                path
            )

    print("=" * 70)
    print("FIXATION DIAGNOSTICS")
    print("=" * 70)

    pt, et = load_data(
        ptgaze_path,
        eyelink_path,
        blink_path,
    )

    output_dir = (
        ptgaze_path.parent
        / "fixation_diagnostics"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    print(
        f"\nTrials: {args.trials}"
    )

    for trial in args.trials:

        output_path = (
            output_dir
            / f"trial_{trial:02d}.png"
        )

        plot_trial(
            trial_number=trial,
            pt=pt,
            et=et,
            output_path=output_path,
        )

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)
    print(output_dir)


if __name__ == "__main__":
    main()