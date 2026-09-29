"""
Diagnostic for VPs where adaptive threshold estimation produced
"no noise windows".

No fixation detection is performed here.
This script only inspects:
    - phase_type
    - blink validity
    - gaze validity
    - timestamp spacing
    - availability of >= 90 ms windows
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FAILED_VPS = [
    "egf5",
    "fbn6",
    "fgt6",
    "jkl7",
    "ogt7",
]

MIN_WINDOW_DURATION_MS = 90.0
TARGET_WINDOW_MS = 100.0


# =============================================================================
# FIND ORIGINAL RUN
# =============================================================================

def find_run(vp_dir):

    analyse_dir = vp_dir / "Analyse"

    candidates = sorted(
        analyse_dir.glob(
            "Run_60hz_FullCalib_mode2_*"
        )
    )

    candidates = [
        p for p in candidates
        if (
            "BlinkFiltered" not in p.name
            and "PTS" not in p.name
        )
    ]

    for run in candidates:

        ptgaze = (
            run
            / "debug_5_ptgaze_calibrated.csv"
        )

        if ptgaze.exists():
            return run, ptgaze

    raise FileNotFoundError(
        f"No calibrated PTGaze run for {vp_dir.name}"
    )


# =============================================================================
# ANALYSE ONE VP
# =============================================================================

def diagnose_vp(vp, root):

    vp_dir = root / vp

    run, ptgaze_path = (
        find_run(vp_dir)
    )

    blink_path = (
        vp_dir
        / "test60"
        / "blink_filter_frame_log.csv"
    )

    if not blink_path.exists():
        raise FileNotFoundError(
            blink_path
        )

    pt = pd.read_csv(
        ptgaze_path
    )

    blink = pd.read_csv(
        blink_path
    )

    # -----------------------------------------------------------------
    # Basic
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print(vp)
    print("=" * 70)

    print(
        f"Run: {run.name}"
    )

    print(
        f"Total PTGaze samples: "
        f"{len(pt)}"
    )

    # -----------------------------------------------------------------
    # Phase types
    # -----------------------------------------------------------------

    if "phase_type" not in pt.columns:

        print(
            "ERROR: phase_type column missing"
        )

        return

    print()
    print("phase_type values:")

    phase_counts = (
        pt["phase_type"]
        .value_counts(
            dropna=False
        )
    )

    print(
        phase_counts.to_string()
    )

    fixation_phase = pt[
        pt["phase_type"]
        == "fixation"
    ].copy()

    print(
        f"\nphase_type == fixation: "
        f"{len(fixation_phase)}"
    )

    # -----------------------------------------------------------------
    # Blink merge
    # -----------------------------------------------------------------

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

    blink_matched = (
        merged[
            "valid_eye_frame"
        ]
        .notna()
    )

    print(
        f"Blink matched: "
        f"{blink_matched.sum()}/"
        f"{len(merged)} "
        f"({100 * blink_matched.mean():.1f}%)"
    )

    fixation = merged[
        merged["phase_type"]
        == "fixation"
    ].copy()

    print()
    print(
        f"Fixation-phase samples "
        f"after merge: {len(fixation)}"
    )

    if len(fixation) == 0:

        print(
            "STOP: no fixation-phase samples."
        )

        return

    # -----------------------------------------------------------------
    # Step-by-step validity
    # -----------------------------------------------------------------

    blink_valid = (
        fixation[
            "valid_eye_frame"
        ]
        .fillna(False)
        .astype(bool)
    )

    print(
        f"valid_eye_frame=True: "
        f"{blink_valid.sum()}"
    )

    gaze_valid = (
        fixation[
            "gaze_deg_x_calib"
        ]
        .notna()
    )

    print(
        f"gaze_deg_x_calib valid: "
        f"{gaze_valid.sum()}"
    )

    if "detected" in fixation.columns:

        detected = (
            fixation[
                "detected"
            ]
            .fillna(False)
            .astype(bool)
        )

        print(
            f"detected=True: "
            f"{detected.sum()}"
        )

    else:

        detected = pd.Series(
            True,
            index=fixation.index,
        )

        print(
            "detected column: MISSING"
        )

    if (
        "plausibility_check"
        in fixation.columns
    ):

        plausible = (
            fixation[
                "plausibility_check"
            ]
            .fillna(False)
            .astype(bool)
        )

        print(
            f"plausibility_check=True: "
            f"{plausible.sum()}"
        )

    else:

        plausible = pd.Series(
            True,
            index=fixation.index,
        )

        print(
            "plausibility_check column: MISSING"
        )

    final_valid = (
        blink_valid
        & gaze_valid
        & detected
        & plausible
    )

    valid_fixation = fixation[
        final_valid
    ].copy()

    print(
        f"\nFINAL valid fixation samples: "
        f"{len(valid_fixation)}"
    )

    if len(valid_fixation) == 0:

        print(
            "STOP: validity filtering "
            "removed all samples."
        )

        return

    # -----------------------------------------------------------------
    # Trials
    # -----------------------------------------------------------------

    print(
        f"Trials with valid fixation data: "
        f"{valid_fixation['trial_number'].nunique()}"
    )

    # -----------------------------------------------------------------
    # Timestamp diagnostics
    # -----------------------------------------------------------------

    valid_fixation = (
        valid_fixation
        .sort_values(
            [
                "trial_number",
                "timestamp_ms_synced",
            ]
        )
    )

    dt = (
        valid_fixation
        .groupby(
            "trial_number"
        )[
            "timestamp_ms_synced"
        ]
        .diff()
    )

    dt_valid = dt[
        dt > 0
    ]

    if len(dt_valid):

        print()
        print("timestamp_ms_synced Δt:")

        print(
            f"  median = "
            f"{dt_valid.median():.3f} ms"
        )

        print(
            f"  mean   = "
            f"{dt_valid.mean():.3f} ms"
        )

        print(
            f"  min    = "
            f"{dt_valid.min():.3f} ms"
        )

        print(
            f"  max    = "
            f"{dt_valid.max():.3f} ms"
        )

    # -----------------------------------------------------------------
    # Trial duration
    # -----------------------------------------------------------------

    trial_stats = []

    total_candidate_windows = 0
    total_valid_windows = 0

    for trial_number, trial in (
        valid_fixation.groupby(
            "trial_number",
            sort=True,
        )
    ):

        trial = (
            trial
            .sort_values(
                "timestamp_ms_synced"
            )
            .reset_index(drop=True)
        )

        t = trial[
            "timestamp_ms_synced"
        ].to_numpy()

        if len(t) < 2:
            duration = 0.0
        else:
            duration = (
                t[-1] - t[0]
            )

        candidate_windows = 0
        valid_windows = 0

        for i in range(len(t)):

            target = (
                t[i]
                + TARGET_WINDOW_MS
            )

            j = np.searchsorted(
                t,
                target,
                side="right",
            )

            if j <= i + 1:
                continue

            candidate_windows += 1

            actual_duration = (
                t[j - 1]
                - t[i]
            )

            if (
                actual_duration
                >= MIN_WINDOW_DURATION_MS
            ):
                valid_windows += 1

        total_candidate_windows += (
            candidate_windows
        )

        total_valid_windows += (
            valid_windows
        )

        trial_stats.append({

            "trial":
                trial_number,

            "samples":
                len(trial),

            "duration_ms":
                duration,

            "candidate_windows":
                candidate_windows,

            "valid_90ms_windows":
                valid_windows,
        })

    trial_df = pd.DataFrame(
        trial_stats
    )

    print()
    print("Per-trial fixation phase:")
    print()

    print(
        trial_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print()
    print(
        f"TOTAL candidate windows: "
        f"{total_candidate_windows}"
    )

    print(
        f"TOTAL >=90 ms windows: "
        f"{total_valid_windows}"
    )

    if total_valid_windows == 0:

        print()
        print(
            ">>> FAILURE LOCATION: "
            "no fixation phase contains "
            "a valid ~100 ms temporal window."
        )

    else:

        print()
        print(
            ">>> Windows exist. "
            "Failure must be elsewhere "
            "in adaptive batch implementation."
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "results_root",
        help=(
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
    print("ADAPTIVE THRESHOLD FAILURE DIAGNOSTIC")
    print("=" * 70)

    for vp in FAILED_VPS:

        try:

            diagnose_vp(
                vp,
                root,
            )

        except Exception as e:

            print()
            print(
                f"{vp}: ERROR: {e}"
            )


if __name__ == "__main__":
    main()