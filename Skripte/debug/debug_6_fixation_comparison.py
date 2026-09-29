"""
X-only I-DT vs EyeLink fixation comparison.

Compares:
    EyeLink fixation events
    X-only I-DT: 4°, 5°, 6°

Important:
    - Every threshold is evaluated against ALL EyeLink fixations.
    - Missing PTGaze fixations in a trial count as no overlap.
    - This is temporal event comparison, not yet a final accuracy metric.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


THRESHOLDS = [4.0, 5.0, 6.0]


# =============================================================================
# LOAD
# =============================================================================

def load_eyelink(path):

    df = pd.read_csv(path)

    if "is_practice" in df.columns:
        df = df[
            ~df["is_practice"].fillna(False).astype(bool)
        ].copy()

    return df


def load_ptgaze(path):
    return pd.read_csv(path)


# =============================================================================
# BASIC SUMMARY
# =============================================================================

def summarize(name, df):

    return {
        "method": name,
        "n_fixations": len(df),
        "mean_duration_ms": df["duration"].mean(),
        "median_duration_ms": df["duration"].median(),
        "min_duration_ms": df["duration"].min(),
        "max_duration_ms": df["duration"].max(),
    }


# =============================================================================
# OVERLAP
# =============================================================================

def interval_overlap(a_start, a_end, b_start, b_end):

    return max(
        0.0,
        min(a_end, b_end)
        - max(a_start, b_start)
    )


def compare_to_eyelink(eyelink, ptgaze):

    """
    Evaluate ALL EyeLink fixation events.

    For each EyeLink fixation:
        - find temporally overlapping PTGaze fixations
        - calculate temporal coverage
        - detect fragmentation
    """

    rows = []

    for _, et in eyelink.iterrows():

        trial = et["trial_number"]

        pt_trial = ptgaze[
            ptgaze["trial_number"] == trial
        ]

        overlaps = []

        for _, pt in pt_trial.iterrows():

            overlap = interval_overlap(
                et["start_time"],
                et["end_time"],
                pt["start_time"],
                pt["end_time"],
            )

            if overlap > 0:
                overlaps.append(overlap)

        total_overlap = sum(overlaps)

        duration = et["duration"]

        overlap_ratio = (
            min(total_overlap / duration, 1.0)
            if duration > 0
            else np.nan
        )

        rows.append({

            "trial_number":
                trial,

            "eyelink_start":
                et["start_time"],

            "eyelink_end":
                et["end_time"],

            "eyelink_duration":
                duration,

            "n_overlapping_ptgaze":
                len(overlaps),

            "total_overlap_ms":
                total_overlap,

            "overlap_ratio":
                overlap_ratio,

        })

    return pd.DataFrame(rows)


# =============================================================================
# TIMELINE
# =============================================================================

def plot_trial(
    trial,
    eyelink,
    ptgaze_sets,
    output_path,
):

    et = eyelink[
        eyelink["trial_number"] == trial
    ]

    if len(et) == 0:
        return

    fig, ax = plt.subplots(
        figsize=(15, 5)
    )

    # -------------------------------------------------------------
    # EyeLink
    # -------------------------------------------------------------

    y_positions = {
        "EyeLink": 4,
        "I-DT X 4°": 3,
        "I-DT X 5°": 2,
        "I-DT X 6°": 1,
    }

    for _, row in et.iterrows():

        ax.hlines(
            y=y_positions["EyeLink"],
            xmin=row["start_time"],
            xmax=row["end_time"],
            linewidth=7,
        )

    # -------------------------------------------------------------
    # PTGaze
    # -------------------------------------------------------------

    for threshold in THRESHOLDS:

        label = (
            f"I-DT X {threshold:.0f}°"
        )

        pt = ptgaze_sets[threshold]

        pt = pt[
            pt["trial_number"] == trial
        ]

        for _, row in pt.iterrows():

            ax.hlines(
                y=y_positions[label],
                xmin=row["start_time"],
                xmax=row["end_time"],
                linewidth=7,
            )

    ax.set_yticks(
        list(y_positions.values())
    )

    ax.set_yticklabels(
        list(y_positions.keys())
    )

    ax.set_xlabel(
        "Synced timestamp (ms)"
    )

    ax.set_title(
        f"X-only fixation comparison - Trial {trial}"
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "run_directory"
    )

    args = parser.parse_args()

    run_dir = Path(
        args.run_directory
    )

    # -----------------------------------------------------------------
    # Load EyeLink
    # -----------------------------------------------------------------

    eyelink_path = (
        run_dir
        / "debug_3_fixations.csv"
    )

    eyelink = load_eyelink(
        eyelink_path
    )

    print("=" * 70)
    print("X-ONLY FIXATION COMPARISON")
    print("=" * 70)

    print(
        f"\nEyeLink fixations: "
        f"{len(eyelink)}"
    )

    # -----------------------------------------------------------------
    # Load PTGaze
    # -----------------------------------------------------------------

    ptgaze_sets = {}

    for threshold in THRESHOLDS:

        name = (
            str(threshold)
            .replace(".", "p")
        )

        path = (
            run_dir
            / (
                "debug_6_ptgaze_fixations_"
                f"idt_X_{name}deg.csv"
            )
        )

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        ptgaze_sets[threshold] = (
            load_ptgaze(path)
        )

    # -----------------------------------------------------------------
    # Basic statistics
    # -----------------------------------------------------------------

    summaries = [
        summarize(
            "EyeLink",
            eyelink,
        )
    ]

    for threshold in THRESHOLDS:

        summaries.append(
            summarize(
                f"I-DT X {threshold:.0f} deg",
                ptgaze_sets[threshold],
            )
        )

    summary_df = pd.DataFrame(
        summaries
    )

    print()
    print("=" * 70)
    print("FIXATION SUMMARY")
    print("=" * 70)
    print()

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.1f}",
        )
    )

    # -----------------------------------------------------------------
    # Temporal comparison
    # -----------------------------------------------------------------

    comparison_summary = []

    for threshold in THRESHOLDS:

        comparison = compare_to_eyelink(
            eyelink,
            ptgaze_sets[threshold],
        )

        # Every threshold MUST evaluate all EyeLink events
        assert len(comparison) == len(eyelink)

        matched = (
            comparison[
                "n_overlapping_ptgaze"
            ] > 0
        )

        fragmented = (
            comparison[
                "n_overlapping_ptgaze"
            ] > 1
        )

        comparison_summary.append({

            "threshold_deg":
                threshold,

            "n_eyelink_fixations":
                len(comparison),

            "eyelink_with_overlap":
                int(matched.sum()),

            "overlap_percent":
                100 * matched.mean(),

            # Across ALL EyeLink fixations,
            # including zero-overlap events
            "mean_overlap_ratio_all":
                comparison[
                    "overlap_ratio"
                ].mean(),

            # Only EyeLink events for which
            # PTGaze detected something
            "mean_overlap_ratio_matched":
                comparison.loc[
                    matched,
                    "overlap_ratio",
                ].mean(),

            "fragmented_eyelink_fixations":
                int(fragmented.sum()),

            "fragmentation_percent":
                100 * fragmented.mean(),
        })

        detail_path = (
            run_dir
            / (
                "debug_6_X_overlap_"
                f"{threshold:.0f}deg.csv"
            )
        )

        comparison.to_csv(
            detail_path,
            index=False,
        )

    comparison_summary_df = (
        pd.DataFrame(
            comparison_summary
        )
    )

    print()
    print("=" * 70)
    print("TEMPORAL OVERLAP")
    print("=" * 70)
    print()

    print(
        comparison_summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.1f}",
        )
    )

    # -----------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------

    summary_df.to_csv(
        run_dir
        / "debug_6_X_fixation_summary.csv",
        index=False,
    )

    comparison_summary_df.to_csv(
        run_dir
        / "debug_6_X_overlap_summary.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Timeline plots
    # -----------------------------------------------------------------

    plot_dir = (
        run_dir
        / "fixation_X_timeline_plots"
    )

    plot_dir.mkdir(
        exist_ok=True
    )

    trials = sorted(
        eyelink[
            "trial_number"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    print(
        f"\nCreating timelines "
        f"for {len(trials)} trials..."
    )

    for trial in trials:

        plot_trial(
            trial=trial,
            eyelink=eyelink,
            ptgaze_sets=ptgaze_sets,
            output_path=(
                plot_dir
                / f"trial_{trial:02d}.png"
            ),
        )

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        run_dir
        / "debug_6_X_fixation_summary.csv"
    )

    print(
        run_dir
        / "debug_6_X_overlap_summary.csv"
    )

    print(plot_dir)


if __name__ == "__main__":
    main()