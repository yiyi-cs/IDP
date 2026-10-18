"""
Fixation detection for webcam-based gaze trajectories.

Method:
    - Horizontal gaze (X) only
    - Time-based centered median smoothing
    - Participant-adaptive I-DT threshold
    - Threshold estimated from known fixation phases (P75)
    - Minimum fixation duration: 100 ms

The implementation is independent of frame rate and uses the existing
synchronized gaze timeline (timestamp_ms_synced).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FixationConfig:
    smoothing_window_ms: float = 100.0
    dispersion_window_ms: float = 100.0
    min_fixation_duration_ms: float = 100.0

    adaptive_percentile: float = 75.0

    # Invalid gaps longer than this split the signal.
    max_gap_ms: float = 50.0


# =============================================================================
# RESULT
# =============================================================================

@dataclass
class FixationResult:
    fixations: pd.DataFrame
    threshold_deg: float
    noise_distribution: np.ndarray
    config: FixationConfig


# =============================================================================
# TIME-BASED SMOOTHING
# =============================================================================

def centered_time_median(
    timestamps: np.ndarray,
    values: np.ndarray,
    window_ms: float,
) -> np.ndarray:

    """
    Centered rolling median based on timestamps rather than frame count.

    Example:
        window_ms = 100
        -> current timestamp +/- 50 ms
    """

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

    half_window = (
        window_ms / 2.0
    )

    left = 0
    right = 0

    for i in range(len(x)):

        lower = (
            t[i] - half_window
        )

        upper = (
            t[i] + half_window
        )

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
# VALIDITY
# =============================================================================

def get_valid_mask(
    df: pd.DataFrame,
) -> pd.Series:

    """
    Determine which gaze samples are valid for fixation analysis.
    """

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
# TEMPORAL SEGMENTS
# =============================================================================

def build_valid_segments(
    trial_df: pd.DataFrame,
    config: FixationConfig,
):

    """
    Split one trial into valid temporal segments.

    Invalid samples are excluded.

    Short invalid gaps <= max_gap_ms may be crossed.
    Longer invalid gaps split the signal.
    """

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

    current_indices = []
    last_valid_idx = None

    for idx in range(
        len(trial_df)
    ):

        if not valid[idx]:
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
            ~valid[
                last_valid_idx + 1:idx
            ]
        )

        should_split = (
            invalid_between
            and gap_ms > config.max_gap_ms
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
# ADAPTIVE THRESHOLD
# =============================================================================

def estimate_adaptive_threshold(
    df: pd.DataFrame,
    config: FixationConfig,
):

    """
    Estimate participant-specific horizontal dispersion threshold.

    Uses known fixation phases:
        phase_type == "fixation"

    Procedure:
        valid samples
        -> 100 ms centered median
        -> 100 ms dispersion windows
        -> P75
    """

    if "phase_type" not in df.columns:

        raise ValueError(
            "phase_type is required for adaptive threshold estimation."
        )

    fixation_phase = df[
        df["phase_type"]
        == "fixation"
    ].copy()

    fixation_phase = fixation_phase[
        get_valid_mask(
            fixation_phase
        )
    ].copy()

    dispersions = []

    for _, trial in fixation_phase.groupby(
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

        raw_x = trial[
            "gaze_deg_x_calib"
        ].to_numpy()

        smooth_x = centered_time_median(
            timestamps=t,
            values=raw_x,
            window_ms=config.smoothing_window_ms,
        )

        for i in range(len(t)):

            target = (
                t[i]
                + config.dispersion_window_ms
            )

            # First sample at or after
            # start + dispersion_window_ms
            j = np.searchsorted(
                t,
                target,
                side="left",
            )

            if j >= len(t):
                continue

            window_x = (
                smooth_x[i:j + 1]
            )

            dispersion = (
                np.max(window_x)
                - np.min(window_x)
            )

            dispersions.append(
                dispersion
            )

    if not dispersions:

        raise RuntimeError(
            "Could not estimate adaptive fixation threshold: "
            "no valid fixation-phase windows."
        )

    dispersions = np.asarray(
        dispersions,
        dtype=float,
    )

    threshold = np.percentile(
        dispersions,
        config.adaptive_percentile,
    )

    return (
        float(threshold),
        dispersions,
    )


# =============================================================================
# I-DT
# =============================================================================

def detect_segment_fixations(
    segment: pd.DataFrame,
    threshold_deg: float,
    config: FixationConfig,
):

    """
    X-only I-DT inside one valid temporal segment.
    """

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

    frames = (
        segment["frame"].to_numpy()
        if "frame" in segment.columns
        else None
    )

    smooth_x = centered_time_median(
        timestamps=t,
        values=raw_x,
        window_ms=config.smoothing_window_ms,
    )

    events = []

    i = 0
    n = len(segment)

    while i < n:

        target = (
            t[i]
            + config.min_fixation_duration_ms
        )

        j = np.searchsorted(
            t,
            target,
            side="left",
        )

        if j >= n:
            break

        dispersion = (
            np.max(
                smooth_x[i:j + 1]
            )
            -
            np.min(
                smooth_x[i:j + 1]
            )
        )

        if dispersion > threshold_deg:

            i += 1
            continue

        # -------------------------------------------------------------
        # Extend fixation
        # -------------------------------------------------------------

        k = j

        while k + 1 < n:

            new_dispersion = (
                np.max(
                    smooth_x[i:k + 2]
                )
                -
                np.min(
                    smooth_x[i:k + 2]
                )
            )

            if (
                new_dispersion
                > threshold_deg
            ):
                break

            k += 1

        event = {

            "start_time":
                t[i],

            "end_time":
                t[k],

            "duration":
                t[k] - t[i],

            "x_pos":
                np.mean(
                    smooth_x[i:k + 1]
                ),

            "dispersion_x_deg":
                (
                    np.max(
                        smooth_x[i:k + 1]
                    )
                    -
                    np.min(
                        smooth_x[i:k + 1]
                    )
                ),

            "n_samples":
                k - i + 1,

            "threshold_used_deg":
                threshold_deg,
        }

        if frames is not None:

            event["start_frame"] = (
                frames[i]
            )

            event["end_frame"] = (
                frames[k]
            )

        events.append(
            event
        )

        i = k + 1

    return events


# =============================================================================
# COMPLETE DETECTOR
# =============================================================================

def detect_fixations(
    df: pd.DataFrame,
    config: FixationConfig | None = None,
) -> FixationResult:

    """
    Complete participant-level fixation detection.

    Input DataFrame must already contain:
        frame
        timestamp_ms_synced
        trial_number
        phase_type
        gaze_deg_x_calib
        valid_eye_frame

    Returns:
        FixationResult
    """

    if config is None:
        config = FixationConfig()
        
    # -----------------------------------------------------------------
    # Normalize trial column
    # -----------------------------------------------------------------

    if "trial_number" not in df.columns:

        if "trial_assignment" in df.columns:

            df = df.copy()
            df["trial_number"] = df["trial_assignment"]

        else:

            raise ValueError(
                "Missing trial information: expected "
                "'trial_number' or 'trial_assignment'."
            )


    required = [
        "timestamp_ms_synced",
        "trial_number",
        "phase_type",
        "gaze_deg_x_calib",
        "valid_eye_frame",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # -----------------------------------------------------------------
    # Estimate participant-specific threshold
    # -----------------------------------------------------------------

    (
        threshold_deg,
        noise_distribution,
    ) = estimate_adaptive_threshold(
        df=df,
        config=config,
    )

    # -----------------------------------------------------------------
    # Detect fixation events
    # -----------------------------------------------------------------

    rows = []

    fixation_id = 0

    for trial_number, trial in df.groupby(
        "trial_number",
        sort=True,
    ):

        segments = build_valid_segments(
            trial_df=trial,
            config=config,
        )

        for segment in segments:

            events = (
                detect_segment_fixations(
                    segment=segment,
                    threshold_deg=threshold_deg,
                    config=config,
                )
            )

            for event in events:

                fixation_id += 1

                event[
                    "fixation_id"
                ] = fixation_id

                event[
                    "trial_number"
                ] = trial_number

                rows.append(
                    event
                )

    fixations = pd.DataFrame(
        rows
    )

    # Stable column order
    if len(fixations):

        preferred_columns = [
            "fixation_id",
            "trial_number",
            "start_time",
            "end_time",
            "duration",
            "x_pos",
            "dispersion_x_deg",
            "n_samples",
            "start_frame",
            "end_frame",
            "threshold_used_deg",
        ]

        columns = [
            c
            for c in preferred_columns
            if c in fixations.columns
        ]

        fixations = fixations[
            columns
        ]

    return FixationResult(
        fixations=fixations,
        threshold_deg=threshold_deg,
        noise_distribution=noise_distribution,
        config=config,
    )