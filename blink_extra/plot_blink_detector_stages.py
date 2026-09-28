#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FRAME_LOG = Path(
    "/Volumes/Empra10/Ergebnisse60/beo7/test60/"
    "blink_filter_frame_log.csv"
)

PHASES = Path(
    "/Volumes/Empra10/Ergebnisse60/beo7/Analyse/"
    "Run_60hz_FullCalib_mode2_20260119_160844/"
    "phases_detected.json"
)

EYELINK = Path(
    "/Volumes/Empra10/Ergebnisse60/beo7/test60/"
    "eyelink_blink_benchmark.csv"
)

TRIAL = 1

OUTPUT = Path("beo7_trial1_blink_detector_stages.pdf")


def get_trial_window():
    with open(PHASES, "r") as f:
        data = json.load(f)

    trials = (
        data["phases"]["experiment_block1"]["trials"]
        + data["phases"]["experiment_block2"]["trials"]
    )

    trial = [
        t for t in trials
        if t["trial_number"] == TRIAL
        and not t.get("is_practice", False)
    ][0]

    start_ms = trial["fixation"]["start_video_s"] * 1000
    end_ms = trial["stimulus"]["end_video_s"] * 1000

    return start_ms, end_ms


start_ms, end_ms = get_trial_window()

df = pd.read_csv(FRAME_LOG)

df = df[
    (df["video_time_ms"] >= start_ms)
    & (df["video_time_ms"] <= end_ms)
].copy()

df["time_s"] = (df["video_time_ms"] - start_ms) / 1000

numeric_columns = [
    "left_ear",
    "right_ear",
    "avg_ear",
    "left_threshold",
    "right_threshold",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


eye = pd.read_csv(EYELINK)

eye = eye[
    pd.to_numeric(eye["trial"], errors="coerce") == TRIAL
].copy()


fig, axes = plt.subplots(
    4,
    1,
    figsize=(14, 11),
    sharex=True
)


# ---------------------------------------------------------
# 1. EAR
# ---------------------------------------------------------

ax = axes[0]

ax.plot(
    df["time_s"],
    df["left_ear"],
    label="Left EAR"
)

ax.plot(
    df["time_s"],
    df["right_ear"],
    label="Right EAR"
)

ax.plot(
    df["time_s"],
    df["avg_ear"],
    label="Average EAR",
    linewidth=2
)

ax.set_ylabel("EAR")
ax.set_title("1. EAR stored in blink_filter_frame_log.csv")
ax.legend()
ax.grid(True, alpha=0.3)


# ---------------------------------------------------------
# 2. Threshold
# ---------------------------------------------------------

ax = axes[1]

ax.plot(
    df["time_s"],
    df["left_ear"],
    label="Left EAR"
)

ax.plot(
    df["time_s"],
    df["right_ear"],
    label="Right EAR"
)

ax.plot(
    df["time_s"],
    df["left_threshold"],
    "--",
    label="Left threshold"
)

ax.plot(
    df["time_s"],
    df["right_threshold"],
    "--",
    label="Right threshold"
)

ax.set_ylabel("EAR")
ax.set_title("2. EAR vs adaptive thresholds")
ax.legend(ncol=2)
ax.grid(True, alpha=0.3)


# ---------------------------------------------------------
# 3. Closed-state logic
# ---------------------------------------------------------

ax = axes[2]

states = [
    ("left_closed_raw", 4),
    ("right_closed_raw", 3),
    ("left_closed_smooth", 2),
    ("right_closed_smooth", 1),
]

for column, level in states:

    values = (
        df[column]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    ax.step(
        df["time_s"],
        values.astype(int) * level,
        where="post",
        label=column
    )

ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels([
    "R smooth",
    "L smooth",
    "R raw",
    "L raw"
])

ax.set_title("3. Raw vs smoothed closed-eye states")
ax.legend(loc="upper right")
ax.grid(True, axis="x", alpha=0.3)


# ---------------------------------------------------------
# 4. Final invalid/blink decision
# ---------------------------------------------------------

ax = axes[3]

invalid = (
    df["valid_eye_frame"]
    .astype(str)
    .str.lower()
    .eq("false")
)

blink = (
    df["is_blink"]
    .astype(str)
    .str.lower()
    .eq("true")
)

suspected = (
    df["is_suspected_blink"]
    .astype(str)
    .str.lower()
    .eq("true")
)

transition = (
    df["is_transition_frame"]
    .astype(str)
    .str.lower()
    .eq("true")
)

ax.step(
    df["time_s"],
    invalid.astype(int) * 4,
    where="post",
    label="Invalid frame"
)

ax.step(
    df["time_s"],
    blink.astype(int) * 3,
    where="post",
    label="Blink"
)

ax.step(
    df["time_s"],
    suspected.astype(int) * 2,
    where="post",
    label="Suspected"
)

ax.step(
    df["time_s"],
    transition.astype(int),
    where="post",
    label="Transition"
)

ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels([
    "Transition",
    "Suspected",
    "Blink",
    "Invalid"
])

ax.set_title("4. Final BlinkDetector output")
ax.set_xlabel("Time from fixation start (s)")
ax.legend(loc="upper right")
ax.grid(True, axis="x", alpha=0.3)


# ---------------------------------------------------------
# EyeLink regions on every panel
# ---------------------------------------------------------

for _, row in eye.iterrows():

    start = row["start_from_fixation_ms"] / 1000
    end = row["end_from_fixation_ms"] / 1000

    for ax in axes:
        ax.axvspan(
            start,
            end,
            alpha=0.12
        )


# Manual blink estimates
for t in [0.7, 3.7, 7.7]:

    for ax in axes:
        ax.axvline(
            t,
            linestyle=":",
            linewidth=1.2
        )


fig.suptitle(
    "beo7 — Trial 1 — BlinkDetector stage diagnosis\n"
    "Shaded = EyeLink EBLINK | dotted = manually observed blink"
)

fig.tight_layout()

fig.savefig(
    OUTPUT,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Saved: {OUTPUT}")