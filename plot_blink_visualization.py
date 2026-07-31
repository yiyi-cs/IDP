#!/usr/bin/env python3
"""
Create trial-level blink-comparison figures for the 25 Hz FullCalib pipeline.

For one configured trial, every VP contributes two panels:

A. Three gaze lines (MediaPipe, PTGaze, EyeLink) plus the webcam-based
   blink-filter regions produced by export_blink_filter_logs.py.

B. The same three gaze lines plus all EyeLink EBLINK benchmark intervals.

Page layout (two VPs per page):

    row 1: VP 1 — A                         VP 2 — A
    row 2: VP 1 — B                         VP 2 — B

Inputs per VP
-------------
/Volumes/Empra10/Ergebnisse60/<VP>/Analyse/Run_60hz_FullCalib_*/
    debug_3_eyetracker_data.csv
    debug_5_pupil_data_calibrated.csv
    debug_5_ptgaze_calibrated.csv
    phases_detected.json

/Volumes/Empra10/Ergebnisse60/<VP>/test60/
    blink_filter_regions.csv
    eyelink_blink_benchmark.csv
    eyelink_trial_markers.csv

Outputs
-------
/Volumes/Empra10/Ergebnisse60/Blink_Visualization_60hz_FullCalib/
    trial_<NN>/trial_<NN>_blink_comparison_all_vps.pdf
    trial_<NN>/selected_input_files.csv
    trial_<NN>/plot_processing_summary.csv

The PDF is multi-page. Each page contains up to six VPs:

    row 1: A panels for VPs 1-3
    row 2: B panels for VPs 1-3
    row 3: A panels for VPs 4-6
    row 4: B panels for VPs 4-6

Timing
------
All gaze lines, phase boundaries, webcam blink regions, and EyeLink blink
events are plotted in one common EyeLink tracker clock:

    relative time = tracker time - EyeLink fixation-start marker

MediaPipe and PTGaze use timestamp_ms_synced from the original pipeline.
EyeLink can be sampled onto the same 25 Hz camera time grid, which is the
default and is closest to the original paper's combined-data plotting.

No individual figures or PNG files are generated.

Required packages
-----------------
    /opt/homebrew/bin/python3.10 -m pip install pandas numpy matplotlib
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import numpy as np
    import pandas as pd

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.backends.backend_pdf import PdfPages
except ModuleNotFoundError as exc:
    missing = exc.name or "unknown package"
    raise SystemExit(
        f"Missing Python package: {missing}\n"
        "Install with:\n"
        "  /opt/homebrew/bin/python3.10 -m pip install "
        "pandas numpy matplotlib"
    ) from exc


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

HZ = 60
EMPRA = 10

RESULTS_ROOT = Path(f"/Volumes/Empra{EMPRA}/Ergebnisse{HZ}")
RUN_REGEX = rf"^Run_{HZ}hz_FullCalib_"
TEST_SUBDIR = f"test{HZ}"

# Change this value to select the one trial processed in the current run.
TRIAL_NUMBER = 1

# VPs shown in the output. Their order determines page pairing.
VP_CODES = [
    "beo7",
    "bjs4",
    "egf5",
    "fbn6",
    "fgt6",
    "jkl7",
    "kdn8",
    # "kly9",
    "kro3",
    # "ldj9",
    "mhe9",
    "oem4",
    "ogt7",
]

OUTPUT_ROOT = RESULTS_ROOT / "Blink_Visualization_{HZ}hz_FullCalib"

# Horizontal EyeLink pixel -> visual-angle conversion.
# Keep these synchronized with the experiment/pipeline configuration.
SCREEN_WIDTH_PX = 1920.0
SCREEN_WIDTH_CM = 53.2
VIEWING_DISTANCE_CM = 70.0
SCREEN_CENTER_X_PX = (SCREEN_WIDTH_PX - 1.0) / 2.0

# EyeLink gaze display:
#
#   "camera_grid":
#       Sample EyeLink gaze at the synchronized MediaPipe camera timestamps.
#       This produces approximately one EyeLink value per 25 Hz frame and is
#       the closest match to the original paper's combined-data workflow.
#
#   "raw":
#       Plot the original EyeLink samples in tracker time.
EYELINK_DISPLAY_MODE = "camera_grid"

# Maximum allowed difference between a synchronized camera timestamp and the
# nearest EyeLink timestamp when camera_grid mode is used. EyeLink timestamps
# are integer milliseconds while camera timestamps may have decimals.
EYELINK_MATCH_TOLERANCE_MS = 1.0

# Used only in raw mode. Increase to 2, 4, ... if raw 2000 Hz plotting is slow.
EYELINK_PLOT_STEP = 1

# The original paper used a fixed 0-10 s horizontal axis. Set to None to use
# the actual EyeLink fixation-start to stimulus-end duration instead.
FIXED_X_MAX_S: Optional[float] = 10.0

# A true 0 ms EBLINK interval has no visible width in axvspan. Keep the raw
# duration in the benchmark CSV, but give such events a tiny display width so
# every extracted event is still visible in panel B.
MIN_EYELINK_EVENT_DISPLAY_WIDTH_S = 0.010

# "fixation" is the agreed comparison axis: fixation start = 0 s.
TIME_ZERO_PHASE = "fixation"

# False keeps gaze lines visible underneath the A-panel background regions.
MASK_LINES_IN_OWN_BLINK_REGIONS = False

SHOW_PHASE_LABELS = True
SHOW_OWN_REGION_LABELS = True
SHOW_EYELINK_REGION_LABELS = True

# None calculates one robust common y range for every A/B panel in this run.
# A fixed example would be: Y_LIMITS = (-25.0, 25.0)
Y_LIMITS: Optional[tuple[float, float]] = None

CONTINUE_ON_ERROR = True
OUTPUT_DPI = 300
INDIVIDUAL_FIGSIZE = (9.2, 4.6)
PAGE_FIGSIZE = (16.8, 13.0)
VP_COLUMNS_PER_PAGE = 3
VPS_PER_PAGE = 6

# Exact input filenames.
EYELINK_GAZE_FILE = "debug_3_eyetracker_data.csv"
MEDIAPIPE_FILE = "debug_5_pupil_data_calibrated.csv"
PTGAZE_FILE = "debug_5_ptgaze_calibrated.csv"
PHASES_FILE = "phases_detected.json"
OWN_BLINK_FILE = "blink_filter_regions.csv"
EYELINK_BLINK_FILE = "eyelink_blink_benchmark.csv"
EYELINK_MARKERS_FILE = "eyelink_trial_markers.csv"

REQUIRED_RUN_FILES = (
    EYELINK_GAZE_FILE,
    MEDIAPIPE_FILE,
    PTGAZE_FILE,
    PHASES_FILE,
)

METHOD_ORDER = ("MediaPipe", "PTGaze", "EyeLink")
METHOD_STYLES = {
    "MediaPipe": {"color": "#D73027", "linewidth": 0.85, "alpha": 0.95},
    "PTGaze": {"color": "#4575B4", "linewidth": 0.85, "alpha": 0.95},
    "EyeLink": {"color": "#111111", "linewidth": 0.72, "alpha": 0.82},
}

OWN_BLINK_STYLES = {
    "blink": {
        "facecolor": "#D73027",
        "edgecolor": "#8B1A14",
        "alpha": 0.20,
        "linestyle": "-",
        "hatch": None,
        "short_label": "B",
        "label": "B — Webcam blink / eyes closed",
    },
    "suspected": {
        "facecolor": "#F0A202",
        "edgecolor": "#9A6500",
        "alpha": 0.19,
        "linestyle": "--",
        "hatch": "////",
        "short_label": "S",
        "label": "S — Suspected / transition",
    },
    "invalid": {
        "facecolor": "#7B4AB5",
        "edgecolor": "#4D2B78",
        "alpha": 0.16,
        "linestyle": ":",
        "hatch": "xxxx",
        "short_label": "I",
        "label": "I — Other invalid",
    },
}

EYELINK_BLINK_STYLE = {
    "facecolor": "#009E73",
    "edgecolor": "#006B4F",
    "alpha": 0.22,
    "linestyle": "-",
    "hatch": None,
    "short_label": "E",
    "label": "E — EyeLink ASC EBLINK event",
}


# =============================================================================
# 2. DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class InputPaths:
    vp_code: str
    run_dir: Path
    run_name: str
    eyelink_gaze_file: Path
    mediapipe_file: Path
    ptgaze_file: Path
    phases_file: Path
    own_blink_file: Path
    eyelink_blink_file: Path
    eyelink_markers_file: Path


@dataclass
class PlotPayload:
    vp_code: str
    trial_number: int
    run_name: str
    line_data: pd.DataFrame
    own_blink_data: pd.DataFrame
    eyelink_blink_data: pd.DataFrame
    fixation_start_s: float
    fixation_end_s: float
    stimulus_start_s: float
    stimulus_end_s: float
    x_limits: tuple[float, float]
    camera_fps: float
    own_blink_fps: float
    n_eyelink_samples: int
    sync_offset_ms: float
    sync_offset_spread_ms: float
    fixation_alignment_error_ms: float
    stimulus_alignment_error_ms: float
    eyelink_display_mode: str


# =============================================================================
# 3. VALIDATION AND FILE DISCOVERY
# =============================================================================

def validate_settings() -> None:
    if not RESULTS_ROOT.is_dir():
        raise FileNotFoundError(f"RESULTS_ROOT does not exist: {RESULTS_ROOT}")
    if TRIAL_NUMBER < 1:
        raise ValueError("TRIAL_NUMBER must be >= 1.")
    if not VP_CODES:
        raise ValueError("VP_CODES is empty.")
    if len(VP_CODES) != len(set(VP_CODES)):
        raise ValueError("VP_CODES contains duplicates.")
    if EYELINK_DISPLAY_MODE not in {"camera_grid", "raw"}:
        raise ValueError(
            'EYELINK_DISPLAY_MODE must be "camera_grid" or "raw".'
        )
    if EYELINK_MATCH_TOLERANCE_MS <= 0:
        raise ValueError("EYELINK_MATCH_TOLERANCE_MS must be > 0.")
    if EYELINK_PLOT_STEP < 1:
        raise ValueError("EYELINK_PLOT_STEP must be >= 1.")
    if FIXED_X_MAX_S is not None and FIXED_X_MAX_S <= 0:
        raise ValueError("FIXED_X_MAX_S must be positive or None.")
    re.compile(RUN_REGEX)


def iter_directories(root: Path) -> Iterable[Path]:
    yield root
    for candidate in root.rglob("*"):
        if candidate.is_dir():
            yield candidate


def find_latest_matching_run(vp_dir: Path) -> Path:
    analyse_dir = vp_dir / "Analyse"
    if not analyse_dir.is_dir():
        raise FileNotFoundError(f"Analyse directory not found: {analyse_dir}")

    pattern = re.compile(RUN_REGEX)
    candidates = [
        directory
        for directory in iter_directories(analyse_dir)
        if pattern.search(directory.name)
    ]
    valid = [
        directory
        for directory in candidates
        if all((directory / filename).is_file() for filename in REQUIRED_RUN_FILES)
    ]

    if not valid:
        required = "\n  ".join(REQUIRED_RUN_FILES)
        raise FileNotFoundError(
            f"No run matching {RUN_REGEX!r} with all required files under:\n"
            f"  {analyse_dir}\nRequired:\n  {required}"
        )

    selected = max(valid, key=lambda path: path.stat().st_mtime)
    if len(valid) > 1:
        print(
            f"[INFO] {vp_dir.name}: multiple matching FullCalib runs; "
            f"using newest: {selected.name}"
        )
    return selected


def discover_inputs(vp_code: str) -> InputPaths:
    vp_dir = RESULTS_ROOT / vp_code
    if not vp_dir.is_dir():
        raise FileNotFoundError(f"VP directory not found: {vp_dir}")

    run_dir = find_latest_matching_run(vp_dir)
    test_dir = vp_dir / TEST_SUBDIR

    paths = InputPaths(
        vp_code=vp_code,
        run_dir=run_dir,
        run_name=run_dir.name,
        eyelink_gaze_file=run_dir / EYELINK_GAZE_FILE,
        mediapipe_file=run_dir / MEDIAPIPE_FILE,
        ptgaze_file=run_dir / PTGAZE_FILE,
        phases_file=run_dir / PHASES_FILE,
        own_blink_file=test_dir / OWN_BLINK_FILE,
        eyelink_blink_file=test_dir / EYELINK_BLINK_FILE,
        eyelink_markers_file=test_dir / EYELINK_MARKERS_FILE,
    )

    missing = [
        path
        for path in (
            paths.eyelink_gaze_file,
            paths.mediapipe_file,
            paths.ptgaze_file,
            paths.phases_file,
            paths.own_blink_file,
            paths.eyelink_blink_file,
            paths.eyelink_markers_file,
        )
        if not path.is_file()
    ]
    if missing:
        missing_text = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing input file(s) for {vp_code}:\n  {missing_text}"
        )

    return paths


def assert_columns(
    data: pd.DataFrame,
    required: Iterable[str],
    file_path: Path,
) -> None:
    missing = set(required).difference(data.columns)
    if missing:
        raise KeyError(
            f"Missing columns in {file_path}: {sorted(missing)}"
        )


# =============================================================================
# 4. READ AND NORMALIZE INPUT DATA
# =============================================================================

def read_calibrated_method(
    file_path: Path,
    method_name: str,
    trial_number: int,
) -> pd.DataFrame:
    """
    Read one calibrated camera method.

    timestamp_ms is the original video clock.
    timestamp_ms_synced is the same frame mapped into EyeLink tracker time by
    the original synchronization pipeline. All final plotting uses the latter.
    """
    data = pd.read_csv(file_path)
    required = (
        "frame",
        "timestamp_ms",
        "timestamp_ms_synced",
        "trial_number",
        "phase_type",
        "gaze_deg_x_calib",
    )
    assert_columns(data, required, file_path)

    normalized = pd.DataFrame(
        {
            "frame": pd.to_numeric(
                data["frame"], errors="coerce"
            ).astype("Int64"),
            "timestamp_ms": pd.to_numeric(
                data["timestamp_ms"], errors="coerce"
            ),
            "timestamp_ms_synced": pd.to_numeric(
                data["timestamp_ms_synced"], errors="coerce"
            ),
            "trial_number": pd.to_numeric(
                data["trial_number"], errors="coerce"
            ).astype("Int64"),
            "phase_type": data["phase_type"].astype("string"),
            "gaze_deg_x": pd.to_numeric(
                data["gaze_deg_x_calib"], errors="coerce"
            ),
            "method": method_name,
        }
    )

    selected = normalized.loc[
        normalized["trial_number"].eq(trial_number)
    ].copy()

    return selected.dropna(
        subset=["timestamp_ms", "timestamp_ms_synced"]
    )


def read_trial_boundaries(phases_file: Path) -> pd.DataFrame:
    with phases_file.open("r", encoding="utf-8") as handle:
        phase_json = json.load(handle)

    phases = phase_json.get("phases", {})
    block1 = phases.get("experiment_block1", {}).get("trials", [])
    block2 = phases.get("experiment_block2", {}).get("trials", [])
    all_trials = [*block1, *block2]

    if not all_trials:
        raise ValueError(f"No experiment trials found in {phases_file}")

    rows: list[dict[str, Any]] = []
    for trial in all_trials:
        fixation = trial.get("fixation", {})
        stimulus = trial.get("stimulus", {})
        rows.append(
            {
                "trial_number": int(trial["trial_number"]),
                "is_practice": bool(trial.get("is_practice", False)),
                "fixation_start_video_ms": float(fixation["start_video_s"]) * 1000.0,
                "fixation_end_video_ms": float(fixation["end_video_s"]) * 1000.0,
                "stimulus_start_video_ms": float(stimulus["start_video_s"]) * 1000.0,
                "stimulus_end_video_ms": float(stimulus["end_video_s"]) * 1000.0,
            }
        )
    return pd.DataFrame(rows)


def select_camera_boundary(
    phases_file: Path,
    trial_number: int,
) -> pd.Series:
    boundaries = read_trial_boundaries(phases_file)
    selected = boundaries.loc[
        boundaries["trial_number"].eq(trial_number)
        & ~boundaries["is_practice"]
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one non-practice camera boundary for Trial "
            f"{trial_number}; found {len(selected)}"
        )
    return selected.iloc[0]


def select_eyelink_marker(
    marker_file: Path,
    trial_number: int,
) -> pd.Series:
    data = pd.read_csv(marker_file)
    required = (
        "trial",
        "is_practice",
        "fixation_start_tracker_ms",
        "fixation_end_tracker_ms",
        "stimulus_start_tracker_ms",
        "stimulus_end_tracker_ms",
    )
    assert_columns(data, required, marker_file)

    trial_numeric = pd.to_numeric(data["trial"], errors="coerce")
    practice = data["is_practice"].astype("string").str.lower().eq("true")
    selected = data.loc[trial_numeric.eq(trial_number) & ~practice].copy()

    if len(selected) != 1:
        raise ValueError(
            f"Expected one non-practice EyeLink marker set for Trial "
            f"{trial_number}; found {len(selected)} in {marker_file}"
        )
    return selected.iloc[0]


def eyelink_x_px_to_deg(x_px: pd.Series) -> pd.Series:
    x_px_numeric = pd.to_numeric(x_px, errors="coerce")
    cm_per_px = SCREEN_WIDTH_CM / SCREEN_WIDTH_PX
    x_cm = (x_px_numeric - SCREEN_CENTER_X_PX) * cm_per_px
    return np.degrees(np.arctan2(x_cm, VIEWING_DISTANCE_CM))


def read_eyelink_gaze_raw(
    gaze_file: Path,
    tracker_start_ms: float,
    tracker_end_ms: float,
) -> pd.DataFrame:
    """
    Read EyeLink gaze in the native tracker clock.

    Duplicate integer timestamps are averaged. This is appropriate for a
    2000 Hz recording stored with millisecond timestamps: typically two
    samples share one integer millisecond.
    """
    data = pd.read_csv(
        gaze_file,
        usecols=lambda name: name in {
            "timestamp_ms", "x_pos", "y_pos", "pupil_size", "block_type"
        },
    )
    assert_columns(data, ("timestamp_ms", "x_pos"), gaze_file)

    data["timestamp_ms"] = pd.to_numeric(
        data["timestamp_ms"], errors="coerce"
    )
    data["gaze_deg_x"] = eyelink_x_px_to_deg(data["x_pos"])

    selected = data.loc[
        data["timestamp_ms"].ge(tracker_start_ms)
        & data["timestamp_ms"].le(tracker_end_ms)
    ].copy()

    # Preserve missing gaze during EyeLink sample loss/blinks. Pandas mean
    # returns NaN when every sample at that timestamp is missing.
    selected = (
        selected.groupby("timestamp_ms", as_index=False, sort=True)
        .agg(gaze_deg_x=("gaze_deg_x", "mean"))
    )
    # merge_asof requires identical key dtypes on both sides.
    selected["timestamp_ms"] = selected["timestamp_ms"].astype("float64")
    selected["method"] = "EyeLink"
    return selected


def align_eyelink_to_camera_grid(
    eyelink_raw: pd.DataFrame,
    camera_tracker_times_ms: pd.Series,
    tracker_zero_ms: float,
) -> pd.DataFrame:
    """
    Match EyeLink gaze to the synchronized 25 Hz camera timestamps.

    The merge includes EyeLink rows with NaN gaze, so camera frames occurring
    during EyeLink sample loss remain NaN rather than being interpolated
    across a blink.
    """
    camera_grid = pd.DataFrame(
        {
            "timestamp_ms": np.sort(
                pd.to_numeric(
                    camera_tracker_times_ms,
                    errors="coerce",
                ).dropna().unique()
            )
        }
    )

    if camera_grid.empty:
        raise ValueError("Camera tracker-time grid is empty.")

    eyelink_sorted = eyelink_raw.sort_values("timestamp_ms").copy()

    aligned = pd.merge_asof(
        camera_grid,
        eyelink_sorted[["timestamp_ms", "gaze_deg_x"]],
        on="timestamp_ms",
        direction="nearest",
        tolerance=float(EYELINK_MATCH_TOLERANCE_MS),
    )
    aligned["time_s"] = (
        aligned["timestamp_ms"] - tracker_zero_ms
    ) / 1000.0
    aligned["method"] = "EyeLink"

    return aligned[
        ["timestamp_ms", "time_s", "gaze_deg_x", "method"]
    ]


def prepare_eyelink_gaze(
    gaze_file: Path,
    tracker_start_ms: float,
    tracker_end_ms: float,
    camera_tracker_times_ms: pd.Series,
) -> pd.DataFrame:
    raw = read_eyelink_gaze_raw(
        gaze_file=gaze_file,
        tracker_start_ms=tracker_start_ms,
        tracker_end_ms=tracker_end_ms,
    )

    if EYELINK_DISPLAY_MODE == "camera_grid":
        return align_eyelink_to_camera_grid(
            eyelink_raw=raw,
            camera_tracker_times_ms=camera_tracker_times_ms,
            tracker_zero_ms=tracker_start_ms,
        )

    raw = raw.iloc[::EYELINK_PLOT_STEP].copy()
    raw["time_s"] = (
        raw["timestamp_ms"] - tracker_start_ms
    ) / 1000.0
    return raw[
        ["timestamp_ms", "time_s", "gaze_deg_x", "method"]
    ]


def read_own_blink_regions(
    file_path: Path,
    sync_offset_ms: float,
    tracker_start_ms: float,
    tracker_end_ms: float,
    x_limits: tuple[float, float],
) -> pd.DataFrame:
    """
    Convert webcam blink regions from video time into EyeLink tracker time.

        tracker_ms = video_ms + sync_offset_ms
        relative_s = (tracker_ms - EyeLink fixation start) / 1000
    """
    data = pd.read_csv(file_path)
    required = (
        "start_video_time_ms",
        "end_video_time_ms",
        "reason_group",
    )
    assert_columns(data, required, file_path)

    normalized = pd.DataFrame(
        {
            "start_video_time_ms": pd.to_numeric(
                data["start_video_time_ms"], errors="coerce"
            ),
            "end_video_time_ms": pd.to_numeric(
                data["end_video_time_ms"], errors="coerce"
            ),
            "reason_group": data["reason_group"].astype("string"),
            "invalid_reason": (
                data["invalid_reason"].astype("string")
                if "invalid_reason" in data.columns
                else pd.Series(pd.NA, index=data.index, dtype="string")
            ),
            "video_fps": (
                pd.to_numeric(data["video_fps"], errors="coerce")
                if "video_fps" in data.columns
                else pd.Series(np.nan, index=data.index, dtype="float64")
            ),
        }
    ).dropna(subset=["start_video_time_ms", "end_video_time_ms"])

    normalized["reason_group"] = normalized["reason_group"].map(
        lambda group: group if group in {"blink", "suspected"} else "invalid"
    )
    normalized["start_tracker_ms"] = (
        normalized["start_video_time_ms"] + sync_offset_ms
    )
    normalized["end_tracker_ms"] = (
        normalized["end_video_time_ms"] + sync_offset_ms
    )

    selected = normalized.loc[
        normalized["end_tracker_ms"].gt(tracker_start_ms)
        & normalized["start_tracker_ms"].lt(tracker_end_ms)
    ].copy()

    if selected.empty:
        selected["start_s"] = pd.Series(dtype="float64")
        selected["end_s"] = pd.Series(dtype="float64")
        return selected

    selected["start_tracker_ms"] = np.maximum(
        selected["start_tracker_ms"].to_numpy(float),
        tracker_start_ms,
    )
    selected["end_tracker_ms"] = np.minimum(
        selected["end_tracker_ms"].to_numpy(float),
        tracker_end_ms,
    )
    selected["start_s"] = (
        selected["start_tracker_ms"] - tracker_start_ms
    ) / 1000.0
    selected["end_s"] = (
        selected["end_tracker_ms"] - tracker_start_ms
    ) / 1000.0

    left, right = x_limits
    selected["start_s"] = np.maximum(
        selected["start_s"].to_numpy(float), left
    )
    selected["end_s"] = np.minimum(
        selected["end_s"].to_numpy(float), right
    )

    return selected.loc[
        selected["end_s"].gt(selected["start_s"])
    ].copy()


def read_eyelink_blink_regions(
    file_path: Path,
    trial_number: int,
    x_limits: tuple[float, float],
) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    required = (
        "trial",
        "start_from_fixation_s",
        "end_from_fixation_s",
    )
    assert_columns(data, required, file_path)

    trial_numeric = pd.to_numeric(data["trial"], errors="coerce")
    selected = data.loc[trial_numeric.eq(trial_number)].copy()
    selected["start_s"] = pd.to_numeric(
        selected["start_from_fixation_s"], errors="coerce"
    )
    selected["end_s"] = pd.to_numeric(
        selected["end_from_fixation_s"], errors="coerce"
    )
    selected = selected.dropna(subset=["start_s", "end_s"])

    left, right = x_limits
    selected["actual_start_s"] = selected["start_s"]
    selected["actual_end_s"] = selected["end_s"]
    selected["start_s"] = np.maximum(selected["start_s"].to_numpy(float), left)
    selected["end_s"] = np.minimum(selected["end_s"].to_numpy(float), right)

    # Preserve all EBLINK events in the visualization. Zero-duration events
    # remain unchanged in the source CSV but receive a tiny plotting width.
    non_visible = selected["end_s"].le(selected["start_s"])
    can_extend_right = non_visible & selected["start_s"].lt(right)
    selected.loc[can_extend_right, "end_s"] = np.minimum(
        selected.loc[can_extend_right, "start_s"]
        + MIN_EYELINK_EVENT_DISPLAY_WIDTH_S,
        right,
    )
    at_right_edge = non_visible & ~can_extend_right & selected["end_s"].ge(left)
    selected.loc[at_right_edge, "start_s"] = np.maximum(
        left,
        right - MIN_EYELINK_EVENT_DISPLAY_WIDTH_S,
    )
    selected.loc[at_right_edge, "end_s"] = right

    return selected.loc[selected["end_s"].gt(selected["start_s"])].copy()


def estimate_fps(timestamp_ms: pd.Series) -> float:
    values = np.sort(pd.to_numeric(timestamp_ms, errors="coerce").dropna().unique())
    differences = np.diff(values)
    differences = differences[np.isfinite(differences) & (differences > 0)]
    if differences.size == 0:
        return float("nan")
    return float(1000.0 / np.median(differences))


def own_blink_fps(blink_data: pd.DataFrame) -> float:
    if "video_fps" not in blink_data.columns:
        return float("nan")
    values = pd.to_numeric(blink_data["video_fps"], errors="coerce")
    values = values[np.isfinite(values)].dropna().unique()
    return float(values[0]) if len(values) else float("nan")


def calculate_sync_offset_ms(
    camera_data: pd.DataFrame,
) -> tuple[float, float]:
    """
    Calculate video-clock -> EyeLink-clock offset.

    The original pipeline currently uses a constant offset:
        timestamp_ms_synced = timestamp_ms + offset

    Return:
        median offset, robust peak-to-peak spread
    """
    offsets = pd.to_numeric(
        camera_data["timestamp_ms_synced"], errors="coerce"
    ) - pd.to_numeric(
        camera_data["timestamp_ms"], errors="coerce"
    )
    offsets = offsets.dropna()

    if offsets.empty:
        raise ValueError("No finite synchronization offsets found.")

    median_offset = float(offsets.median())
    spread = float(offsets.max() - offsets.min())

    return median_offset, spread


def first_phase_alignment_error_ms(
    camera_data: pd.DataFrame,
    phase_name: str,
    tracker_marker_ms: float,
) -> float:
    selected = camera_data.loc[
        camera_data["phase_type"].astype("string").str.lower().eq(
            phase_name.lower()
        )
    ]
    values = pd.to_numeric(
        selected["timestamp_ms_synced"], errors="coerce"
    ).dropna()

    if values.empty:
        return float("nan")

    return float(values.min() - tracker_marker_ms)


# =============================================================================
# 5. PREPARE ONE VP
# =============================================================================

def mask_camera_lines(
    line_data: pd.DataFrame,
    regions: pd.DataFrame,
) -> pd.DataFrame:
    if regions.empty:
        return line_data

    masked = line_data.copy()
    for method in ("MediaPipe", "PTGaze"):
        method_mask = masked["method"].eq(method)
        time_values = masked.loc[method_mask, "time_s"].to_numpy(float)
        invalid = np.zeros(time_values.size, dtype=bool)
        for region in regions.itertuples(index=False):
            invalid |= (
                (time_values >= float(region.start_s))
                & (time_values < float(region.end_s))
            )
        method_indices = masked.index[method_mask]
        masked.loc[method_indices[invalid], "gaze_deg_x"] = np.nan
    return masked


def prepare_one_vp(paths: InputPaths) -> PlotPayload:
    print(
        f"\n[LOAD] {paths.vp_code} | Trial {TRIAL_NUMBER} | "
        f"{paths.run_name}"
    )

    # Camera phase information is retained only for diagnostics. Final phase
    # boundaries and the common zero point come from EyeLink ASC markers.
    camera_boundary = select_camera_boundary(
        paths.phases_file,
        TRIAL_NUMBER,
    )
    eyelink_marker = select_eyelink_marker(
        paths.eyelink_markers_file,
        TRIAL_NUMBER,
    )

    tracker_fixation_start_ms = float(
        eyelink_marker["fixation_start_tracker_ms"]
    )
    tracker_fixation_end_ms = float(
        eyelink_marker["fixation_end_tracker_ms"]
    )
    tracker_stimulus_start_ms = float(
        eyelink_marker["stimulus_start_tracker_ms"]
    )
    tracker_stimulus_end_ms = float(
        eyelink_marker["stimulus_end_tracker_ms"]
    )

    natural_end_s = (
        tracker_stimulus_end_ms - tracker_fixation_start_ms
    ) / 1000.0
    plot_end_s = (
        float(FIXED_X_MAX_S)
        if FIXED_X_MAX_S is not None
        else natural_end_s
    )
    x_limits = (0.0, plot_end_s)

    mediapipe = read_calibrated_method(
        paths.mediapipe_file,
        "MediaPipe",
        TRIAL_NUMBER,
    )
    ptgaze = read_calibrated_method(
        paths.ptgaze_file,
        "PTGaze",
        TRIAL_NUMBER,
    )

    if mediapipe.empty:
        raise ValueError(
            f"No MediaPipe rows for {paths.vp_code} "
            f"Trial {TRIAL_NUMBER}"
        )
    if ptgaze.empty:
        raise ValueError(
            f"No PTGaze rows for {paths.vp_code} "
            f"Trial {TRIAL_NUMBER}"
        )

    # Both methods should carry the same synchronization transform. Estimate
    # from MediaPipe and validate PTGaze against it.
    mp_offset_ms, mp_spread_ms = calculate_sync_offset_ms(mediapipe)
    pt_offset_ms, pt_spread_ms = calculate_sync_offset_ms(ptgaze)

    if abs(mp_offset_ms - pt_offset_ms) > 0.5:
        raise ValueError(
            f"MediaPipe/PTGaze synchronization offsets differ for "
            f"{paths.vp_code}: {mp_offset_ms:.3f} vs "
            f"{pt_offset_ms:.3f} ms"
        )

    sync_offset_ms = float(np.median([mp_offset_ms, pt_offset_ms]))
    sync_offset_spread_ms = max(mp_spread_ms, pt_spread_ms)

    camera_lines = pd.concat(
        (mediapipe, ptgaze),
        ignore_index=True,
    )
    camera_lines = camera_lines.loc[
        camera_lines["timestamp_ms_synced"].ge(
            tracker_fixation_start_ms
        )
        & camera_lines["timestamp_ms_synced"].le(
            tracker_stimulus_end_ms
        )
    ].copy()

    camera_lines["time_s"] = (
        camera_lines["timestamp_ms_synced"]
        - tracker_fixation_start_ms
    ) / 1000.0

    # EyeLink is displayed on the synchronized MediaPipe 25 Hz grid by
    # default, matching the structure of the original combined-data plot.
    camera_grid_tracker_times = mediapipe.loc[
        mediapipe["timestamp_ms_synced"].ge(
            tracker_fixation_start_ms
        )
        & mediapipe["timestamp_ms_synced"].le(
            tracker_stimulus_end_ms
        ),
        "timestamp_ms_synced",
    ]

    eyelink = prepare_eyelink_gaze(
        gaze_file=paths.eyelink_gaze_file,
        tracker_start_ms=tracker_fixation_start_ms,
        tracker_end_ms=tracker_stimulus_end_ms,
        camera_tracker_times_ms=camera_grid_tracker_times,
    )

    camera_for_plot = camera_lines.rename(
        columns={"timestamp_ms_synced": "tracker_timestamp_ms"}
    )
    camera_for_plot["timestamp_ms"] = camera_for_plot[
        "tracker_timestamp_ms"
    ]

    line_data = pd.concat(
        (
            camera_for_plot[
                ["timestamp_ms", "time_s", "gaze_deg_x", "method"]
            ],
            eyelink,
        ),
        ignore_index=True,
    )
    line_data = line_data.loc[
        line_data["time_s"].ge(x_limits[0])
        & line_data["time_s"].le(x_limits[1])
    ].sort_values(
        ["method", "time_s"],
        kind="stable",
    )

    own_regions = read_own_blink_regions(
        file_path=paths.own_blink_file,
        sync_offset_ms=sync_offset_ms,
        tracker_start_ms=tracker_fixation_start_ms,
        tracker_end_ms=tracker_stimulus_end_ms,
        x_limits=x_limits,
    )
    eyelink_regions = read_eyelink_blink_regions(
        paths.eyelink_blink_file,
        TRIAL_NUMBER,
        x_limits,
    )

    if MASK_LINES_IN_OWN_BLINK_REGIONS:
        line_data = mask_camera_lines(line_data, own_regions)

    camera_fps = estimate_fps(mediapipe["timestamp_ms"])
    blink_fps = own_blink_fps(own_regions)

    if (
        math.isfinite(camera_fps)
        and math.isfinite(blink_fps)
        and abs(camera_fps - blink_fps) > 0.5
    ):
        raise ValueError(
            f"FPS mismatch for {paths.vp_code}: calibrated data ~"
            f"{camera_fps:.3f} Hz, own blink regions "
            f"{blink_fps:.3f} Hz"
        )

    fixation_alignment_error_ms = first_phase_alignment_error_ms(
        mediapipe,
        "fixation",
        tracker_fixation_start_ms,
    )
    stimulus_alignment_error_ms = first_phase_alignment_error_ms(
        mediapipe,
        "stimulus",
        tracker_stimulus_start_ms,
    )

    # Also compare the phase JSON boundaries after applying the pipeline
    # offset. This is diagnostic only and does not change the plot.
    camera_fixation_from_json_tracker = (
        float(camera_boundary["fixation_start_video_ms"])
        + sync_offset_ms
    )
    camera_stimulus_from_json_tracker = (
        float(camera_boundary["stimulus_start_video_ms"])
        + sync_offset_ms
    )
    json_fix_error = (
        camera_fixation_from_json_tracker
        - tracker_fixation_start_ms
    )
    json_stim_error = (
        camera_stimulus_from_json_tracker
        - tracker_stimulus_start_ms
    )

    print(
        f"[SYNC] {paths.vp_code}: offset={sync_offset_ms:.3f} ms"
        f" | offset spread={sync_offset_spread_ms:.6f} ms"
    )
    print(
        f"       first camera fixation frame - ASC fixation marker: "
        f"{fixation_alignment_error_ms:.3f} ms"
    )
    print(
        f"       first camera stimulus frame - ASC stimulus marker: "
        f"{stimulus_alignment_error_ms:.3f} ms"
    )
    print(
        f"       phase JSON fixation/stimulus errors after sync: "
        f"{json_fix_error:.3f} / {json_stim_error:.3f} ms"
    )
    print(
        f"       EyeLink display mode: {EYELINK_DISPLAY_MODE}"
        f" | plotted EyeLink points: {len(eyelink)}"
    )

    return PlotPayload(
        vp_code=paths.vp_code,
        trial_number=TRIAL_NUMBER,
        run_name=paths.run_name,
        line_data=line_data.reset_index(drop=True),
        own_blink_data=own_regions.reset_index(drop=True),
        eyelink_blink_data=eyelink_regions.reset_index(drop=True),
        fixation_start_s=0.0,
        fixation_end_s=(
            tracker_fixation_end_ms
            - tracker_fixation_start_ms
        ) / 1000.0,
        stimulus_start_s=(
            tracker_stimulus_start_ms
            - tracker_fixation_start_ms
        ) / 1000.0,
        stimulus_end_s=natural_end_s,
        x_limits=x_limits,
        camera_fps=camera_fps,
        own_blink_fps=blink_fps,
        n_eyelink_samples=int(len(eyelink)),
        sync_offset_ms=sync_offset_ms,
        sync_offset_spread_ms=sync_offset_spread_ms,
        fixation_alignment_error_ms=fixation_alignment_error_ms,
        stimulus_alignment_error_ms=stimulus_alignment_error_ms,
        eyelink_display_mode=EYELINK_DISPLAY_MODE,
    )


# =============================================================================
# 6. AXIS LIMITS AND PLOT ELEMENTS
# =============================================================================

def determine_common_y_limits(payloads: list[PlotPayload]) -> tuple[float, float]:
    if Y_LIMITS is not None:
        return float(Y_LIMITS[0]), float(Y_LIMITS[1])

    arrays = [
        pd.to_numeric(payload.line_data["gaze_deg_x"], errors="coerce").to_numpy(float)
        for payload in payloads
    ]
    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite gaze values found.")

    lower, upper = np.quantile(values, (0.01, 0.99))
    symmetric = max(abs(float(lower)), abs(float(upper)))
    symmetric = max(5.0, math.ceil(symmetric / 5.0) * 5.0)
    return -symmetric, symmetric


def add_phase_backgrounds(ax: Axes, payload: PlotPayload) -> None:
    ax.axvspan(
        payload.fixation_start_s,
        payload.fixation_end_s,
        facecolor="#F2F2F2",
        edgecolor="none",
        zorder=0,
    )

    if SHOW_PHASE_LABELS:
        style = {
            "ha": "center",
            "va": "bottom",
            "fontsize": 7.2,
            "color": "#666666",
            "fontweight": "bold",
            "transform": ax.get_xaxis_transform(),
            "zorder": 7,
            "clip_on": True,
        }
        ax.text(
            (payload.fixation_start_s + payload.fixation_end_s) / 2.0,
            0.012,
            "Fixation",
            **style,
        )
        ax.text(
            (payload.stimulus_start_s + payload.stimulus_end_s) / 2.0,
            0.012,
            "Stimulus",
            **style,
        )


def add_gaze_lines(ax: Axes, line_data: pd.DataFrame) -> None:
    for method in METHOD_ORDER:
        selected = line_data.loc[line_data["method"].eq(method)].sort_values("time_s")
        if selected.empty:
            continue
        style = METHOD_STYLES[method]
        ax.plot(
            selected["time_s"],
            selected["gaze_deg_x"],
            color=style["color"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            label=method,
            zorder=3,
        )


def add_region_label(
    ax: Axes,
    start_s: float,
    end_s: float,
    short_label: str,
    edgecolor: str,
) -> None:
    midpoint = (start_s + end_s) / 2.0
    ax.text(
        midpoint,
        0.975,
        short_label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color=edgecolor,
        bbox={
            "boxstyle": "round,pad=0.14",
            "facecolor": "white",
            "edgecolor": edgecolor,
            "linewidth": 0.6,
            "alpha": 0.86,
        },
        clip_on=True,
        zorder=8,
    )


def add_own_blink_regions(ax: Axes, data: pd.DataFrame) -> None:
    for region in data.itertuples(index=False):
        group = str(region.reason_group)
        style = OWN_BLINK_STYLES.get(group, OWN_BLINK_STYLES["invalid"])
        start_s = float(region.start_s)
        end_s = float(region.end_s)
        ax.axvspan(
            start_s,
            end_s,
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            alpha=style["alpha"],
            linestyle=style["linestyle"],
            linewidth=1.05,
            hatch=style["hatch"],
            zorder=1,
        )
        if SHOW_OWN_REGION_LABELS:
            add_region_label(
                ax,
                start_s,
                end_s,
                style["short_label"],
                style["edgecolor"],
            )


def add_eyelink_blink_regions(ax: Axes, data: pd.DataFrame) -> None:
    style = EYELINK_BLINK_STYLE
    for region in data.itertuples(index=False):
        start_s = float(region.start_s)
        end_s = float(region.end_s)
        ax.axvspan(
            start_s,
            end_s,
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            alpha=style["alpha"],
            linestyle=style["linestyle"],
            linewidth=1.05,
            hatch=style["hatch"],
            zorder=1,
        )
        if SHOW_EYELINK_REGION_LABELS:
            add_region_label(
                ax,
                start_s,
                end_s,
                style["short_label"],
                style["edgecolor"],
            )


def format_panel(
    ax: Axes,
    payload: PlotPayload,
    panel_type: str,
    y_limits: tuple[float, float],
) -> None:
    add_phase_backgrounds(ax, payload)
    if panel_type == "A":
        add_own_blink_regions(ax, payload.own_blink_data)
        subtitle = "A — Webcam blink detection"
    elif panel_type == "B":
        add_eyelink_blink_regions(ax, payload.eyelink_blink_data)
        subtitle = "B — EyeLink ASC EBLINK benchmark"
    else:
        raise ValueError(f"Unknown panel type: {panel_type}")

    add_gaze_lines(ax, payload.line_data)
    ax.axvline(
        payload.stimulus_start_s,
        color="#595959",
        linewidth=0.95,
        linestyle="-.",
        zorder=2,
    )

    ax.set_xlim(payload.x_limits)
    ax.set_ylim(y_limits)
    ax.set_title(
        f"{payload.vp_code} — Trial {payload.trial_number}\n{subtitle}",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.text(
        1.0,
        1.01,
        payload.run_name,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="#666666",
    )
    ax.set_xlabel("Time from fixation start (s)")
    ax.set_ylabel("Horizontal gaze angle (°)")
    ax.grid(True, which="major", color="#E0E0E0", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def gaze_legend_handles() -> list[Line2D]:
    handles = [
        Line2D(
            [0], [0],
            color=METHOD_STYLES[method]["color"],
            linewidth=1.6,
            alpha=METHOD_STYLES[method]["alpha"],
            label=method,
        )
        for method in METHOD_ORDER
    ]
    handles.append(
        Line2D(
            [0], [0],
            color="#595959",
            linewidth=0.95,
            linestyle="-.",
            label="Stimulus onset",
        )
    )
    return handles


def region_legend_handles() -> list[Patch]:
    handles: list[Patch] = []
    for group in ("blink", "suspected", "invalid"):
        style = OWN_BLINK_STYLES[group]
        handles.append(
            Patch(
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                alpha=style["alpha"],
                linestyle=style["linestyle"],
                hatch=style["hatch"],
                label=style["label"],
            )
        )
    style = EYELINK_BLINK_STYLE
    handles.append(
        Patch(
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            alpha=style["alpha"],
            linestyle=style["linestyle"],
            label=style["label"],
        )
    )
    return handles


def add_common_legends(figure: plt.Figure, bottom_y: float = 0.005) -> None:
    gaze_legend = figure.legend(
        handles=gaze_legend_handles(),
        title="Gaze lines",
        loc="lower center",
        bbox_to_anchor=(0.5, bottom_y + 0.042),
        ncol=4,
        frameon=False,
    )
    figure.add_artist(gaze_legend)
    figure.legend(
        handles=region_legend_handles(),
        title="Blink / invalid intervals",
        loc="lower center",
        bbox_to_anchor=(0.5, bottom_y),
        ncol=4,
        frameon=False,
    )


# =============================================================================
# 7. SAVE ONE MULTI-PAGE PDF
# =============================================================================

def output_dir_for_trial() -> Path:
    return OUTPUT_ROOT / f"trial_{TRIAL_NUMBER:02d}"


def chunk_payloads(
    items: list[PlotPayload],
    chunk_size: int,
) -> Iterable[list[PlotPayload]]:
    for index in range(0, len(items), chunk_size):
        yield items[index:index + chunk_size]


def output_pdf_path() -> Path:
    return (
        output_dir_for_trial()
        / f"trial_{TRIAL_NUMBER:02d}_blink_comparison_all_vps.pdf"
    )


def save_single_multipage_pdf(
    payloads: list[PlotPayload],
    y_limits: tuple[float, float],
) -> Path:
    """
    Save all VP A/B panels into one multi-page PDF.

    One PDF page contains up to six VPs, arranged as:
        rows 1-2: VP 1-3  -> A above B
        rows 3-4: VP 4-6  -> A above B

    Concretely:
        row 1: A panels for VPs 1, 2, 3
        row 2: B panels for VPs 1, 2, 3
        row 3: A panels for VPs 4, 5, 6
        row 4: B panels for VPs 4, 5, 6

    The output path is fixed for a trial, so a rerun replaces the previous
    PDF instead of accumulating stale page and individual files.
    """
    output_dir = output_dir_for_trial()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_pdf_path()

    with PdfPages(
        pdf_path,
        metadata={
            "Title": f"Blink comparison — Trial {TRIAL_NUMBER}",
            "Subject": (
                "MediaPipe, PTGaze, and EyeLink gaze; webcam blink "
                "detection versus EyeLink ASC EBLINK events"
            ),
            "Creator": "plot_blink_visualization.py",
        },
    ) as pdf:
        for page_number, page_payloads in enumerate(
            chunk_payloads(payloads, VPS_PER_PAGE),
            start=1,
        ):
            figure, axes = plt.subplots(
                nrows=4,
                ncols=VP_COLUMNS_PER_PAGE,
                figsize=PAGE_FIGSIZE,
                squeeze=False,
            )

            # Hide everything first; only activate the slots we actually use.
            for ax in axes.flat:
                ax.set_visible(False)

            for index, payload in enumerate(page_payloads):
                block_row = index // VP_COLUMNS_PER_PAGE   # 0 or 1
                column = index % VP_COLUMNS_PER_PAGE       # 0, 1, 2

                ax_a = axes[2 * block_row, column]
                ax_b = axes[2 * block_row + 1, column]

                ax_a.set_visible(True)
                ax_b.set_visible(True)

                format_panel(ax_a, payload, "A", y_limits)
                format_panel(ax_b, payload, "B", y_limits)

            vp_part = " / ".join(payload.vp_code for payload in page_payloads)
            figure.suptitle(
                (
                    f"Blink comparison — Trial {TRIAL_NUMBER}"
                    f" — {vp_part}"
                    f" — page {page_number}"
                ),
                fontsize=14,
                fontweight="bold",
                y=0.985,
            )
            add_common_legends(figure)
            figure.subplots_adjust(
                left=0.055,
                right=0.99,
                top=0.93,
                bottom=0.11,
                hspace=0.35,
                wspace=0.18,
            )

            pdf.savefig(figure, facecolor="white")
            plt.close(figure)

    return pdf_path



# =============================================================================
# 8. OUTPUT RECORDS
# =============================================================================

def save_selected_inputs(inputs: list[InputPaths]) -> None:
    """Record the exact input files selected for every successfully loaded VP."""
    rows: list[dict[str, Any]] = []

    for item in inputs:
        rows.append(
            {
                "vp_code": item.vp_code,
                "trial_number": TRIAL_NUMBER,
                "run_name": item.run_name,
                "run_dir": str(item.run_dir),
                "eyelink_gaze_file": str(item.eyelink_gaze_file),
                "mediapipe_file": str(item.mediapipe_file),
                "ptgaze_file": str(item.ptgaze_file),
                "phases_file": str(item.phases_file),
                "own_blink_file": str(item.own_blink_file),
                "eyelink_blink_file": str(item.eyelink_blink_file),
                "eyelink_markers_file": str(item.eyelink_markers_file),
            }
        )

    pd.DataFrame(rows).to_csv(
        output_dir_for_trial() / "selected_input_files.csv",
        index=False,
    )


def save_processing_summary(payloads: list[PlotPayload]) -> None:
    """Save basic counts and timing information for the generated PDF."""
    rows: list[dict[str, Any]] = []

    for payload in payloads:
        own_counts = (
            payload.own_blink_data["reason_group"]
            .value_counts()
            .to_dict()
        )

        rows.append(
            {
                "vp_code": payload.vp_code,
                "trial_number": payload.trial_number,
                "run_name": payload.run_name,
                "camera_fps": payload.camera_fps,
                "own_blink_fps": payload.own_blink_fps,
                "n_mediapipe_points": int(
                    payload.line_data["method"].eq("MediaPipe").sum()
                ),
                "n_ptgaze_points": int(
                    payload.line_data["method"].eq("PTGaze").sum()
                ),
                "n_eyelink_points_plotted": int(
                    payload.line_data["method"].eq("EyeLink").sum()
                ),
                "n_own_blink_regions": int(
                    own_counts.get("blink", 0)
                ),
                "n_own_suspected_regions": int(
                    own_counts.get("suspected", 0)
                ),
                "n_own_other_invalid_regions": int(
                    own_counts.get("invalid", 0)
                ),
                "n_eyelink_benchmark_regions": int(
                    len(payload.eyelink_blink_data)
                ),
                "fixation_end_s": payload.fixation_end_s,
                "stimulus_onset_s": payload.stimulus_start_s,
                "stimulus_end_s": payload.stimulus_end_s,
                "x_axis_end_s": payload.x_limits[1],
                "sync_offset_ms": payload.sync_offset_ms,
                "sync_offset_spread_ms":
                    payload.sync_offset_spread_ms,
                "fixation_alignment_error_ms":
                    payload.fixation_alignment_error_ms,
                "stimulus_alignment_error_ms":
                    payload.stimulus_alignment_error_ms,
                "eyelink_display_mode":
                    payload.eyelink_display_mode,
            }
        )

    pd.DataFrame(rows).to_csv(
        output_dir_for_trial() / "plot_processing_summary.csv",
        index=False,
    )


# =============================================================================
# 9. MAIN
# =============================================================================

def main() -> int:
    try:
        validate_settings()
        output_dir_for_trial().mkdir(parents=True, exist_ok=True)

        inputs: list[InputPaths] = []
        payloads: list[PlotPayload] = []
        failures: list[tuple[str, str]] = []

        for vp_code in VP_CODES:
            try:
                item = discover_inputs(vp_code)
                payload = prepare_one_vp(item)
                inputs.append(item)
                payloads.append(payload)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                failures.append((vp_code, message))
                print(f"\n[ERROR] {vp_code}: {message}", file=sys.stderr)
                if not CONTINUE_ON_ERROR:
                    raise

        if not payloads:
            raise RuntimeError("No VP could be prepared successfully.")

        y_limits = determine_common_y_limits(payloads)
        print(
            f"\n[INFO] Common y limits: {y_limits[0]:.1f} to "
            f"{y_limits[1]:.1f} degrees"
        )

        save_selected_inputs(inputs)
        pdf_path = save_single_multipage_pdf(payloads, y_limits)
        save_processing_summary(payloads)

        if failures:
            pd.DataFrame(failures, columns=["vp_code", "error"]).to_csv(
                output_dir_for_trial() / "plot_failures.csv",
                index=False,
            )

        print("\n=== Finished ===")
        print(f"Trial:  {TRIAL_NUMBER}")
        print(f"Output directory: {output_dir_for_trial()}")
        print(f"Output PDF:       {pdf_path}")
        print(f"VPs plotted: {len(payloads)}")
        print(f"VPs failed:  {len(failures)}")
        return 0

    except Exception as exc:
        print(f"\n[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())