#!/usr/bin/env python3
"""
Figure 7: MediaPipe + PTGaze + new blink-filter regions.

This is the Python equivalent of:
    03_figure7_mediapipe_ptgaze_blink.R

It:
- loops through all configured VP folders;
- finds the newest matching Run_... directory;
- reads calibrated MediaPipe and PTGaze CSV files;
- reads trial boundaries from phases_detected.json;
- reads blink regions from Ergebnisse/<VP>/test/blink_filter_regions.csv;
- plots one selected trial per VP;
- marks blink / suspected / invalid regions with different styles;
- saves individual PNG/PDF files and one combined multi-panel figure.

Run:
    /opt/homebrew/bin/python3.10 \
        Skripte/debug/plot_figure7_mediapipe_ptgaze_blink.py

Required packages:
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
except ModuleNotFoundError as exc:
    missing_package = exc.name or "unknown package"
    raise SystemExit(
        "\nMissing Python package: "
        f"{missing_package}\n\n"
        "Install the plotting dependencies with:\n"
        "  /opt/homebrew/bin/python3.10 -m pip install "
        "pandas numpy matplotlib\n"
    ) from exc


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

# Only this project results directory needs to be an absolute path.
ROOT_DIR = Path("/Users/yiyi_mac/IDP_Code/Ergebnisse")

# Output folder for the new figures.
OUTPUT_DIR = ROOT_DIR / "Figure7_new_blink_filter"

# One selected trial per VP.
#
# run_regex is matched against the Run folder name, not the full path.
# "^Run_25hz_F" matches folders whose names begin with Run_25hz_F.
VP_CONFIG = [
    {"vp_code": "beo7", "trial_number": 1, "run_regex": r"^Run_25hz_F"},
    {"vp_code": "bjs4", "trial_number": 1, "run_regex": r"^Run_25hz_F"},
    {"vp_code": "egf5", "trial_number": 1, "run_regex": r"^Run_25hz_F"},
    {"vp_code": "fbn6", "trial_number": 1, "run_regex": r"^Run_25hz_F"},
    {"vp_code": "fgt6", "trial_number": 1, "run_regex": r"^Run_25hz_F"},
    {"vp_code": "jkl7", "trial_number": 1, "run_regex": r"^Run_25hz_F"},
    {"vp_code": "kdn8", "trial_number": 1, "run_regex": r"^Run_25hz_F"},
    # {"vp_code": "kly9", "trial_number": 1, "run_regex": r"^Run_25hz_F"},
    {"vp_code": "kro3", "trial_number": 1, "run_regex": r"^Run_25hz_F"},
    # {"vp_code": "ldj9", "trial_number": 1, "run_regex": r"^Run_25hz_F"},
    {"vp_code": "mhe9", "trial_number": 1, "run_regex": r"^Run_25hz_F"},
    {"vp_code": "oem4", "trial_number": 1, "run_regex": r"^Run_25hz_F"},
    {"vp_code": "ogt7", "trial_number": 1, "run_regex": r"^Run_25hz_F"},
]

# Trial-relative x-axis zero:
#   "fixation": fixation begins at 0 s.
#   "stimulus": stimulus begins at 0 s; fixation appears at negative time.
TIME_ZERO_PHASE = "fixation"

# True:
#   MediaPipe and PTGaze lines are removed inside every invalid region.
# False:
#   Lines remain visible; only the background regions are drawn.
MASK_LINES_IN_INVALID_RANGES = False

# Add one-letter labels above each invalid range:
#   B = Blink / eyes closed
#   S = Suspected blink / transition
#   I = Other invalid frame range
SHOW_REGION_LABELS = True

# Label the two experimental phases directly inside each panel.
SHOW_PHASE_LABELS = True

# None:
#   calculate one robust common y range for all VP panels.
# Or use a fixed tuple, for example:
#   Y_LIMITS = (-25.0, 25.0)
Y_LIMITS: Optional[tuple[float, float]] = None

# Combined figure layout.
COMBINED_NCOL = 2

# Output sizes.
INDIVIDUAL_WIDTH_IN = 9.0
INDIVIDUAL_HEIGHT_IN = 4.5
COMBINED_WIDTH_IN = 14.0
COMBINED_HEIGHT_PER_ROW_IN = 4.0
OUTPUT_DPI = 300

# Exact input file names inside each Run_... directory.
MEDIAPIPE_FILE = "debug_5_pupil_data_calibrated.csv"
PTGAZE_FILE = "debug_5_ptgaze_calibrated.csv"
PHASES_FILE = "phases_detected.json"
BLINK_FILE = "blink_filter_regions.csv"

REQUIRED_RUN_FILES = (
    MEDIAPIPE_FILE,
    PTGAZE_FILE,
    PHASES_FILE,
)

METHOD_STYLES = {
    "MediaPipe": {
        "color": "#D73027",
        "linewidth": 0.85,
    },
    "PTGaze": {
        "color": "#4575B4",
        "linewidth": 0.85,
    },
}

# Different visual styles for the three blink-filter groups.
BLINK_STYLES = {
    "blink": {
        "facecolor": "#D73027",
        "edgecolor": "#8B1A14",
        "alpha": 0.22,
        "linestyle": "-",
        "hatch": None,
        "short_label": "B",
        "label": "B — Blink / eyes closed",
    },
    "suspected": {
        "facecolor": "#F0A202",
        "edgecolor": "#9A6500",
        "alpha": 0.20,
        "linestyle": "--",
        "hatch": "////",
        "short_label": "S",
        "label": "S — Suspected / transition",
    },
    "invalid": {
        "facecolor": "#7B4AB5",
        "edgecolor": "#4D2B78",
        "alpha": 0.17,
        "linestyle": ":",
        "hatch": "xxxx",
        "short_label": "I",
        "label": "I — Other invalid",
    },
}


# =============================================================================
# 2. DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class InputPaths:
    vp_code: str
    trial_number: int
    run_regex: str
    vp_dir: Path
    run_dir: Path
    run_name: str
    mediapipe_file: Path
    ptgaze_file: Path
    phases_file: Path
    blink_file: Path


@dataclass
class PlotPayload:
    vp_code: str
    trial_number: int
    run_name: str
    line_data: pd.DataFrame
    blink_data: pd.DataFrame
    fixation_start_s: float
    fixation_end_s: float
    stimulus_start_s: float
    stimulus_end_s: float
    x_limits: tuple[float, float]
    calibrated_fps: float
    blink_fps: float


# =============================================================================
# 3. VALIDATION
# =============================================================================

def validate_settings() -> None:
    if not ROOT_DIR.is_dir():
        raise FileNotFoundError(f"ROOT_DIR does not exist: {ROOT_DIR}")

    if TIME_ZERO_PHASE not in {"fixation", "stimulus"}:
        raise ValueError(
            'TIME_ZERO_PHASE must be either "fixation" or "stimulus".'
        )

    if not VP_CONFIG:
        raise ValueError("VP_CONFIG is empty.")

    vp_codes = [item["vp_code"] for item in VP_CONFIG]

    if len(vp_codes) != len(set(vp_codes)):
        raise ValueError(
            "VP_CONFIG contains duplicate vp_code values. "
            "Use one selected trial per VP."
        )

    for item in VP_CONFIG:
        required_keys = {"vp_code", "trial_number", "run_regex"}
        missing_keys = required_keys.difference(item)

        if missing_keys:
            raise KeyError(
                f"VP_CONFIG entry is missing: {sorted(missing_keys)}"
            )

        if int(item["trial_number"]) < 1:
            raise ValueError(
                f"Invalid trial number for {item['vp_code']}: "
                f"{item['trial_number']}"
            )

        # Validate the regular expression before scanning directories.
        re.compile(str(item["run_regex"]))

    if COMBINED_NCOL <= 0:
        raise ValueError("COMBINED_NCOL must be positive.")

    if Y_LIMITS is not None:
        if (
            len(Y_LIMITS) != 2
            or not all(math.isfinite(float(value)) for value in Y_LIMITS)
            or float(Y_LIMITS[0]) >= float(Y_LIMITS[1])
        ):
            raise ValueError(
                "Y_LIMITS must be None or a tuple (minimum, maximum)."
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 4. FILE DISCOVERY
# =============================================================================

def iter_directories(root: Path) -> Iterable[Path]:
    yield root

    for candidate in root.rglob("*"):
        if candidate.is_dir():
            yield candidate


def find_latest_matching_run(vp_dir: Path, run_regex: str) -> Path:
    pattern = re.compile(run_regex)

    candidates = [
        directory
        for directory in iter_directories(vp_dir)
        if pattern.search(directory.name)
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No Run directory matching {run_regex!r} found under:\n"
            f"  {vp_dir}"
        )

    candidates_with_files = [
        directory
        for directory in candidates
        if all((directory / filename).is_file()
               for filename in REQUIRED_RUN_FILES)
    ]

    if not candidates_with_files:
        required_text = "\n  ".join(REQUIRED_RUN_FILES)

        raise FileNotFoundError(
            f"Run directories matching {run_regex!r} were found under "
            f"{vp_dir}, but none contains all required files:\n"
            f"  {required_text}"
        )

    selected = max(
        candidates_with_files,
        key=lambda directory: directory.stat().st_mtime,
    )

    if len(candidates_with_files) > 1:
        print(
            f"[INFO] Multiple matching runs for {vp_dir.name}; "
            f"using newest:\n       {selected}"
        )

    return selected


def find_blink_file(vp_dir: Path, run_dir: Path) -> Path:
    # Primary/default location requested by the project:
    # Ergebnisse/<VP>/test/blink_filter_regions.csv
    default_blink_file = vp_dir / "test" / BLINK_FILE

    if default_blink_file.is_file():
        return default_blink_file

    # Backup locations retained for compatibility.
    alternatives = (
        run_dir / BLINK_FILE,
        run_dir / "blink_filter" / BLINK_FILE,
        run_dir / "post_processing" / BLINK_FILE,
    )

    for candidate in alternatives:
        if candidate.is_file():
            print(
                "[WARN] Default blink file not found; using:\n"
                f"       {candidate}"
            )
            return candidate

    raise FileNotFoundError(
        f"Could not find blink region file for VP folder:\n"
        f"  {vp_dir}\n"
        f"Expected default location:\n"
        f"  {default_blink_file}"
    )


def discover_vp_inputs(config: dict[str, Any]) -> InputPaths:
    vp_code = str(config["vp_code"])
    trial_number = int(config["trial_number"])
    run_regex = str(config["run_regex"])

    vp_dir = ROOT_DIR / vp_code

    if not vp_dir.is_dir():
        raise FileNotFoundError(f"VP directory does not exist: {vp_dir}")

    run_dir = find_latest_matching_run(vp_dir, run_regex)
    blink_file = find_blink_file(vp_dir, run_dir)

    return InputPaths(
        vp_code=vp_code,
        trial_number=trial_number,
        run_regex=run_regex,
        vp_dir=vp_dir,
        run_dir=run_dir,
        run_name=run_dir.name,
        mediapipe_file=run_dir / MEDIAPIPE_FILE,
        ptgaze_file=run_dir / PTGAZE_FILE,
        phases_file=run_dir / PHASES_FILE,
        blink_file=blink_file,
    )


def save_selected_inputs(inputs: list[InputPaths]) -> None:
    rows = [
        {
            "vp_code": item.vp_code,
            "trial_number": item.trial_number,
            "run_regex": item.run_regex,
            "vp_dir": str(item.vp_dir),
            "run_dir": str(item.run_dir),
            "run_name": item.run_name,
            "mediapipe_file": str(item.mediapipe_file),
            "ptgaze_file": str(item.ptgaze_file),
            "phases_file": str(item.phases_file),
            "blink_file": str(item.blink_file),
        }
        for item in inputs
    ]

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "selected_input_files.csv",
        index=False,
    )


# =============================================================================
# 5. READ AND NORMALIZE DATA
# =============================================================================

def assert_columns(
    data: pd.DataFrame,
    required_columns: Iterable[str],
    file_path: Path,
) -> None:
    missing = set(required_columns).difference(data.columns)

    if missing:
        raise KeyError(
            f"File is missing required columns:\n"
            f"  {file_path}\n"
            f"Missing:\n"
            f"  " + "\n  ".join(sorted(missing))
        )


def read_calibrated_method(
    file_path: Path,
    method_name: str,
    target_trial: int,
) -> pd.DataFrame:
    data = pd.read_csv(file_path)

    required = (
        "frame",
        "timestamp_ms",
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

    return normalized.loc[
        normalized["trial_number"].eq(target_trial)
    ].copy()


def read_trial_boundaries(phases_file: Path) -> pd.DataFrame:
    with phases_file.open("r", encoding="utf-8") as handle:
        phase_json = json.load(handle)

    phases = phase_json.get("phases", {})

    block1 = phases.get("experiment_block1", {}).get("trials", [])
    block2 = phases.get("experiment_block2", {}).get("trials", [])

    all_trials = [*block1, *block2]

    if not all_trials:
        raise ValueError(
            f"No experiment trials found in: {phases_file}"
        )

    rows: list[dict[str, Any]] = []

    for trial in all_trials:
        fixation = trial.get("fixation", {})
        stimulus = trial.get("stimulus", {})

        rows.append(
            {
                "trial_number": int(trial["trial_number"]),
                "is_practice": bool(trial.get("is_practice", False)),
                "fixation_start_video_ms":
                    float(fixation["start_video_s"]) * 1000.0,
                "fixation_end_video_ms":
                    float(fixation["end_video_s"]) * 1000.0,
                "stimulus_start_video_ms":
                    float(stimulus["start_video_s"]) * 1000.0,
                "stimulus_end_video_ms":
                    float(stimulus["end_video_s"]) * 1000.0,
            }
        )

    return pd.DataFrame(rows)


def read_blink_regions(blink_file: Path) -> pd.DataFrame:
    data = pd.read_csv(blink_file)

    required = (
        "start_video_time_ms",
        "end_video_time_ms",
        "reason_group",
    )
    assert_columns(data, required, blink_file)

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
                else pd.Series(
                    pd.NA,
                    index=data.index,
                    dtype="string",
                )
            ),
            "video_fps": (
                pd.to_numeric(data["video_fps"], errors="coerce")
                if "video_fps" in data.columns
                else pd.Series(
                    np.nan,
                    index=data.index,
                    dtype="float64",
                )
            ),
        }
    )

    normalized["reason_group"] = normalized[
        "reason_group"
    ].map(
        lambda group: (
            group
            if group in {"blink", "suspected"}
            else "invalid"
        )
    )

    normalized = normalized.dropna(
        subset=("start_video_time_ms", "end_video_time_ms")
    )

    return normalized


def estimate_data_fps(timestamp_ms: pd.Series) -> float:
    unique_times = np.sort(
        pd.to_numeric(timestamp_ms, errors="coerce")
        .dropna()
        .unique()
    )

    differences = np.diff(unique_times)
    differences = differences[
        np.isfinite(differences) & (differences > 0)
    ]

    if differences.size == 0:
        return float("nan")

    return float(1000.0 / np.median(differences))


def validate_fps_compatibility(
    line_data: pd.DataFrame,
    blink_data: pd.DataFrame,
    vp_code: str,
) -> tuple[float, float]:
    data_fps = estimate_data_fps(line_data["timestamp_ms"])

    finite_blink_fps = (
        pd.to_numeric(blink_data["video_fps"], errors="coerce")
        .dropna()
        .loc[lambda series: np.isfinite(series)]
        .unique()
    )

    blink_fps = (
        float(finite_blink_fps[0])
        if len(finite_blink_fps) > 0
        else float("nan")
    )

    if (
        math.isfinite(data_fps)
        and math.isfinite(blink_fps)
        and abs(data_fps - blink_fps) > 0.5
    ):
        raise ValueError(
            f"FPS mismatch for {vp_code}:\n"
            f"  calibrated data: approximately {data_fps:.3f} Hz\n"
            f"  blink regions: {blink_fps:.3f} Hz\n"
            "Use blink output from the same video/run."
        )

    return data_fps, blink_fps


# =============================================================================
# 6. PREPARE ONE VP/TRIAL
# =============================================================================

def apply_invalid_mask(
    line_data: pd.DataFrame,
    blink_trial: pd.DataFrame,
) -> pd.DataFrame:
    masked = line_data.copy()

    if blink_trial.empty:
        return masked

    invalid_mask = np.zeros(len(masked), dtype=bool)
    timestamps = masked["timestamp_ms"].to_numpy(dtype=float)

    for region in blink_trial.itertuples(index=False):
        invalid_mask |= (
            (timestamps >= float(region.start_video_time_ms))
            & (timestamps < float(region.end_video_time_ms))
        )

    masked.loc[invalid_mask, "gaze_deg_x"] = np.nan
    return masked


def prepare_one_vp(input_paths: InputPaths) -> PlotPayload:
    print(
        f"\n[LOAD] {input_paths.vp_code}"
        f" | Trial {input_paths.trial_number}"
        f" | {input_paths.run_name}"
    )

    mediapipe = read_calibrated_method(
        input_paths.mediapipe_file,
        method_name="MediaPipe",
        target_trial=input_paths.trial_number,
    )
    ptgaze = read_calibrated_method(
        input_paths.ptgaze_file,
        method_name="PTGaze",
        target_trial=input_paths.trial_number,
    )

    if mediapipe.empty:
        raise ValueError(
            f"No MediaPipe rows found for "
            f"{input_paths.vp_code} Trial "
            f"{input_paths.trial_number}"
        )

    if ptgaze.empty:
        raise ValueError(
            f"No PTGaze rows found for "
            f"{input_paths.vp_code} Trial "
            f"{input_paths.trial_number}"
        )

    boundaries = read_trial_boundaries(input_paths.phases_file)

    trial_boundary = boundaries.loc[
        boundaries["trial_number"].eq(input_paths.trial_number)
        & ~boundaries["is_practice"]
    ].copy()

    if len(trial_boundary) != 1:
        raise ValueError(
            f"Expected exactly one non-practice boundary for "
            f"{input_paths.vp_code} Trial "
            f"{input_paths.trial_number}; found "
            f"{len(trial_boundary)}"
        )

    boundary = trial_boundary.iloc[0]

    fixation_start = float(boundary["fixation_start_video_ms"])
    fixation_end = float(boundary["fixation_end_video_ms"])
    stimulus_start = float(boundary["stimulus_start_video_ms"])
    stimulus_end = float(boundary["stimulus_end_video_ms"])

    window_start = fixation_start
    window_end = stimulus_end

    zero_time = (
        fixation_start
        if TIME_ZERO_PHASE == "fixation"
        else stimulus_start
    )

    line_data = pd.concat(
        (mediapipe, ptgaze),
        ignore_index=True,
    )

    line_data = line_data.loc[
        line_data["timestamp_ms"].ge(window_start)
        & line_data["timestamp_ms"].le(window_end)
    ].copy()

    line_data["time_s"] = (
        line_data["timestamp_ms"] - zero_time
    ) / 1000.0

    line_data = line_data.sort_values(
        ["method", "timestamp_ms"],
        kind="stable",
    ).reset_index(drop=True)

    blink_data_all = read_blink_regions(
        input_paths.blink_file
    )

    calibrated_fps, blink_fps = validate_fps_compatibility(
        line_data=line_data,
        blink_data=blink_data_all,
        vp_code=input_paths.vp_code,
    )

    # Select regions overlapping this trial and clip them to the
    # exact fixation-start to stimulus-end window.
    blink_trial = blink_data_all.loc[
        blink_data_all["end_video_time_ms"].gt(window_start)
        & blink_data_all["start_video_time_ms"].lt(window_end)
    ].copy()

    if not blink_trial.empty:
        blink_trial["start_video_time_ms"] = np.maximum(
            blink_trial["start_video_time_ms"].to_numpy(dtype=float),
            window_start,
        )
        blink_trial["end_video_time_ms"] = np.minimum(
            blink_trial["end_video_time_ms"].to_numpy(dtype=float),
            window_end,
        )

        blink_trial["start_s"] = (
            blink_trial["start_video_time_ms"] - zero_time
        ) / 1000.0
        blink_trial["end_s"] = (
            blink_trial["end_video_time_ms"] - zero_time
        ) / 1000.0

        blink_trial = blink_trial.loc[
            blink_trial["end_s"].gt(blink_trial["start_s"])
        ].copy()
    else:
        blink_trial["start_s"] = pd.Series(dtype="float64")
        blink_trial["end_s"] = pd.Series(dtype="float64")

    if MASK_LINES_IN_INVALID_RANGES:
        line_data = apply_invalid_mask(
            line_data=line_data,
            blink_trial=blink_trial,
        )

    return PlotPayload(
        vp_code=input_paths.vp_code,
        trial_number=input_paths.trial_number,
        run_name=input_paths.run_name,
        line_data=line_data,
        blink_data=blink_trial,
        fixation_start_s=(fixation_start - zero_time) / 1000.0,
        fixation_end_s=(fixation_end - zero_time) / 1000.0,
        stimulus_start_s=(stimulus_start - zero_time) / 1000.0,
        stimulus_end_s=(stimulus_end - zero_time) / 1000.0,
        x_limits=(
            (window_start - zero_time) / 1000.0,
            (window_end - zero_time) / 1000.0,
        ),
        calibrated_fps=calibrated_fps,
        blink_fps=blink_fps,
    )


# =============================================================================
# 7. COMMON Y LIMIT
# =============================================================================

def determine_common_y_limits(
    payloads: list[PlotPayload],
) -> tuple[float, float]:
    if Y_LIMITS is not None:
        return float(Y_LIMITS[0]), float(Y_LIMITS[1])

    arrays = [
        pd.to_numeric(
            payload.line_data["gaze_deg_x"],
            errors="coerce",
        ).to_numpy(dtype=float)
        for payload in payloads
    ]

    all_y = np.concatenate(arrays)
    all_y = all_y[np.isfinite(all_y)]

    if all_y.size == 0:
        raise ValueError("No finite gaze values found.")

    lower, upper = np.quantile(all_y, (0.01, 0.99))
    symmetric_max = max(abs(float(lower)), abs(float(upper)))
    symmetric_max = max(5.0, math.ceil(symmetric_max / 5.0) * 5.0)

    return -symmetric_max, symmetric_max


# =============================================================================
# 8. PLOTTING
# =============================================================================

def add_phase_backgrounds(
    ax: Axes,
    payload: PlotPayload,
) -> None:
    # Very light fixation background. The stimulus phase remains white so that
    # blink-filter regions stay visually dominant.
    ax.axvspan(
        payload.fixation_start_s,
        payload.fixation_end_s,
        facecolor="#F5F5F5",
        edgecolor="none",
        zorder=0,
    )

    if SHOW_PHASE_LABELS:
        fixation_mid = (
            payload.fixation_start_s + payload.fixation_end_s
        ) / 2.0
        stimulus_mid = (
            payload.stimulus_start_s + payload.stimulus_end_s
        ) / 2.0

        phase_label_style = {
            "ha": "center",
            "va": "bottom",
            "fontsize": 7.5,
            "color": "#666666",
            "fontweight": "bold",
            "transform": ax.get_xaxis_transform(),
            "zorder": 6,
            "clip_on": True,
        }

        ax.text(
            fixation_mid,
            0.015,
            "Fixation",
            **phase_label_style,
        )
        ax.text(
            stimulus_mid,
            0.015,
            "Stimulus",
            **phase_label_style,
        )


def add_blink_regions(
    ax: Axes,
    blink_data: pd.DataFrame,
) -> None:
    for region in blink_data.itertuples(index=False):
        group = str(region.reason_group)
        style = BLINK_STYLES.get(
            group,
            BLINK_STYLES["invalid"],
        )

        start_s = float(region.start_s)
        end_s = float(region.end_s)

        patch = ax.axvspan(
            start_s,
            end_s,
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            alpha=style["alpha"],
            linestyle=style["linestyle"],
            linewidth=1.15,
            hatch=style["hatch"],
            zorder=1,
        )

        patch.set_hatch(style["hatch"])

        if SHOW_REGION_LABELS:
            midpoint = (start_s + end_s) / 2.0

            ax.text(
                midpoint,
                0.975,
                style["short_label"],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.2,
                fontweight="bold",
                color=style["edgecolor"],
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": style["edgecolor"],
                    "linewidth": 0.65,
                    "alpha": 0.88,
                },
                clip_on=True,
                zorder=7,
            )


def add_gaze_lines(
    ax: Axes,
    line_data: pd.DataFrame,
) -> None:
    for method in ("MediaPipe", "PTGaze"):
        method_data = line_data.loc[
            line_data["method"].eq(method)
        ].sort_values("time_s")

        if method_data.empty:
            continue

        style = METHOD_STYLES[method]

        ax.plot(
            method_data["time_s"],
            method_data["gaze_deg_x"],
            color=style["color"],
            linewidth=style["linewidth"],
            label=method,
            zorder=3,
        )


def format_axis(
    ax: Axes,
    payload: PlotPayload,
    y_limits: tuple[float, float],
) -> None:
    add_phase_backgrounds(ax, payload)
    add_blink_regions(ax, payload.blink_data)
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
        f"{payload.vp_code} — Trial {payload.trial_number}",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        0.0,
        1.01,
        payload.run_name,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#595959",
    )

    ax.set_xlabel(
        (
            "Time from fixation start (s)"
            if TIME_ZERO_PHASE == "fixation"
            else "Time from stimulus start (s)"
        )
    )
    ax.set_ylabel("Horizontal gaze angle (°)")

    ax.grid(
        True,
        which="major",
        color="#E0E0E0",
        linewidth=0.45,
    )
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def method_legend_handles() -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLES[method]["color"],
            linewidth=1.6,
            label=method,
        )
        for method in ("MediaPipe", "PTGaze")
    ]

    handles.append(
        Line2D(
            [0],
            [0],
            color="#595959",
            linewidth=0.95,
            linestyle="-.",
            label="Stimulus onset",
        )
    )

    return handles


def blink_legend_handles() -> list[Patch]:
    handles: list[Patch] = []

    for group in ("blink", "suspected", "invalid"):
        style = BLINK_STYLES[group]

        handles.append(
            Patch(
                facecolor=style["facecolor"],
                edgecolor=style["edgecolor"],
                linewidth=0.9,
                linestyle=style["linestyle"],
                hatch=style["hatch"],
                alpha=style["alpha"],
                label=style["label"],
            )
        )

    return handles


def add_figure_legends(
    figure: plt.Figure,
    bottom_y: float,
) -> None:
    method_legend = figure.legend(
        handles=method_legend_handles(),
        title="Gaze method",
        loc="lower center",
        bbox_to_anchor=(0.5, bottom_y + 0.04),
        ncol=3,
        frameon=False,
    )

    figure.add_artist(method_legend)

    figure.legend(
        handles=blink_legend_handles(),
        title="Blink-filter group",
        loc="lower center",
        bbox_to_anchor=(0.5, bottom_y),
        ncol=3,
        frameon=False,
    )


def create_individual_figure(
    payload: PlotPayload,
    y_limits: tuple[float, float],
) -> plt.Figure:
    figure, ax = plt.subplots(
        figsize=(INDIVIDUAL_WIDTH_IN, INDIVIDUAL_HEIGHT_IN)
    )

    format_axis(ax, payload, y_limits)
    add_figure_legends(figure, bottom_y=0.005)

    figure.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.88,
        bottom=0.28,
    )

    return figure


def save_individual_figures(
    payloads: list[PlotPayload],
    y_limits: tuple[float, float],
) -> None:
    individual_dir = OUTPUT_DIR / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for payload in payloads:
        figure = create_individual_figure(payload, y_limits)

        base_name = (
            f"{payload.vp_code}"
            f"_trial_{payload.trial_number}"
            f"_mediapipe_ptgaze_blink"
        )

        figure.savefig(
            individual_dir / f"{base_name}.png",
            dpi=OUTPUT_DPI,
            facecolor="white",
        )
        figure.savefig(
            individual_dir / f"{base_name}.pdf",
            facecolor="white",
        )

        plt.close(figure)


def create_combined_figure(
    payloads: list[PlotPayload],
    y_limits: tuple[float, float],
) -> plt.Figure:
    n_plots = len(payloads)
    n_rows = math.ceil(n_plots / COMBINED_NCOL)

    figure, axes = plt.subplots(
        nrows=n_rows,
        ncols=COMBINED_NCOL,
        figsize=(
            COMBINED_WIDTH_IN,
            max(
                COMBINED_HEIGHT_PER_ROW_IN,
                COMBINED_HEIGHT_PER_ROW_IN * n_rows,
            ),
        ),
        squeeze=False,
    )

    flat_axes = axes.flatten()

    for ax, payload in zip(flat_axes, payloads):
        format_axis(ax, payload, y_limits)

    for unused_ax in flat_axes[n_plots:]:
        unused_ax.set_visible(False)

    add_figure_legends(figure, bottom_y=0.005)

    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.97,
        bottom=0.105,
        hspace=0.42,
        wspace=0.18,
    )

    return figure


def save_combined_figure(
    payloads: list[PlotPayload],
    y_limits: tuple[float, float],
) -> None:
    figure = create_combined_figure(payloads, y_limits)

    figure.savefig(
        OUTPUT_DIR / "figure7_mediapipe_ptgaze_new_blink.png",
        dpi=OUTPUT_DPI,
        facecolor="white",
    )
    figure.savefig(
        OUTPUT_DIR / "figure7_mediapipe_ptgaze_new_blink.pdf",
        facecolor="white",
    )

    plt.close(figure)


# =============================================================================
# 9. PROCESSING SUMMARY
# =============================================================================

def build_processing_summary(
    payloads: list[PlotPayload],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for payload in payloads:
        counts = (
            payload.blink_data["reason_group"]
            .value_counts()
            .to_dict()
        )

        rows.append(
            {
                "vp_code": payload.vp_code,
                "trial_number": payload.trial_number,
                "run_name": payload.run_name,
                "calibrated_fps": payload.calibrated_fps,
                "blink_fps": payload.blink_fps,
                "n_mediapipe_points": int(
                    payload.line_data["method"]
                    .eq("MediaPipe")
                    .sum()
                ),
                "n_ptgaze_points": int(
                    payload.line_data["method"]
                    .eq("PTGaze")
                    .sum()
                ),
                "n_blink_regions": int(
                    counts.get("blink", 0)
                ),
                "n_suspected_regions": int(
                    counts.get("suspected", 0)
                ),
                "n_other_invalid_regions": int(
                    counts.get("invalid", 0)
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 10. MAIN
# =============================================================================

def main() -> int:
    try:
        validate_settings()

        inputs = [
            discover_vp_inputs(config)
            for config in VP_CONFIG
        ]

        save_selected_inputs(inputs)

        print("\n=== Selected inputs ===")
        for item in inputs:
            print(
                f"{item.vp_code:5s}"
                f" | Trial {item.trial_number:2d}"
                f" | {item.run_name}"
            )
            print(f"       Blink: {item.blink_file}")

        payloads = [
            prepare_one_vp(input_paths)
            for input_paths in inputs
        ]

        common_y_limits = determine_common_y_limits(
            payloads
        )

        print(
            "\n[INFO] Common y limits: "
            f"{common_y_limits[0]:.1f} to "
            f"{common_y_limits[1]:.1f} degrees"
        )

        save_individual_figures(
            payloads,
            common_y_limits,
        )
        save_combined_figure(
            payloads,
            common_y_limits,
        )

        summary = build_processing_summary(payloads)
        summary.to_csv(
            OUTPUT_DIR / "figure7_processing_summary.csv",
            index=False,
        )

        print("\n=== Finished ===")
        print("Output directory:")
        print(f"  {OUTPUT_DIR}")
        print("\nCombined figures:")
        print(
            "  "
            + str(
                OUTPUT_DIR
                / "figure7_mediapipe_ptgaze_new_blink.png"
            )
        )
        print(
            "  "
            + str(
                OUTPUT_DIR
                / "figure7_mediapipe_ptgaze_new_blink.pdf"
            )
        )

        return 0

    except Exception as exc:
        print(
            f"\n[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
