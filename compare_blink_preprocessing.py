#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def read_regions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"start_video_time_ms", "end_video_time_ms", "reason_group"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing columns {sorted(missing)}")
    out = df.copy()
    out["start_video_time_ms"] = pd.to_numeric(out["start_video_time_ms"], errors="coerce")
    out["end_video_time_ms"] = pd.to_numeric(out["end_video_time_ms"], errors="coerce")
    return out.dropna(subset=["start_video_time_ms", "end_video_time_ms"])


def read_eyelink(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"trial", "start_from_fixation_ms", "end_from_fixation_ms"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing columns {sorted(missing)}")
    return df


def read_markers(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"trial", "fixation_start_tracker_ms"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path}: missing columns {sorted(missing)}")
    return df



def read_trial_video_window(run_dir: Path, trial: int) -> tuple[float, float]:
    path = run_dir / "phases_detected.json"
    import json
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    phases = payload.get("phases", {})
    trials = [
        *phases.get("experiment_block1", {}).get("trials", []),
        *phases.get("experiment_block2", {}).get("trials", []),
    ]
    matches = [
        item for item in trials
        if int(item.get("trial_number", -1)) == trial
        and not bool(item.get("is_practice", False))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one non-practice Trial {trial} in {path}, found {len(matches)}"
        )

    item = matches[0]
    start_ms = float(item["fixation"]["start_video_s"]) * 1000.0
    end_ms = float(item["stimulus"]["end_video_s"]) * 1000.0
    return start_ms, end_ms


def read_sync_offset(run_dir: Path, trial: int) -> float:
    path = run_dir / "debug_5_pupil_data_calibrated.csv"
    df = pd.read_csv(path, usecols=["trial_number", "timestamp_ms", "timestamp_ms_synced"])
    t = pd.to_numeric(df["trial_number"], errors="coerce")
    sel = df.loc[t.eq(trial)].copy()
    if sel.empty:
        raise ValueError(f"No Trial {trial} rows in {path}")
    offset = (
        pd.to_numeric(sel["timestamp_ms_synced"], errors="coerce")
        - pd.to_numeric(sel["timestamp_ms"], errors="coerce")
    ).dropna()
    if offset.empty:
        raise ValueError(f"No synchronization offset in {path}")
    return float(offset.median())


def trial_eyelink_intervals(blinks: pd.DataFrame, trial: int) -> list[tuple[float, float]]:
    t = pd.to_numeric(blinks["trial"], errors="coerce")
    sel = blinks.loc[t.eq(trial)].copy()
    starts = pd.to_numeric(sel["start_from_fixation_ms"], errors="coerce")
    ends = pd.to_numeric(sel["end_from_fixation_ms"], errors="coerce")
    return [(float(a), float(b)) for a, b in zip(starts, ends) if np.isfinite(a) and np.isfinite(b)]


def webcam_intervals(
    regions: pd.DataFrame,
    trial_start_video_ms: float,
    trial_end_video_ms: float,
    included_groups: set[str],
) -> list[tuple[float, float, str]]:
    sel = regions.loc[
        regions["reason_group"].astype(str).isin(included_groups)
        & regions["end_video_time_ms"].gt(trial_start_video_ms)
        & regions["start_video_time_ms"].lt(trial_end_video_ms)
    ].copy()

    out = []
    for row in sel.itertuples():
        start_video_ms = max(float(row.start_video_time_ms), trial_start_video_ms)
        end_video_ms = min(float(row.end_video_time_ms), trial_end_video_ms)
        a = start_video_ms - trial_start_video_ms
        b = end_video_ms - trial_start_video_ms
        if b > a:
            out.append((a, b, str(row.reason_group)))
    return out


def overlap(a, b) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def match_events(pred, truth, tolerance_ms: float):
    candidates = []
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            expanded = (t[0] - tolerance_ms, t[1] + tolerance_ms)
            p_interval = (p[0], p[1])
            ov = overlap(p_interval, expanded)
            if ov > 0 or (p[0] <= expanded[1] and p[1] >= expanded[0]):
                center_dist = abs((p[0] + p[1]) / 2 - (t[0] + t[1]) / 2)
                candidates.append((center_dist, -overlap(p_interval, t), i, j))
    candidates.sort()
    used_p, used_t, matches = set(), set(), []
    for _, _, i, j in candidates:
        if i in used_p or j in used_t:
            continue
        used_p.add(i); used_t.add(j)
        matches.append((i, j))
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    if np.isfinite(precision) and np.isfinite(recall):
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
    else:
        f1 = np.nan
    return matches, tp, fp, fn, precision, recall, f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-run", type=Path, required=True,
                   help="Original raw pipeline Run directory, used for synchronization.")
    p.add_argument("--raw-blink", type=Path, default=None,
                   help="Raw blink_filter_regions.csv. Defaults to <raw-run>/blink_filter_regions.csv.")
    p.add_argument("--processed-run", type=Path, required=True)
    p.add_argument("--eyelink-dir", type=Path, required=True)
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--processed-label", default="Lanczos 2x")
    p.add_argument("--tolerance-ms", type=float, default=100.0)
    p.add_argument(
        "--blink-only",
        action="store_true",
        help="Evaluate only reason_group=blink. Default evaluates blink, suspected, and invalid.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    raw_file = (
        args.raw_blink.expanduser().resolve()
        if args.raw_blink is not None
        else args.raw_run / "blink_filter_regions.csv"
    )
    proc_file = args.processed_run / "blink_filter_regions.csv"
    args.raw_run = args.raw_run.expanduser().resolve()
    args.processed_run = args.processed_run.expanduser().resolve()
    args.eyelink_dir = args.eyelink_dir.expanduser().resolve()

    eye_file = args.eyelink_dir / "eyelink_blink_benchmark.csv"
    marker_file = args.eyelink_dir / "eyelink_trial_markers.csv"
    for f in [raw_file, proc_file, eye_file, marker_file]:
        if not f.is_file():
            raise FileNotFoundError(f)

    raw = read_regions(raw_file)
    proc = read_regions(proc_file)
    eye = read_eyelink(eye_file)
    markers = read_markers(marker_file)

    mt = pd.to_numeric(markers["trial"], errors="coerce")
    marker = markers.loc[mt.eq(args.trial)]
    if len(marker) != 1:
        raise ValueError(f"Expected one marker row for Trial {args.trial}, found {len(marker)}")
    fixation_tracker_ms = float(marker.iloc[0]["fixation_start_tracker_ms"])

    raw_trial_start, raw_trial_end = read_trial_video_window(args.raw_run, args.trial)
    proc_trial_start, proc_trial_end = read_trial_video_window(args.processed_run, args.trial)

    included_groups = {"blink"} if args.blink_only else {"blink", "suspected", "invalid"}

    truth = trial_eyelink_intervals(eye, args.trial)
    raw_pred = webcam_intervals(
        raw,
        raw_trial_start,
        raw_trial_end,
        included_groups,
    )
    proc_pred = webcam_intervals(
        proc,
        proc_trial_start,
        proc_trial_end,
        included_groups,
    )

    print("\n=== Trial-relative intervals (ms) ===")
    print("RAW:")
    for interval in raw_pred:
        print(f"  {interval[0]:.1f} - {interval[1]:.1f}  [{interval[2]}]")
    print(f"{args.processed_label}:")
    for interval in proc_pred:
        print(f"  {interval[0]:.1f} - {interval[1]:.1f}  [{interval[2]}]")
    print("EyeLink:")
    for interval in truth:
        print(f"  {interval[0]:.1f} - {interval[1]:.1f}")

    rows, match_rows = [], []
    for label, pred in [("Raw", raw_pred), (args.processed_label, proc_pred)]:
        matches, tp, fp, fn, precision, recall, f1 = match_events(pred, truth, args.tolerance_ms)
        rows.append({"method": label, "trial": args.trial, "tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall, "f1": f1,
                     "n_predicted": len(pred), "n_eyelink": len(truth),
                     "tolerance_ms": args.tolerance_ms,
                     "included_groups": ",".join(sorted(included_groups)),
                     "trial_window_source": "phases_detected.json"})
        matched_p = {i:j for i,j in matches}
        for i, interval in enumerate(pred):
            j = matched_p.get(i)
            match_rows.append({"method": label, "trial": args.trial, "pred_index": i,
                               "pred_start_ms": interval[0], "pred_end_ms": interval[1], "reason_group": interval[2],
                               "matched": j is not None, "eyelink_index": j,
                               "eyelink_start_ms": truth[j][0] if j is not None else np.nan,
                               "eyelink_end_ms": truth[j][1] if j is not None else np.nan})

    out = args.output_dir or (args.processed_run / "blink_preprocessing_comparison")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "blink_preprocessing_metrics.csv", index=False)
    pd.DataFrame(match_rows).to_csv(out / "blink_event_matches.csv", index=False)

    pdf = out / f"trial_{args.trial:02d}_raw_vs_{args.processed_label.replace(' ','_')}_vs_eyelink.pdf"
    with PdfPages(pdf) as pages:
        fig, axes = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True)
        datasets = [("Raw 60 Hz", raw_pred), (args.processed_label, proc_pred)]
        group_styles = {
            "blink": {"alpha": 0.38, "hatch": None},
            "suspected": {"alpha": 0.24, "hatch": "////"},
            "invalid": {"alpha": 0.18, "hatch": "xxxx"},
        }
        for ax, (label, intervals) in zip(axes[:2], datasets):
            for a, b, group in intervals:
                style = group_styles.get(group, group_styles["invalid"])
                ax.axvspan(a / 1000, b / 1000, alpha=style["alpha"], hatch=style["hatch"])
            ax.set_ylabel(label)
            ax.set_yticks([])
            ax.grid(True, axis="x", linewidth=0.4)

        for a, b in truth:
            axes[2].axvspan(a / 1000, b / 1000, alpha=0.3)
        axes[2].set_ylabel("EyeLink EBLINK")
        axes[2].set_yticks([])
        axes[2].grid(True, axis="x", linewidth=0.4)
        axes[-1].set_xlabel("Time from fixation start (s)")
        fig.suptitle(f"Blink preprocessing comparison — Trial {args.trial}")
        fig.tight_layout()
        pages.savefig(fig)
        plt.close(fig)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nPDF:     {pdf}")
    print(f"Metrics: {out / 'blink_preprocessing_metrics.csv'}")
    print(f"Matches: {out / 'blink_event_matches.csv'}")


if __name__ == "__main__":
    main()
