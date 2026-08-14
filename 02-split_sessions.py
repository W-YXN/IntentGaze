#!/usr/bin/env python3
# Copyright 2026 Xinan Yan and The Hong Kong University of Science and Technology (Guangzhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Split resampled IntentGaze CSV files into task sessions.
"""

from __future__ import annotations

import argparse
import csv
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

TASK_COL = "Task"
ZERO_GAZE_COL = "isZeroGaze"

TRIM_FIXATION_START = 10
PRE_CONTEXT_FRAMES = 20
POST_CONTEXT_FRAMES = 30
MAX_SACCADE_FRAMES = 27

SESSION_ROLE_COL = "SessionRole"
SESSION_EVENT_COL = "SessionEventType"
SOURCE_FILE_COL = "SourceFile"
SOURCE_RUN_INDEX_COL = "SourceRunIndex"
SOURCE_START_COL = "SourceStartIdx"
SOURCE_END_COL = "SourceEndIdx"

FILENAME_PREFIX_REPLACE = {
    "TrialData_Saccade-Fixation_": "TDSF_",
    "TrialData_Pursuit_": "TDP_",
}


def normalize_task(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def get_task_runs(df: pd.DataFrame) -> list[dict[str, Any]]:
    if TASK_COL not in df.columns:
        raise ValueError(f"Missing required column: {TASK_COL}")
    tasks = [normalize_task(v) for v in df[TASK_COL].tolist()]
    runs: list[dict[str, Any]] = []
    start = None
    current = None
    for i, task in enumerate(tasks):
        if task is None:
            if start is not None:
                runs.append({"task": current, "start": start, "end": i - 1})
                start = None
                current = None
            continue
        if start is None:
            start = i
            current = task
        elif task != current:
            runs.append({"task": current, "start": start, "end": i - 1})
            start = i
            current = task
    if start is not None:
        runs.append({"task": current, "start": start, "end": len(tasks) - 1})
    return runs


def is_saccade(task: Any) -> bool:
    return normalize_task(task) == "Saccade"


def is_fixation(task: Any) -> bool:
    return normalize_task(task) == "Fixation"


def should_keep_saccade(start: int, end: int) -> bool:
    length = end - start + 1
    return MAX_SACCADE_FRAMES <= 0 or length <= MAX_SACCADE_FRAMES


def shorten_basename(basename: str, max_len: int = 60) -> str:
    out = basename
    for old, new in FILENAME_PREFIX_REPLACE.items():
        out = out.replace(old, new)
    return out[:max_len] if len(out) > max_len else out


def role_tag(task: str) -> str:
    mapping = {
        "Fixation": "FIX",
        "Pursuit": "PUR",
    }
    return mapping.get(task, task.upper()[:8])


def attach_metadata(
    seg: pd.DataFrame,
    *,
    session_role: str,
    event_type: str,
    source_file: str,
    run_index: int,
    run_start: int,
    run_end: int,
) -> pd.DataFrame:
    out = seg.copy()
    out[SESSION_ROLE_COL] = session_role
    out[SESSION_EVENT_COL] = event_type
    out[SOURCE_FILE_COL] = source_file
    out[SOURCE_RUN_INDEX_COL] = run_index
    out[SOURCE_START_COL] = run_start
    out[SOURCE_END_COL] = run_end
    return out


def trim_fixation_onset(seg: pd.DataFrame, n_trim: int = TRIM_FIXATION_START) -> tuple[pd.DataFrame, int]:
    if n_trim <= 0 or len(seg) == 0:
        return seg, 0
    dropped = min(n_trim, len(seg))
    return seg.iloc[dropped:].reset_index(drop=True), dropped


def make_saccade_label(length: int, local_start: int, local_end: int) -> np.ndarray:
    label = np.zeros(length, dtype=np.float64)
    label[local_start:local_end + 1] = 1.0
    return label


def session_output_name(base_short: str, seq_id: int, tag: str) -> str:
    return f"{base_short}__S{seq_id:04d}__{tag}.csv"


def transition_output_name(base_short: str, seq_id: int) -> str:
    return f"{base_short}__T{seq_id:04d}__TRANS.csv"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def process_one(task: tuple[str, str, str, str]) -> dict[str, Any]:
    in_path_s, input_root_s, session_root_s, transition_root_s = task
    in_path = Path(in_path_s)
    input_root = Path(input_root_s)
    session_root = Path(session_root_s)
    transition_root = Path(transition_root_s)
    rel = in_path.relative_to(input_root)
    base_short = shorten_basename(in_path.stem)
    source_name = str(rel).replace(os.sep, "/")

    df = pd.read_csv(in_path, keep_default_na=False, na_values=[], low_memory=False)
    if TASK_COL not in df.columns:
        raise ValueError(f"{in_path}: missing {TASK_COL}")
    if ZERO_GAZE_COL not in df.columns:
        raise ValueError(f"{in_path}: missing {ZERO_GAZE_COL}; run 01-resample_dataset.py first")

    runs = get_task_runs(df)
    session_count = 0
    transition_count = 0
    trimmed_count = 0
    skipped_long_saccade = 0
    session_records: list[dict[str, Any]] = []
    transition_records: list[dict[str, Any]] = []

    for run_idx, run in enumerate(runs):
        task_name = str(run["task"])
        start = int(run["start"])
        end = int(run["end"])
        if is_saccade(task_name):
            if not should_keep_saccade(start, end):
                skipped_long_saccade += 1
                continue
            pre_start = max(0, start - PRE_CONTEXT_FRAMES)
            post_end = min(len(df) - 1, end + POST_CONTEXT_FRAMES)
            seg = df.iloc[pre_start:post_end + 1].reset_index(drop=True)
            local_start = start - pre_start
            local_end = end - pre_start
            seg = attach_metadata(
                seg,
                session_role="Transition",
                event_type="SaccadeTransition",
                source_file=source_name,
                run_index=run_idx,
                run_start=pre_start,
                run_end=post_end,
            )
            seg["SaccadeProb"] = make_saccade_label(len(seg), local_start, local_end)
            out_rel = rel.parent / transition_output_name(base_short, transition_count + 1)
            out_path = transition_root / out_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            seg.to_csv(out_path, index=False, encoding="utf-8")
            transition_count += 1
            transition_records.append({
                "source_file": source_name,
                "output_file": str(out_rel).replace(os.sep, "/"),
                "run_index": run_idx,
                "saccade_start_idx": start,
                "saccade_end_idx": end,
                "saccade_len": end - start + 1,
                "segment_start_idx": pre_start,
                "segment_end_idx": post_end,
                "segment_len": len(seg),
                "saccade_local_start": local_start,
                "saccade_local_end": local_end,
                "pre_context_frames": local_start,
                "post_context_frames": len(seg) - 1 - local_end,
            })
            continue

        seg = df.iloc[start:end + 1].reset_index(drop=True)
        dropped = 0
        if is_fixation(task_name):
            seg, dropped = trim_fixation_onset(seg, TRIM_FIXATION_START)
            trimmed_count += dropped
        if len(seg) == 0:
            continue
        seg = attach_metadata(
            seg,
            session_role=task_name,
            event_type=task_name,
            source_file=source_name,
            run_index=run_idx,
            run_start=start + dropped,
            run_end=end,
        )
        tag = role_tag(task_name)
        out_rel = rel.parent / session_output_name(base_short, session_count + 1, tag)
        out_path = session_root / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seg.to_csv(out_path, index=False, encoding="utf-8")
        session_count += 1
        session_records.append({
            "source_file": source_name,
            "output_file": str(out_rel).replace(os.sep, "/"),
            "run_index": run_idx,
            "task": task_name,
            "start_idx": start,
            "end_idx": end,
            "trimmed_start_frames": dropped,
            "output_len": len(seg),
            "zero_gaze_frames": int(pd.to_numeric(seg[ZERO_GAZE_COL], errors="coerce").fillna(0).astype(bool).sum()),
        })

    return {
        "input": str(in_path),
        "session_count": session_count,
        "transition_count": transition_count,
        "trimmed_count": trimmed_count,
        "skipped_long_saccade": skipped_long_saccade,
        "session_records": session_records,
        "transition_records": transition_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split 90 Hz CSV files into Task sessions.")
    parser.add_argument("--input-dir", type=Path, default=SCRIPT_DIR / "resampled_csv")
    parser.add_argument("--session-dir", type=Path, default=SCRIPT_DIR / "session_csv")
    parser.add_argument("--transition-dir", type=Path, default=SCRIPT_DIR / "session_csv_transition")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(p for p in args.input_dir.rglob("*.csv") if not p.name.startswith("_"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {args.input_dir}")
    args.session_dir.mkdir(parents=True, exist_ok=True)
    args.transition_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(p), str(args.input_dir), str(args.session_dir), str(args.transition_dir)) for p in files]
    workers = max(1, min(int(args.workers), len(tasks)))
    print(f"Splitting {len(tasks)} files with {workers} workers")
    if workers == 1:
        results = [process_one(t) for t in tasks]
    else:
        with Pool(processes=workers) as pool:
            results = list(pool.imap_unordered(process_one, tasks, chunksize=1))

    session_records = [r for item in results for r in item["session_records"]]
    transition_records = [r for item in results for r in item["transition_records"]]
    write_csv(args.session_dir / "_session_index.csv", session_records)
    write_csv(args.transition_dir / "_transition_index.csv", transition_records)
    summary_rows = []
    for item in sorted(results, key=lambda x: x["input"]):
        summary_rows.append({
            "input": item["input"],
            "session_count": item["session_count"],
            "transition_count": item["transition_count"],
            "trimmed_count": item["trimmed_count"],
            "skipped_long_saccade": item["skipped_long_saccade"],
        })
    write_csv(args.session_dir / "_split_summary.csv", summary_rows)
    print(f"Done. Sessions: {len(session_records)} -> {args.session_dir}")
    print(f"Transitions: {len(transition_records)} -> {args.transition_dir}")


if __name__ == "__main__":
    main()
