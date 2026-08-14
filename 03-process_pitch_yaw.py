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
Convert session CSV direction vectors to world yaw/pitch.
"""

from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MAX_ANGLE = np.pi / 2.0

GAZE_COLS = ["eyeDirX", "eyeDirY", "eyeDirZ"]
INTEND_COLS = ["IntendX", "IntendY", "IntendZ"]
ZERO_GAZE_COL = "isZeroGaze"
ZERO_INTEND_COL = "isZeroIntend"
TASK_COL = "Task"

IO_JOBS = [
    ("main", "session_csv", "processed_csv"),
    ("transition", "session_csv_transition", "processed_csv_transition"),
]


def world_yaw_pitch(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    valid = np.all(np.isfinite(vectors), axis=1) & (norms >= 1e-8)
    yaw = np.full(len(vectors), np.nan, dtype=np.float64)
    pitch = np.full(len(vectors), np.nan, dtype=np.float64)
    if np.any(valid):
        d = vectors[valid] / norms[valid, None]
        x, y, z = d[:, 0], d[:, 1], d[:, 2]
        yaw[valid] = np.arctan2(x, z)
        pitch[valid] = np.arctan2(y, np.sqrt(x * x + z * z))
    return yaw, pitch


def fill_angles(values: np.ndarray) -> tuple[np.ndarray, int]:
    s = pd.Series(values, dtype="float64")
    n_nan = int(s.isna().sum())
    if n_nan:
        s = s.ffill().bfill().fillna(0.0)
    return s.to_numpy(dtype=np.float64), n_nan


def add_angle_columns(
    df: pd.DataFrame,
    *,
    vector_cols: list[str],
    prefix: str,
) -> dict[str, int]:
    missing = [c for c in vector_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required vector columns for {prefix}: {missing}")
    vectors = df[vector_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    yaw, pitch = world_yaw_pitch(vectors)
    yaw, yaw_nan = fill_angles(yaw)
    pitch, pitch_nan = fill_angles(pitch)
    df[f"{prefix}Yaw"] = yaw
    df[f"{prefix}Pitch"] = pitch
    df[f"{prefix}YawNorm"] = np.clip(yaw / MAX_ANGLE, -1.0, 1.0)
    df[f"{prefix}PitchNorm"] = np.clip(pitch / MAX_ANGLE, -1.0, 1.0)
    return {
        f"{prefix}_yaw_nan_filled": yaw_nan,
        f"{prefix}_pitch_nan_filled": pitch_nan,
    }


def summarize_error(df: pd.DataFrame) -> dict[str, Any]:
    gaze = df[GAZE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    intend = df[INTEND_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    gn = np.linalg.norm(gaze, axis=1)
    tn = np.linalg.norm(intend, axis=1)
    valid = (
        np.all(np.isfinite(gaze), axis=1)
        & np.all(np.isfinite(intend), axis=1)
        & (gn >= 1e-8)
        & (tn >= 1e-8)
    )
    if ZERO_GAZE_COL in df.columns:
        valid &= ~pd.to_numeric(df[ZERO_GAZE_COL], errors="coerce").fillna(0).astype(bool).to_numpy()
    if ZERO_INTEND_COL in df.columns:
        valid &= ~pd.to_numeric(df[ZERO_INTEND_COL], errors="coerce").fillna(0).astype(bool).to_numpy()
    if not np.any(valid):
        return {"valid_error_count": 0}
    gv = gaze[valid] / gn[valid, None]
    tv = intend[valid] / tn[valid, None]
    dot = np.clip(np.sum(gv * tv, axis=1), -1.0, 1.0)
    err = np.degrees(np.arccos(dot))
    return {
        "valid_error_count": int(err.size),
        "angle_error_mean_deg": float(np.mean(err)),
        "angle_error_median_deg": float(np.median(err)),
        "angle_error_p90_deg": float(np.percentile(err, 90)),
    }


def process_one(task: tuple[str, str, str]) -> dict[str, Any]:
    input_path_s, input_root_s, output_root_s = task
    input_path = Path(input_path_s)
    input_root = Path(input_root_s)
    output_root = Path(output_root_s)
    rel = input_path.relative_to(input_root)
    output_path = output_root / rel.with_name(rel.stem + "_proc.csv")

    df = pd.read_csv(input_path, keep_default_na=False, na_values=[], low_memory=False)
    gaze_counts = add_angle_columns(df, vector_cols=GAZE_COLS, prefix="gaze")
    intend_counts = add_angle_columns(df, vector_cols=INTEND_COLS, prefix="intend")
    err_summary = summarize_error(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    row: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "rows": int(len(df)),
    }
    row.update(gaze_counts)
    row.update(intend_counts)
    if ZERO_GAZE_COL in df.columns:
        row["zero_gaze_frames"] = int(pd.to_numeric(df[ZERO_GAZE_COL], errors="coerce").fillna(0).astype(bool).sum())
    if ZERO_INTEND_COL in df.columns:
        row["zero_intend_frames"] = int(pd.to_numeric(df[ZERO_INTEND_COL], errors="coerce").fillna(0).astype(bool).sum())
    row.update(err_summary)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_job(label: str, input_dir: Path, output_dir: Path, workers: int) -> dict[str, Any]:
    files = sorted(p for p in input_dir.rglob("*.csv") if not p.name.startswith("_"))
    if not files:
        return {"label": label, "input_dir": str(input_dir), "output_dir": str(output_dir), "files": 0}
    tasks = [(str(p), str(input_dir), str(output_dir)) for p in files]
    n_workers = max(1, min(int(workers), len(tasks)))
    print(f"[{label}] processing {len(tasks)} files with {n_workers} workers")
    if n_workers == 1:
        rows = [process_one(t) for t in tasks]
    else:
        with Pool(processes=n_workers) as pool:
            rows = list(pool.imap_unordered(process_one, tasks, chunksize=1))
    rows = sorted(rows, key=lambda r: r["input"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "_process_pitch_yaw_manifest.csv", rows)
    summary = {
        "label": label,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": len(rows),
        "rows": int(sum(r.get("rows", 0) for r in rows)),
        "gaze_yaw_nan_filled": int(sum(r.get("gaze_yaw_nan_filled", 0) for r in rows)),
        "gaze_pitch_nan_filled": int(sum(r.get("gaze_pitch_nan_filled", 0) for r in rows)),
        "intend_yaw_nan_filled": int(sum(r.get("intend_yaw_nan_filled", 0) for r in rows)),
        "intend_pitch_nan_filled": int(sum(r.get("intend_pitch_nan_filled", 0) for r in rows)),
        "zero_gaze_frames": int(sum(r.get("zero_gaze_frames", 0) for r in rows)),
        "zero_intend_frames": int(sum(r.get("zero_intend_frames", 0) for r in rows)),
        "coordinate_system": "world",
    }
    (output_dir / "process_pitch_yaw_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add world yaw/pitch columns to session CSV files.")
    parser.add_argument("--base-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for label, input_name, output_name in IO_JOBS:
        summaries.append(run_job(label, args.base_dir / input_name, args.base_dir / output_name, args.workers))
    out = args.base_dir / "process_pitch_yaw_all_summary.json"
    out.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Summary: {out}")


if __name__ == "__main__":
    main()
