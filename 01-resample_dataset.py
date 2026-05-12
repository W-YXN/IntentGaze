#!/usr/bin/env python3
"""
Resample the published IntentGaze CSV dataset onto a uniform 90 Hz grid.
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
from scipy.spatial.transform import Rotation as R, Slerp


FS = 90.0
DT = 1.0 / FS
SCRIPT_DIR = Path(__file__).resolve().parent

TIME_COL = "Timestamp"
TASK_COL = "Task"
CONTEXT_COL = "CONTEXT"
USER_COL = "UserID"

HEAD_POS_COLS = ["headPosX", "headPosY", "headPosZ"]
HEAD_ROT_COLS = ["headRotX", "headRotY", "headRotZ", "headRotW"]
HEAD_SDK_VEL_COLS = ["headVelX", "headVelY", "headVelZ"]
HEAD_SDK_ANGVEL_COLS = ["headAngVelX", "headAngVelY", "headAngVelZ"]

CHEST_POS_COLS = ["chestTrackerPosX", "chestTrackerPosY", "chestTrackerPosZ"]
CHEST_ROT_COLS = ["chestTrackerRotX", "chestTrackerRotY", "chestTrackerRotZ", "chestTrackerRotW"]
CHEST_SDK_VEL_COLS = ["chestTrackerVelX", "chestTrackerVelY", "chestTrackerVelZ"]
CHEST_SDK_ANGVEL_COLS = ["chestTrackerAngVelX", "chestTrackerAngVelY", "chestTrackerAngVelZ"]

GAZE_COLS = ["eyeDirX", "eyeDirY", "eyeDirZ"]
BINOCULUS_COLS = ["BinoculusX", "BinoculusY", "BinoculusZ"]
TARGET_POS_COLS = ["TargetPosX", "TargetPosY", "TargetPosZ"]
INTEND_COLS = ["IntendX", "IntendY", "IntendZ"]

REQUIRED_COLUMNS = (
    [TIME_COL, USER_COL, TASK_COL, CONTEXT_COL]
    + HEAD_POS_COLS
    + HEAD_ROT_COLS
    + HEAD_SDK_VEL_COLS
    + HEAD_SDK_ANGVEL_COLS
    + CHEST_POS_COLS
    + CHEST_ROT_COLS
    + CHEST_SDK_VEL_COLS
    + CHEST_SDK_ANGVEL_COLS
    + GAZE_COLS
    + BINOCULUS_COLS
    + TARGET_POS_COLS
)


def build_uniform_time(t_sec: np.ndarray, fs: float = FS) -> np.ndarray:
    t_sec = np.asarray(t_sec, dtype=np.float64)
    if t_sec.size == 0:
        return np.empty((0,), dtype=np.float64)
    t0, t1 = float(t_sec[0]), float(t_sec[-1])
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.array([t0], dtype=np.float64)
    n = int(np.floor((t1 - t0) * fs + 1e-9)) + 1
    return t0 + np.arange(n, dtype=np.float64) / fs


def interp_scalar(t_sec: np.ndarray, t_new_sec: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.interp(t_new_sec, t_sec, values)


def interp_columns(df: pd.DataFrame, t_sec: np.ndarray, t_new_sec: np.ndarray, cols: list[str]) -> dict[str, np.ndarray]:
    return {col: interp_scalar(t_sec, t_new_sec, pd.to_numeric(df[col], errors="coerce").to_numpy()) for col in cols}


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float64)
    finite = np.all(np.isfinite(vectors), axis=1)
    norms = np.linalg.norm(vectors, axis=1)
    valid = finite & (norms >= eps)
    out = np.zeros_like(vectors, dtype=np.float64)
    out[valid] = vectors[valid] / norms[valid, None]
    return out, ~valid


def interp_direction(
    df: pd.DataFrame,
    t_sec: np.ndarray,
    t_new_sec: np.ndarray,
    cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    raw = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    _, invalid_original = normalize_vectors(raw)
    interp = np.stack([interp_scalar(t_sec, t_new_sec, raw[:, i]) for i in range(3)], axis=1)
    normalized, invalid_resampled = normalize_vectors(interp)
    invalid_marker = resample_binary(t_sec, t_new_sec, invalid_original) | invalid_resampled
    normalized[invalid_marker] = 0.0
    return normalized, invalid_marker


def interp_quaternion(df: pd.DataFrame, t_sec: np.ndarray, t_new_sec: np.ndarray, cols: list[str]) -> np.ndarray:
    quats = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    quats[~np.isfinite(quats)] = np.nan
    norms = np.linalg.norm(quats, axis=1)
    quats[(~np.isfinite(norms)) | (norms < 1e-12)] = np.nan
    if np.isnan(quats).any():
        quats = pd.DataFrame(quats).ffill().bfill().to_numpy(dtype=np.float64)
    if np.isnan(quats).any():
        quats[np.any(np.isnan(quats), axis=1)] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    quats = quats / np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12, None)
    if len(quats) == 1:
        return np.repeat(quats, len(t_new_sec), axis=0)
    rots = R.from_quat(quats)
    slerp = Slerp(t_sec, rots)
    return slerp(np.clip(t_new_sec, t_sec[0], t_sec[-1])).as_quat()


def resample_task(t_sec: np.ndarray, t_new_sec: np.ndarray, task_series: pd.Series) -> list[str]:
    arr = task_series.astype(str).to_numpy(dtype=object)
    idxs = np.searchsorted(t_sec, t_new_sec, side="left")
    out: list[str] = []
    for idx in idxs:
        j_right = idx if idx < len(arr) else len(arr) - 1
        out.append(str(arr[j_right]))
    return out


def resample_binary(t_sec: np.ndarray, t_new_sec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    idxs = np.searchsorted(t_sec, t_new_sec, side="left")
    out = np.zeros(len(t_new_sec), dtype=bool)
    for i, idx in enumerate(idxs):
        j_right = idx if idx < len(mask) else len(mask) - 1
        j_left = max(j_right - 1, 0)
        out[i] = bool(mask[j_right] or mask[j_left])
    return out


def first_value(df: pd.DataFrame, col: str) -> Any:
    values = df[col].dropna()
    return values.iloc[0] if not values.empty else ""


def fill_binoculus_for_intend(origin_raw: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, int]:
    origin_raw = np.asarray(origin_raw, dtype=np.float64)
    origin_norm = np.linalg.norm(origin_raw, axis=1)
    valid_origin = np.all(np.isfinite(origin_raw), axis=1) & (origin_norm >= eps)
    origin_for_intend = origin_raw.copy()
    origin_for_intend[~valid_origin] = np.nan
    origin_filled = pd.DataFrame(origin_for_intend).ffill().to_numpy(dtype=np.float64)

    replaced = ~valid_origin & np.all(np.isfinite(origin_filled), axis=1)
    return origin_filled, valid_origin, int(replaced.sum())


def build_intend_vectors(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    target = df[TARGET_POS_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    origin_raw = df[BINOCULUS_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    origin, valid_origin, replaced_count = fill_binoculus_for_intend(origin_raw)
    intend, invalid_intend = normalize_vectors(target - origin)
    stats = {
        "invalid_binoculus_original_rows": int((~valid_origin).sum()),
        "binoculus_rows_filled_from_previous_for_intend": replaced_count,
        "binoculus_rows_without_previous_for_intend": int((~valid_origin).sum() - replaced_count),
    }
    return intend, invalid_intend, stats


def process_one(task: tuple[str, str, str]) -> dict[str, Any]:
    in_path_s, input_root_s, output_root_s = task
    in_path = Path(in_path_s)
    input_root = Path(input_root_s)
    output_root = Path(output_root_s)
    rel = in_path.relative_to(input_root)
    out_path = output_root / rel.with_name(rel.stem + "_resampled.csv")

    df = pd.read_csv(in_path, keep_default_na=False, na_values=[], low_memory=False)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{in_path}: missing required columns: {missing}")

    df = df.sort_values(TIME_COL).drop_duplicates(TIME_COL)
    if len(df) < 2:
        raise ValueError(f"{in_path}: not enough rows after Timestamp de-duplication")

    timestamp = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=np.float64)
    valid_time = np.isfinite(timestamp)
    if not np.all(valid_time):
        df = df.loc[valid_time].copy()
        timestamp = timestamp[valid_time]
    if len(df) < 2:
        raise ValueError(f"{in_path}: not enough finite Timestamp rows")

    t0 = float(timestamp[0])
    t_sec = (timestamp - t0) / 1000.0
    t_new_sec = build_uniform_time(t_sec, FS)
    timestamp_new = t0 + t_new_sec * 1000.0

    head_pos = interp_columns(df, t_sec, t_new_sec, HEAD_POS_COLS)
    chest_pos = interp_columns(df, t_sec, t_new_sec, CHEST_POS_COLS)
    head_vel = interp_columns(df, t_sec, t_new_sec, HEAD_SDK_VEL_COLS)
    head_angvel = interp_columns(df, t_sec, t_new_sec, HEAD_SDK_ANGVEL_COLS)
    chest_vel = interp_columns(df, t_sec, t_new_sec, CHEST_SDK_VEL_COLS)
    chest_angvel = interp_columns(df, t_sec, t_new_sec, CHEST_SDK_ANGVEL_COLS)
    binoculus = interp_columns(df, t_sec, t_new_sec, BINOCULUS_COLS)
    target_pos = interp_columns(df, t_sec, t_new_sec, TARGET_POS_COLS)

    head_rot = interp_quaternion(df, t_sec, t_new_sec, HEAD_ROT_COLS)
    chest_rot = interp_quaternion(df, t_sec, t_new_sec, CHEST_ROT_COLS)
    gaze, is_zero_gaze = interp_direction(df, t_sec, t_new_sec, GAZE_COLS)

    intend_original, is_zero_intend_original, intend_stats = build_intend_vectors(df)
    intend_df = pd.DataFrame(intend_original, columns=INTEND_COLS)
    intend, is_zero_intend_interp = interp_direction(intend_df, t_sec, t_new_sec, INTEND_COLS)
    is_zero_intend = resample_binary(t_sec, t_new_sec, is_zero_intend_original) | is_zero_intend_interp

    out = pd.DataFrame()
    out[TIME_COL] = timestamp_new.astype(np.int64)
    out["timeSec"] = t_new_sec
    out[USER_COL] = first_value(df, USER_COL)
    out[CONTEXT_COL] = first_value(df, CONTEXT_COL)
    out[TASK_COL] = resample_task(t_sec, t_new_sec, df[TASK_COL])

    for col in HEAD_POS_COLS:
        out[col] = head_pos[col]
    out[HEAD_ROT_COLS] = head_rot
    for col in HEAD_SDK_VEL_COLS:
        out[col] = head_vel[col]
    for col in HEAD_SDK_ANGVEL_COLS:
        out[col] = head_angvel[col]

    for col in CHEST_POS_COLS:
        out[col] = chest_pos[col]
    out[CHEST_ROT_COLS] = chest_rot
    for col in CHEST_SDK_VEL_COLS:
        out[col] = chest_vel[col]
    for col in CHEST_SDK_ANGVEL_COLS:
        out[col] = chest_angvel[col]

    out[GAZE_COLS] = gaze
    for col in BINOCULUS_COLS:
        out[col] = binoculus[col]
    for col in TARGET_POS_COLS:
        out[col] = target_pos[col]
    out[INTEND_COLS] = intend
    out["isZeroGaze"] = is_zero_gaze.astype(np.int64)
    out["isZeroIntend"] = is_zero_intend.astype(np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return {
        "input": str(in_path),
        "output": str(out_path),
        "rows_in": int(len(df)),
        "rows_out": int(len(out)),
        "zero_gaze_frames": int(is_zero_gaze.sum()),
        "zero_intend_frames": int(is_zero_intend.sum()),
        **intend_stats,
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resample IntentGaze CSV files to a 90 Hz grid.")
    parser.add_argument("--input-dir", type=Path, default=SCRIPT_DIR / "Dataset")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "resampled_csv")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    files = sorted(input_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(p), str(input_dir), str(output_dir)) for p in files]
    workers = max(1, min(int(args.workers), len(tasks)))
    print(f"Resampling {len(tasks)} files with {workers} workers")
    if workers == 1:
        rows = [process_one(t) for t in tasks]
    else:
        with Pool(processes=workers) as pool:
            rows = list(pool.imap_unordered(process_one, tasks, chunksize=1))
    rows = sorted(rows, key=lambda r: r["input"])
    write_manifest(output_dir / "_resample_manifest.csv", rows)
    print(f"Done. Output: {output_dir}")
    print(f"Manifest: {output_dir / '_resample_manifest.csv'}")


if __name__ == "__main__":
    main()
