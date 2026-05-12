#!/usr/bin/env python3
"""
Convert processed CSV sessions to compact NPY tensors.
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
DEG_SCALE = 90.0

EVENT_MAP = {
    "Fixation": 0,
    "Pursuit": 1,
    "Saccade": 2,
}
CONTEXT_MAP = {
    "Vehicle-Movement": 0,
    "Standing": 1,
    "Walking": 2,
    "Sitting": 3,
    "Lying": 4,
}

GAZE_COLS = ["gazeYawNorm", "gazePitchNorm"]
INTEND_COLS = ["intendYawNorm", "intendPitchNorm"]
HEAD_GYRO_COLS = ["headAngVelX", "headAngVelY", "headAngVelZ"]
HEAD_VEL_COLS = ["headVelX", "headVelY", "headVelZ"]
CHEST_GYRO_COLS = ["chestTrackerAngVelX", "chestTrackerAngVelY", "chestTrackerAngVelZ"]
CHEST_VEL_COLS = ["chestTrackerVelX", "chestTrackerVelY", "chestTrackerVelZ"]
ZERO_GAZE_COL = "isZeroGaze"
ZERO_INTEND_COL = "isZeroIntend"
TASK_COL = "Task"
CONTEXT_COL = "CONTEXT"
USER_COL = "UserID"

SESSION_META_COLS = [
    "SessionRole",
    "SessionEventType",
    "SourceFile",
    "SourceRunIndex",
    "SourceStartIdx",
    "SourceEndIdx",
]

NORMALIZED_FEATURES = [
    "gaze_vel_yaw",
    "gaze_vel_pitch",
    "gaze_acc_yaw",
    "gaze_acc_pitch",
    "head_gyro_x",
    "head_gyro_y",
    "head_gyro_z",
    "head_vel_x",
    "head_vel_y",
    "head_vel_z",
    "chest_gyro_x",
    "chest_gyro_y",
    "chest_gyro_z",
    "chest_vel_x",
    "chest_vel_y",
    "chest_vel_z",
]

INPUT_FEATURE_NAMES = [
    "gaze_yaw_norm",
    "gaze_pitch_norm",
    "gaze_vel_yaw_norm",
    "gaze_vel_pitch_norm",
    "gaze_acc_yaw_norm",
    "gaze_acc_pitch_norm",
    "head_gyro_x_norm",
    "head_gyro_y_norm",
    "head_gyro_z_norm",
    "head_vel_x_norm",
    "head_vel_y_norm",
    "head_vel_z_norm",
    "chest_gyro_x_norm",
    "chest_gyro_y_norm",
    "chest_gyro_z_norm",
    "chest_vel_x_norm",
    "chest_vel_y_norm",
    "chest_vel_z_norm",
]

DATA_JOBS = [
    {"label": "main", "input_name": "processed_csv", "output_name": "npy_data", "apply_iqr": True},
    {
        "label": "transition",
        "input_name": "processed_csv_transition",
        "output_name": "npy_data_transition",
        "apply_iqr": False,
    },
]


def as_float(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)


def first_text(df: pd.DataFrame, col: str, default: str = "") -> str:
    if col not in df.columns:
        return default
    vals = df[col].dropna()
    if vals.empty:
        return default
    text = str(vals.iloc[0]).strip()
    return text if text else default


def first_int(df: pd.DataFrame, col: str, default: int = -1) -> int:
    if col not in df.columns:
        return default
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return default
    return int(vals.iloc[0])


def bool_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(bool).to_numpy()


def task_to_cls(values: pd.Series) -> np.ndarray:
    out = np.full(len(values), -1, dtype=np.int64)
    for name, idx in EVENT_MAP.items():
        out[values.astype(str).to_numpy() == name] = int(idx)
    return out


def context_to_id(values: pd.Series) -> np.ndarray:
    out = np.full(len(values), -1, dtype=np.int64)
    raw = values.astype(str).to_numpy()
    for name, idx in CONTEXT_MAP.items():
        out[raw == name] = int(idx)
    return out


def raw_feature_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    gaze = as_float(df, GAZE_COLS)
    gaze_yaw = gaze[:, 0]
    gaze_pitch = gaze[:, 1]
    gaze_vel_yaw = np.gradient(gaze_yaw)
    gaze_vel_pitch = np.gradient(gaze_pitch)
    gaze_acc_yaw = np.gradient(gaze_vel_yaw)
    gaze_acc_pitch = np.gradient(gaze_vel_pitch)

    head_gyro = as_float(df, HEAD_GYRO_COLS)
    head_vel = as_float(df, HEAD_VEL_COLS)
    chest_gyro = as_float(df, CHEST_GYRO_COLS)
    chest_vel = as_float(df, CHEST_VEL_COLS)

    return {
        "gaze_vel_yaw": gaze_vel_yaw,
        "gaze_vel_pitch": gaze_vel_pitch,
        "gaze_acc_yaw": gaze_acc_yaw,
        "gaze_acc_pitch": gaze_acc_pitch,
        "head_gyro_x": head_gyro[:, 0],
        "head_gyro_y": head_gyro[:, 1],
        "head_gyro_z": head_gyro[:, 2],
        "head_vel_x": head_vel[:, 0],
        "head_vel_y": head_vel[:, 1],
        "head_vel_z": head_vel[:, 2],
        "chest_gyro_x": chest_gyro[:, 0],
        "chest_gyro_y": chest_gyro[:, 1],
        "chest_gyro_z": chest_gyro[:, 2],
        "chest_vel_x": chest_vel[:, 0],
        "chest_vel_y": chest_vel[:, 1],
        "chest_vel_z": chest_vel[:, 2],
    }


def stats_for_file(path: str) -> dict[str, dict[str, float]] | None:
    try:
        df = pd.read_csv(path, keep_default_na=False, na_values=[], low_memory=False)
        arrays = raw_feature_arrays(df)
        out: dict[str, dict[str, float]] = {}
        for name, arr in arrays.items():
            arr = np.asarray(arr, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            out[name] = {
                "sum": float(np.sum(arr)),
                "sum_sq": float(np.sum(arr * arr)),
                "count": int(arr.size),
            }
        return out
    except Exception as exc:
        print(f"[WARN] stats failed for {path}: {exc}")
        return None


def aggregate_stats(stats_list: list[dict[str, dict[str, float]] | None]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, float]] = {}
    for stats in stats_list:
        if not stats:
            continue
        for name, payload in stats.items():
            store = merged.setdefault(name, {"sum": 0.0, "sum_sq": 0.0, "count": 0.0})
            store["sum"] += float(payload["sum"])
            store["sum_sq"] += float(payload["sum_sq"])
            store["count"] += int(payload["count"])

    params: dict[str, dict[str, Any]] = {}
    for name in NORMALIZED_FEATURES:
        store = merged.get(name, {"sum": 0.0, "sum_sq": 0.0, "count": 0.0})
        n = int(store["count"])
        if n < 2:
            mean, std = 0.0, 1.0
        else:
            mean = float(store["sum"] / n)
            var = max(0.0, float(store["sum_sq"] / n) - mean * mean)
            std = float(np.sqrt(var))
            if std < 1e-8:
                std = 1.0
        params[name] = {"mean": mean, "std": std, "count": n, "clip_std": 5.0}
    return params


def normalize_feature(name: str, values: np.ndarray, params: dict[str, dict[str, Any]]) -> np.ndarray:
    p = params[name]
    mean = float(p["mean"])
    std = float(p["std"]) if float(p["std"]) >= 1e-8 else 1.0
    clip_std = float(p.get("clip_std", 5.0))
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, mean - clip_std * std, mean + clip_std * std)
    return (clipped - mean) / std


def build_input(df: pd.DataFrame, params: dict[str, dict[str, Any]]) -> np.ndarray:
    gaze = as_float(df, GAZE_COLS)
    raw = raw_feature_arrays(df)
    cols = [
        gaze[:, 0],
        gaze[:, 1],
        normalize_feature("gaze_vel_yaw", raw["gaze_vel_yaw"], params),
        normalize_feature("gaze_vel_pitch", raw["gaze_vel_pitch"], params),
        normalize_feature("gaze_acc_yaw", raw["gaze_acc_yaw"], params),
        normalize_feature("gaze_acc_pitch", raw["gaze_acc_pitch"], params),
        normalize_feature("head_gyro_x", raw["head_gyro_x"], params),
        normalize_feature("head_gyro_y", raw["head_gyro_y"], params),
        normalize_feature("head_gyro_z", raw["head_gyro_z"], params),
        normalize_feature("head_vel_x", raw["head_vel_x"], params),
        normalize_feature("head_vel_y", raw["head_vel_y"], params),
        normalize_feature("head_vel_z", raw["head_vel_z"], params),
        normalize_feature("chest_gyro_x", raw["chest_gyro_x"], params),
        normalize_feature("chest_gyro_y", raw["chest_gyro_y"], params),
        normalize_feature("chest_gyro_z", raw["chest_gyro_z"], params),
        normalize_feature("chest_vel_x", raw["chest_vel_x"], params),
        normalize_feature("chest_vel_y", raw["chest_vel_y"], params),
        normalize_feature("chest_vel_z", raw["chest_vel_z"], params),
    ]
    return np.stack(cols, axis=1).astype(np.float32)


def build_targets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    gaze = as_float(df, GAZE_COLS)
    intend = as_float(df, INTEND_COLS)
    target_reg_true = (intend - gaze).astype(np.float32)
    target_cls = task_to_cls(df[TASK_COL])
    target_ctx = context_to_id(df[CONTEXT_COL])
    is_zero_gaze = bool_col(df, ZERO_GAZE_COL)
    is_zero_intend = bool_col(df, ZERO_INTEND_COL)
    saccade_mask = target_cls == EVENT_MAP["Saccade"]

    target_reg = target_reg_true.copy()
    target_reg[saccade_mask] = 0.0
    target_vel = np.stack(
        [np.gradient(target_reg[:, 0]), np.gradient(target_reg[:, 1])],
        axis=1,
    ).astype(np.float32)

    valid_regression_mask = (
        np.isin(target_cls, [EVENT_MAP["Fixation"], EVENT_MAP["Pursuit"]])
        & np.isfinite(target_reg_true).all(axis=1)
        & (~is_zero_gaze)
        & (~is_zero_intend)
    )
    return {
        "target_reg": target_reg.astype(np.float32),
        "target_reg_true": target_reg_true.astype(np.float32),
        "target_vel": target_vel,
        "target_cls": target_cls.astype(np.int64),
        "target_ctx": target_ctx.astype(np.int64),
        "is_zero_gaze": is_zero_gaze,
        "is_zero_intend": is_zero_intend,
        "valid_regression_mask": valid_regression_mask,
        "saccade_zero_target_mask": saccade_mask,
    }


def output_metadata(df: pd.DataFrame, input_path: Path) -> dict[str, Any]:
    meta = {
        "subject_id": first_int(df, USER_COL, -1),
        "original_file": input_path.name,
        "source_file": first_text(df, "SourceFile", input_path.name),
        "session_role": first_text(df, "SessionRole", ""),
        "session_event_type": first_text(df, "SessionEventType", ""),
        "source_run_index": first_int(df, "SourceRunIndex", -1),
        "source_start_idx": first_int(df, "SourceStartIdx", -1),
        "source_end_idx": first_int(df, "SourceEndIdx", -1),
        "event_map": EVENT_MAP,
        "context_map": CONTEXT_MAP,
        "input_feature_names": INPUT_FEATURE_NAMES,
        "input_dim": 18,
        "motion_feature_mode": "sdk_vel_angvel",
        "regression_coordinate_mode": "yaw_pitch",
    }
    return meta


def process_one(task: tuple[str, str, str, str]) -> dict[str, Any]:
    path_s, input_root_s, output_root_s, norm_json_s = task
    input_path = Path(path_s)
    input_root = Path(input_root_s)
    output_root = Path(output_root_s)
    with open(norm_json_s, "r", encoding="utf-8") as f:
        norm_params = json.load(f)

    df = pd.read_csv(input_path, keep_default_na=False, na_values=[], low_memory=False)
    required = GAZE_COLS + INTEND_COLS + HEAD_GYRO_COLS + HEAD_VEL_COLS + CHEST_GYRO_COLS + CHEST_VEL_COLS + [
        TASK_COL,
        CONTEXT_COL,
        USER_COL,
        ZERO_GAZE_COL,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{input_path}: missing required columns {missing}")

    x = build_input(df, norm_params)
    targets = build_targets(df)
    rel = input_path.relative_to(input_root)
    out_path = output_root / rel.with_suffix(".npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "input": x,
        "norm_params": norm_params,
        "normalization_source": "global",
    }
    payload.update(targets)
    payload.update(output_metadata(df, input_path))
    if "SaccadeProb" in df.columns:
        payload["saccade_prob"] = pd.to_numeric(df["SaccadeProb"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    else:
        payload["saccade_prob"] = None

    np.save(out_path, payload)
    valid = targets["valid_regression_mask"]
    err_deg = np.linalg.norm(targets["target_reg_true"][valid].astype(np.float64) * DEG_SCALE, axis=1)
    mae = float(np.mean(err_deg)) if err_deg.size else np.nan
    dominant_context = int(np.bincount(targets["target_ctx"][targets["target_ctx"] >= 0]).argmax()) if np.any(targets["target_ctx"] >= 0) else -1
    valid_cls = targets["target_cls"][np.isin(targets["target_cls"], [0, 1])]
    dominant_event = int(np.bincount(valid_cls, minlength=2).argmax()) if valid_cls.size else -1
    return {
        "input": str(input_path),
        "output": str(out_path),
        "rows": int(len(df)),
        "mae_deg": mae,
        "dominant_context": dominant_context,
        "dominant_event": dominant_event,
        "valid_regression_frames": int(valid.sum()),
        "zero_gaze_frames": int(targets["is_zero_gaze"].sum()),
        "zero_intend_frames": int(targets["is_zero_intend"].sum()),
        "saccade_frames": int(targets["saccade_zero_target_mask"].sum()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_csv_files(base_dir: Path, input_names: list[str]) -> list[Path]:
    files: list[Path] = []
    for name in input_names:
        root = base_dir / name
        if root.exists():
            files.extend(p for p in root.rglob("*.csv") if not p.name.startswith("_"))
    return sorted(files)


def prepare_norm_params(base_dir: Path, jobs: list[dict[str, Any]], workers: int) -> Path:
    norm_path = base_dir / "global_norm_params_shared.json"
    files = collect_csv_files(base_dir, [j["input_name"] for j in jobs])
    if not files:
        raise FileNotFoundError("No processed CSV files found for global normalization")
    tasks = [str(p) for p in files]
    n_workers = max(1, min(workers, len(tasks)))
    print(f"[norm] scanning {len(tasks)} files with {n_workers} workers")
    if n_workers == 1:
        stats = [stats_for_file(t) for t in tasks]
    else:
        with Pool(processes=n_workers) as pool:
            stats = list(pool.imap_unordered(stats_for_file, tasks, chunksize=1))
    params = aggregate_stats(stats)
    norm_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    return norm_path


def apply_iqr_filter(rows: list[dict[str, Any]], output_dir: Path, k: float = 1.5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    df = pd.DataFrame(rows)
    if df.empty or "mae_deg" not in df.columns:
        return rows, []
    eligible = df[
        np.isfinite(pd.to_numeric(df["mae_deg"], errors="coerce"))
        & (df["dominant_event"].isin([0, 1]))
        & (df["dominant_context"] >= 0)
    ].copy()
    remove_indices: set[int] = set()
    for (_ctx, _evt), group in eligible.groupby(["dominant_context", "dominant_event"]):
        if len(group) < 4:
            continue
        q1 = float(group["mae_deg"].quantile(0.25))
        q3 = float(group["mae_deg"].quantile(0.75))
        upper = q3 + k * (q3 - q1)
        remove_indices.update(int(i) for i in group[group["mae_deg"] > upper].index)
    removed = []
    kept = []
    for idx, row in enumerate(rows):
        if idx in remove_indices:
            out_path = Path(str(row["output"]))
            if out_path.exists():
                out_path.unlink()
            r = dict(row)
            r["iqr_removed"] = True
            removed.append(r)
        else:
            r = dict(row)
            r["iqr_removed"] = False
            kept.append(r)
    write_csv(output_dir / "_iqr_removed_sessions.csv", removed)
    return kept, removed


def run_job(job: dict[str, Any], base_dir: Path, norm_path: Path, workers: int) -> dict[str, Any]:
    input_dir = base_dir / str(job["input_name"])
    output_dir = base_dir / str(job["output_name"])
    files = sorted(p for p in input_dir.rglob("*.csv") if not p.name.startswith("_")) if input_dir.exists() else []
    if not files:
        return {"label": job["label"], "files": 0, "converted": 0, "removed_by_iqr": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(str(p), str(input_dir), str(output_dir), str(norm_path)) for p in files]
    n_workers = max(1, min(workers, len(tasks)))
    print(f"[{job['label']}] converting {len(tasks)} files with {n_workers} workers")
    if n_workers == 1:
        rows = [process_one(t) for t in tasks]
    else:
        with Pool(processes=n_workers) as pool:
            rows = list(pool.imap_unordered(process_one, tasks, chunksize=1))
    rows = sorted(rows, key=lambda r: r["input"])
    removed: list[dict[str, Any]] = []
    if bool(job.get("apply_iqr", False)):
        rows, removed = apply_iqr_filter(rows, output_dir, k=1.5)
    write_csv(output_dir / "_npy_manifest.csv", rows)
    summary = {
        "label": job["label"],
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": len(files),
        "converted": len(rows),
        "removed_by_iqr": len(removed),
        "input_dim": 18,
        "motion_feature_mode": "sdk_vel_angvel",
        "regression_coordinate_mode": "yaw_pitch",
        "iqr_enabled": bool(job.get("apply_iqr", False)),
        "iqr_k": 1.5 if bool(job.get("apply_iqr", False)) else None,
    }
    (output_dir / "process_to_npy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert processed CSV files to 18D NPY tensors.")
    parser.add_argument("--base-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workers = max(1, int(args.workers))
    norm_path = prepare_norm_params(args.base_dir, DATA_JOBS, workers)
    summaries = []
    for job in DATA_JOBS:
        summaries.append(run_job(job, args.base_dir, norm_path, workers))
    (args.base_dir / "process_to_npy_all_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Shared norm: {norm_path}")


if __name__ == "__main__":
    main()
