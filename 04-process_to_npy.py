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

import fold_split


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


def process_one(task: tuple[str, str, str, str, int | None]) -> dict[str, Any]:
    path_s, input_root_s, output_root_s, norm_json_s, fold_index = task
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
    if fold_index is not None:
        payload["fold_index"] = int(fold_index)
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
        "subject_id": int(payload["subject_id"]),
        "fold_index": fold_index,
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


def subject_id_for_csv(path: Path) -> int:
    try:
        values = pd.read_csv(path, usecols=[USER_COL], nrows=1)[USER_COL]
    except Exception as exc:
        raise ValueError(f"Could not read {USER_COL} from {path}: {exc}") from exc
    if values.empty:
        raise ValueError(f"{path}: no {USER_COL} value")
    subject_id = pd.to_numeric(values.iloc[0], errors="coerce")
    if not np.isfinite(subject_id):
        raise ValueError(f"{path}: invalid {USER_COL} value {values.iloc[0]!r}")
    return int(subject_id)


def validate_cv_subjects(files: list[Path]) -> None:
    observed = {subject_id_for_csv(path) for path in files}
    unexpected = sorted(observed - fold_split.all_subjects())
    if unexpected:
        raise ValueError(
            "CV preprocessing refuses unknown subject IDs. "
            f"Observed IDs outside fold_split.py: {unexpected}"
        )


def files_for_subjects(files: list[Path], subject_ids: set[int]) -> list[Path]:
    return [path for path in files if subject_id_for_csv(path) in subject_ids]


def prepare_norm_params(norm_path: Path, files: list[Path], workers: int) -> Path:
    if not files:
        raise FileNotFoundError("No processed CSV files found for normalization")
    tasks = [str(p) for p in files]
    n_workers = max(1, min(workers, len(tasks)))
    print(f"[norm] scanning {len(tasks)} files with {n_workers} workers -> {norm_path.name}")
    if n_workers == 1:
        stats = [stats_for_file(t) for t in tasks]
    else:
        with Pool(processes=n_workers) as pool:
            stats = list(pool.imap_unordered(stats_for_file, tasks, chunksize=1))
    params = aggregate_stats(stats)
    norm_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    return norm_path


def apply_iqr_filter(
    rows: list[dict[str, Any]],
    output_dir: Path,
    k: float = 1.5,
    fit_subject_ids: set[int] | None = None,
    removal_subject_ids: set[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    df = pd.DataFrame(rows)
    if df.empty or "mae_deg" not in df.columns:
        return rows, [], []
    eligible = df[
        np.isfinite(pd.to_numeric(df["mae_deg"], errors="coerce"))
        & (df["dominant_event"].isin([0, 1]))
        & (df["dominant_context"] >= 0)
    ].copy()
    remove_indices: set[int] = set()
    thresholds: list[dict[str, Any]] = []
    for (_ctx, _evt), group in eligible.groupby(["dominant_context", "dominant_event"]):
        fit_group = group if fit_subject_ids is None else group[group["subject_id"].isin(fit_subject_ids)]
        threshold_info: dict[str, Any] = {
            "dominant_context": int(_ctx),
            "dominant_event": int(_evt),
            "n_all_eligible": int(len(group)),
            "n_threshold_fit": int(len(fit_group)),
        }
        if len(fit_group) < 4:
            threshold_info["status"] = "insufficient_training_sessions"
            thresholds.append(threshold_info)
            continue
        q1 = float(fit_group["mae_deg"].quantile(0.25))
        q3 = float(fit_group["mae_deg"].quantile(0.75))
        upper = q3 + k * (q3 - q1)
        candidates = group[group["mae_deg"] > upper]
        if removal_subject_ids is not None:
            candidates = candidates[candidates["subject_id"].isin(removal_subject_ids)]
        remove_indices.update(int(i) for i in candidates.index)
        threshold_info.update({"status": "applied", "q1": q1, "q3": q3, "upper": upper, "n_removed": int(len(candidates))})
        thresholds.append(threshold_info)
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
    (output_dir / "_iqr_thresholds.json").write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return kept, removed, thresholds


def run_job(
    job: dict[str, Any],
    base_dir: Path,
    norm_path: Path,
    workers: int,
    fold: dict[str, Any] | None,
    iqr_train_only: bool,
) -> dict[str, Any]:
    input_dir = base_dir / str(job["input_name"])
    fold_suffix = f"_{fold['fold_name']}" if fold is not None else ""
    output_dir = base_dir / f"{job['output_name']}{fold_suffix}"
    files = sorted(p for p in input_dir.rglob("*.csv") if not p.name.startswith("_")) if input_dir.exists() else []
    if not files:
        return {"label": job["label"], "files": 0, "converted": 0, "removed_by_iqr": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_index = int(fold["fold_index"]) if fold is not None else None
    tasks = [(str(p), str(input_dir), str(output_dir), str(norm_path), fold_index) for p in files]
    n_workers = max(1, min(workers, len(tasks)))
    print(f"[{job['label']}] converting {len(tasks)} files with {n_workers} workers")
    if n_workers == 1:
        rows = [process_one(t) for t in tasks]
    else:
        with Pool(processes=n_workers) as pool:
            rows = list(pool.imap_unordered(process_one, tasks, chunksize=1))
    rows = sorted(rows, key=lambda r: r["input"])
    removed: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    if bool(job.get("apply_iqr", False)):
        if fold is not None and iqr_train_only:
            train_subjects = set(fold["train_subject_ids"])
            rows, removed, thresholds = apply_iqr_filter(
                rows,
                output_dir,
                k=1.5,
                fit_subject_ids=train_subjects,
                removal_subject_ids=train_subjects,
            )
        else:
            rows, removed, thresholds = apply_iqr_filter(rows, output_dir, k=1.5)
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
        "fold_index": fold_index,
        "normalization_fit_subject_ids": fold.get("normalization_fit_subject_ids") if fold else "all_csv_explicit",
        "iqr_threshold_fit_subject_ids": fold.get("train_subject_ids") if (fold and iqr_train_only) else "all_trials",
        "iqr_removal_subject_ids": fold.get("train_subject_ids") if (fold and iqr_train_only) else "all_trials",
        "iqr_threshold_groups": thresholds,
        "iqr_policy": (
            "Validation and test trials were not excluded based on target-referenced error IQR "
            "and did not contribute to threshold estimation."
            if fold and iqr_train_only
            else "IQR was applied to all eligible trials (explicit legacy/global mode)."
        ),
    }
    (output_dir / "process_to_npy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_fold_provenance(base_dir: Path, fold: dict[str, Any], norm_path: Path, all_files: list[Path]) -> None:
    main_dir = base_dir / f"npy_data_{fold['fold_name']}"
    main_dir.mkdir(parents=True, exist_ok=True)
    norm_fit_subjects = set(fold["normalization_fit_subject_ids"])
    fit_files = files_for_subjects(all_files, norm_fit_subjects)
    fit_subjects_seen = sorted({subject_id_for_csv(path) for path in fit_files})
    test_subjects = set(fold["test_subject_ids"])
    leaked = sorted(test_subjects & set(fit_subjects_seen))
    if leaked:
        raise RuntimeError(f"Leakage guard tripped for {fold['fold_name']}: test subjects {leaked} entered normalization.")
    provenance = {
        **fold,
        "normalization_params": str(norm_path),
        "normalization_fit_subjects_seen": fit_subjects_seen,
        "normalization_fit_file_count": len(fit_files),
        "leaked_test_subjects": leaked,
        "iqr_threshold_fit_subject_ids": fold["train_subject_ids"],
        "iqr_removal_subject_ids": fold["train_subject_ids"],
        "iqr_policy": "Validation and test trials were not excluded based on target-referenced error IQR and did not contribute to threshold estimation.",
    }
    (main_dir / f"{fold['fold_name']}_preprocessing_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert processed CSV files to 18D NPY tensors.")
    parser.add_argument("--base-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    parser.add_argument(
        "--fold-index",
        type=int,
        action="append",
        help="Process only this 0-based CV fold. May be supplied more than once; default processes all folds.",
    )
    parser.add_argument(
        "--global-normalization-all-csv",
        action="store_true",
        help="Explicit legacy mode: write one unsuffixed dataset using normalization fit on all CSV files.",
    )
    iqr_policy = parser.add_mutually_exclusive_group()
    iqr_policy.add_argument(
        "--iqr-train-only",
        dest="iqr_train_only",
        action="store_true",
        default=True,
        help=(
            "Default CV policy: estimate IQR thresholds from training trials and remove only training trials; "
            "validation/test trials are retained."
        ),
    )
    iqr_policy.add_argument(
        "--iqr-all-trials",
        dest="iqr_train_only",
        action="store_false",
        help="Explicit legacy mode: let validation/test trials contribute to and be removed by IQR filtering.",
    )
    args = parser.parse_args()
    if args.global_normalization_all_csv and args.fold_index:
        parser.error("--global-normalization-all-csv cannot be combined with --fold-index")
    return args


def main() -> None:
    args = parse_args()
    workers = max(1, int(args.workers))
    summaries = []
    all_files = collect_csv_files(args.base_dir, [j["input_name"] for j in DATA_JOBS])
    if not all_files:
        raise FileNotFoundError("No processed CSV files found")

    if args.global_normalization_all_csv:
        norm_path = prepare_norm_params(args.base_dir / "global_norm_params_shared.json", all_files, workers)
        for job in DATA_JOBS:
            summaries.append(run_job(job, args.base_dir, norm_path, workers, fold=None, iqr_train_only=False))
    else:
        validate_cv_subjects(all_files)
        requested_folds = args.fold_index or list(range(fold_split.N_FOLDS))
        for fold_index in requested_folds:
            fold = fold_split.build_fold_split(fold_index)
            norm_files = files_for_subjects(all_files, set(fold["normalization_fit_subject_ids"]))
            norm_path = prepare_norm_params(
                args.base_dir / f"global_norm_params_{fold['fold_name']}.json", norm_files, workers
            )
            write_fold_provenance(args.base_dir, fold, norm_path, all_files)
            for job in DATA_JOBS:
                summaries.append(run_job(job, args.base_dir, norm_path, workers, fold=fold, iqr_train_only=args.iqr_train_only))
    (args.base_dir / "process_to_npy_all_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Done. Independent fold preprocessing completed." if not args.global_normalization_all_csv else "Done. Explicit all-CSV global preprocessing completed.")


if __name__ == "__main__":
    main()
