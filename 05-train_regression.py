#!/usr/bin/env python3
"""
Train the published IntentGaze regression model on a given training split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fold_split


SCRIPT_DIR = Path(__file__).resolve().parent

WINDOW_SIZE = 64
HORIZON = 1
INPUT_DIM = 18
GAZE_DIM = 6
MOTION_DIM = 12
HIDDEN = 32
MOTION_HIDDEN = 16
TCN_KERNEL_SIZE = 3
TCN_DILATIONS = [1, 2, 4, 8]
DROPOUT = 0.15
DEG_SCALE = 90.0
DELTA_MAX = 1.0


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def yawpitch_to_dir(yawpitch: torch.Tensor) -> torch.Tensor:
    rad = yawpitch * (math.pi / 2.0)
    yaw = rad[..., 0]
    pitch = rad[..., 1]
    cos_p = torch.cos(pitch)
    return torch.stack([cos_p * torch.sin(yaw), torch.sin(pitch), cos_p * torch.cos(yaw)], dim=-1)


def dir_to_yawpitch(direction: torch.Tensor) -> torch.Tensor:
    d = F.normalize(direction, dim=-1, eps=1e-12)
    yaw = torch.atan2(d[..., 0], d[..., 2])
    pitch = torch.asin(d[..., 1].clamp(-1.0, 1.0))
    return torch.stack([yaw, pitch], dim=-1) / (math.pi / 2.0)


def mean3_anchor_from_raw(rawyp: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    d = (
        yawpitch_to_dir(rawyp[target_idx - 1])
        + yawpitch_to_dir(rawyp[target_idx - 2])
        + yawpitch_to_dir(rawyp[target_idx - 3])
    )
    return dir_to_yawpitch(d)


def angle_from_yawpitch_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = yawpitch_to_dir(pred)
    t = yawpitch_to_dir(target)
    cos = (p * t).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad_len = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad_len, 0)))


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)
        h = self.drop(F.relu(self.bn1(self.conv1(x))))
        h = self.drop(F.relu(self.bn2(self.conv2(h))))
        return F.relu(h + res)


class MotionTCNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        in_ch = MOTION_DIM
        for dilation in TCN_DILATIONS:
            blocks.append(TCNBlock(in_ch, MOTION_HIDDEN, TCN_KERNEL_SIZE, dilation, DROPOUT))
            in_ch = MOTION_HIDDEN
        self.tcn = nn.Sequential(*blocks)

    def forward(self, x_m: torch.Tensor) -> torch.Tensor:
        h_seq = self.tcn(x_m).transpose(1, 2).contiguous()
        return h_seq[:, -1, :]


class GazeTwoLayerGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(GAZE_DIM, HIDDEN, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(DROPOUT)
        self.gru2 = nn.GRU(HIDDEN, HIDDEN, num_layers=1, batch_first=True)

    def forward(self, x_g: torch.Tensor) -> torch.Tensor:
        h1, _ = self.gru1(x_g.transpose(1, 2).contiguous())
        h1 = self.drop(h1)
        h2, _ = self.gru2(h1)
        return h2[:, -1, :]


class ExpertHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 2),
        )
        self.delta_max = DELTA_MAX

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.delta_max * torch.tanh(self.net(h))


class GazeRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gaze = GazeTwoLayerGRU()
        self.motion = MotionTCNEncoder()
        self.motion_proj = nn.Linear(MOTION_HIDDEN, HIDDEN)
        self.film = nn.Linear(HIDDEN, HIDDEN * 2)
        self.ln = nn.LayerNorm(HIDDEN)
        self.head = ExpertHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != INPUT_DIM:
            raise ValueError(f"expected x shape [B,{INPUT_DIM},T], got {tuple(x.shape)}")
        x_g = x[:, :GAZE_DIM, :]
        x_m = x[:, GAZE_DIM:, :]
        h_g = self.gaze(x_g)
        h_m = self.motion_proj(self.motion(x_m))
        raw_gamma, raw_beta = self.film(h_m).chunk(2, dim=-1)
        gamma = 1.0 + 0.25 * torch.tanh(raw_gamma)
        beta = 0.25 * torch.tanh(raw_beta)
        z = self.ln(gamma * h_g + beta)
        return self.head(z)


class GpuWindowDataset:
    def __init__(self, files: list[Path], device: torch.device):
        if not files:
            raise ValueError("empty training file list")
        self.device = device
        frame_inputs: list[torch.Tensor] = []
        frame_rawyp: list[torch.Tensor] = []
        frame_gtyp: list[torch.Tensor] = []
        starts: list[torch.Tensor] = []
        file_rows: list[dict[str, Any]] = []
        offset = 0
        for path in files:
            payload = np.load(path, allow_pickle=True).item()
            x_np = np.asarray(payload["input"], dtype=np.float32)
            if x_np.ndim != 2 or x_np.shape[1] != INPUT_DIM:
                raise ValueError(f"{path}: expected input shape [N,{INPUT_DIM}], got {x_np.shape}")
            rawyp_np = x_np[:, :2].astype(np.float32, copy=False)
            target_true_np = np.asarray(payload["target_reg_true"], dtype=np.float32)
            if "valid_regression_mask" not in payload:
                raise ValueError(f"{path}: missing required valid_regression_mask")
            valid_np = np.asarray(payload["valid_regression_mask"], dtype=bool).reshape(-1)
            if len(target_true_np) != len(x_np) or len(valid_np) != len(x_np):
                raise ValueError(f"{path}: inconsistent frame lengths")
            gtyp_np = rawyp_np + target_true_np
            n = int(len(x_np))
            if n <= WINDOW_SIZE:
                file_rows.append({"file": str(path), "frames": n, "windows": 0, "used_as_train": 0})
                continue
            local_starts = np.arange(0, n - WINDOW_SIZE, dtype=np.int64)
            target_idx = local_starts + WINDOW_SIZE
            trainable = valid_np[target_idx]
            local_starts = local_starts[trainable]
            if local_starts.size:
                starts.append(torch.from_numpy(local_starts + offset))
            frame_inputs.append(torch.from_numpy(np.ascontiguousarray(x_np)))
            frame_rawyp.append(torch.from_numpy(np.ascontiguousarray(rawyp_np)))
            frame_gtyp.append(torch.from_numpy(np.ascontiguousarray(gtyp_np)))
            file_rows.append({
                "file": str(path),
                "frames": n,
                "windows": int(n - WINDOW_SIZE),
                "used_as_train": int(local_starts.size),
            })
            offset += n
        if not starts:
            raise RuntimeError("no trainable windows; check valid_regression_mask and sequence lengths")
        self.frame_inputs = torch.cat(frame_inputs, dim=0).to(device)
        self.frame_rawyp = torch.cat(frame_rawyp, dim=0).to(device)
        self.frame_gtyp = torch.cat(frame_gtyp, dim=0).to(device)
        self.window_starts = torch.cat(starts, dim=0).long().to(device)
        self.input_offsets = torch.arange(WINDOW_SIZE, device=device, dtype=torch.long)
        self.file_rows = file_rows

    def __len__(self) -> int:
        return int(self.window_starts.numel())

    def get_batch(self, batch_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        starts = self.window_starts[batch_indices]
        input_idx = starts.unsqueeze(1) + self.input_offsets.unsqueeze(0)
        target_idx = starts + WINDOW_SIZE
        x = self.frame_inputs[input_idx].permute(0, 2, 1).contiguous()
        anchor = mean3_anchor_from_raw(self.frame_rawyp, target_idx)
        gt = self.frame_gtyp[target_idx]
        return x, anchor, gt

    def summary(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "files": len(self.file_rows),
            "frames": int(self.frame_inputs.shape[0]),
            "train_windows": len(self),
            "window_size": WINDOW_SIZE,
            "horizon": HORIZON,
            "input_dim": INPUT_DIM,
            "gaze_dim": GAZE_DIM,
            "motion_dim": MOTION_DIM,
        }


def read_train_list(path: Path, train_dir: Path) -> list[Path]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
        out: list[Path] = []
        for row in rows:
            raw = row.get("path") or row.get("file") or row.get("npy") or row.get("output")
            if raw:
                p = Path(raw)
                out.append(p if p.is_absolute() else train_dir / p)
        return out
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        out.append(p if p.is_absolute() else train_dir / p)
    return out


def resolve_train_files(train_dir: Path, train_list: Path | None) -> list[Path]:
    files = read_train_list(train_list, train_dir) if train_list else sorted(train_dir.rglob("*.npy"))
    files = [p.resolve() for p in files if p.exists() and p.suffix.lower() == ".npy"]
    if not files:
        raise FileNotFoundError(f"no .npy training files found; train_dir={train_dir}, train_list={train_list}")
    return files


def validate_fold_training_files(
    files: list[Path], fold: dict[str, Any], normalization_params: dict[str, Any]
) -> list[Path]:
    """Keep only this fold's train subjects and verify fold-specific preprocessing."""
    train_subjects = set(fold["train_subject_ids"])
    selected: list[Path] = []
    for path in files:
        payload = np.load(path, allow_pickle=True).item()
        subject_id = int(payload.get("subject_id", -1))
        if subject_id not in train_subjects:
            continue
        if payload.get("fold_index") != fold["fold_index"]:
            raise ValueError(f"{path}: expected fold_index {fold['fold_index']}, found {payload.get('fold_index')}")
        if payload.get("norm_params") != normalization_params:
            raise ValueError(f"{path}: normalization parameters do not match {fold['fold_name']}")
        selected.append(path)
    if not selected:
        raise FileNotFoundError(f"No train-subject NPY files found for {fold['fold_name']}")
    return selected


def configure_fold(args: argparse.Namespace) -> None:
    if args.fold_index is None:
        args.train_dir = args.train_dir or (SCRIPT_DIR / "npy_data")
        args.output_dir = args.output_dir or (SCRIPT_DIR / "trained_regression")
        args.fold_info = None
        args.fold_norm_params = None
        return
    if args.train_dir is not None:
        raise ValueError("--train-dir cannot be combined with --fold-index; use --fold-base-dir instead")
    fold = fold_split.build_fold_split(args.fold_index)
    args.train_dir = args.fold_base_dir / f"npy_data_{fold['fold_name']}"
    args.output_dir = args.output_dir or (args.fold_base_dir / f"trained_regression_{fold['fold_name']}")
    norm_path = args.fold_base_dir / f"global_norm_params_{fold['fold_name']}.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing fold-specific normalization parameters: {norm_path}")
    args.fold_info = fold
    args.fold_norm_params = json.loads(norm_path.read_text(encoding="utf-8"))
    args.fold_norm_path = norm_path


def cosine_restart_lr_factor(
    epoch: int,
    warmup_epochs: int,
    cosine_t0: int,
    cosine_t_mult: int,
    cosine_eta_min: float,
    restart_decay: float,
) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return max(epoch / warmup_epochs, 1e-6)

    t = max(epoch - warmup_epochs - 1, 0)
    cycle_len = max(int(cosine_t0), 1)
    cycle = 0
    while t >= cycle_len:
        t -= cycle_len
        cycle += 1
        cycle_len *= max(int(cosine_t_mult), 1)

    phase = t / max(cycle_len, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * phase))
    amplitude = float(restart_decay) ** cycle
    return float(cosine_eta_min) + (1.0 - float(cosine_eta_min)) * amplitude * cosine


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_files = resolve_train_files(args.train_dir, args.train_list)
    if args.fold_info is not None:
        train_files = validate_fold_training_files(train_files, args.fold_info, args.fold_norm_params)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = GpuWindowDataset(train_files, device)
    model = GazeRegressionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = {
        "script": Path(__file__).name,
        "task": "gaze_regression",
        "train_dir": str(args.train_dir),
        "train_list": str(args.train_list) if args.train_list else "",
        "fold": args.fold_info,
        "fold_normalization_params": str(args.fold_norm_path) if args.fold_info else "",
        "output_dir": str(out_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "cosine_t0": args.cosine_t0,
        "cosine_t_mult": args.cosine_t_mult,
        "cosine_eta_min": args.cosine_eta_min,
        "restart_decay": args.restart_decay,
        "seed": args.seed,
        "device": str(device),
        "model": {
            "name": "GazeRegressionModel",
            "input_dim": INPUT_DIM,
            "gaze_dim": GAZE_DIM,
            "motion_dim": MOTION_DIM,
            "hidden": HIDDEN,
            "motion_hidden": MOTION_HIDDEN,
            "tcn_kernel_size": TCN_KERNEL_SIZE,
            "tcn_dilations": TCN_DILATIONS,
            "dropout": DROPOUT,
            "delta_max": DELTA_MAX,
            "window_size": WINDOW_SIZE,
            "horizon": HORIZON,
        },
        "dataset": dataset.summary(),
    }
    save_json(out_dir / "train_config.json", config)
    write_csv(out_dir / "train_files.csv", dataset.file_rows)

    rows: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        lr_mult = cosine_restart_lr_factor(
            epoch=epoch,
            warmup_epochs=args.warmup_epochs,
            cosine_t0=args.cosine_t0,
            cosine_t_mult=args.cosine_t_mult,
            cosine_eta_min=args.cosine_eta_min,
            restart_decay=args.restart_decay,
        )
        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_mult
        order = torch.randperm(len(dataset), device=device)
        sum_loss = torch.zeros((), device=device)
        sum_acc = torch.zeros((), device=device)
        sum_delta_norm = torch.zeros((), device=device)
        count = 0
        for start in range(0, len(dataset), args.batch_size):
            idx = order[start:start + args.batch_size]
            x, anchor, gt = dataset.get_batch(idx)
            delta = model(x)
            pred = anchor + delta
            loss = F.smooth_l1_loss(pred * DEG_SCALE, gt * DEG_SCALE)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            with torch.no_grad():
                n = int(idx.numel())
                acc = angle_from_yawpitch_deg(pred, gt).mean()
                delta_norm = torch.sqrt(((delta * DEG_SCALE) ** 2).sum(dim=1)).mean()
                sum_loss += loss.detach() * n
                sum_acc += acc.detach() * n
                sum_delta_norm += delta_norm.detach() * n
                count += n

        rec = {
            "epoch": epoch,
            "train_loss_deg_smoothl1": float((sum_loss / max(count, 1)).item()),
            "train_accuracy_deg": float((sum_acc / max(count, 1)).item()),
            "train_delta_norm_deg": float((sum_delta_norm / max(count, 1)).item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_clip": args.grad_clip,
            "last_grad_norm": float(torch.as_tensor(grad_norm).detach().cpu().item()),
            "train_windows": count,
        }
        rows.append(rec)
        torch.save(
            {"model_state_dict": model.state_dict(), "epoch": epoch, "config": config, "train_metrics": rec},
            out_dir / f"epoch_{epoch:03d}.pt",
        )
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"loss={rec['train_loss_deg_smoothl1']:.6f} "
            f"acc={rec['train_accuracy_deg']:.6f} "
            f"lr={rec['lr']:.3e}",
            flush=True,
        )
    write_csv(out_dir / "train_epoch_metrics.csv", rows)
    save_json(out_dir / "train_summary.json", {
        "status": "ok",
        "final_epoch": rows[-1],
        "config": config,
    })
    print(f"Done. Per-epoch checkpoints saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the published IntentGaze regression model.")
    parser.add_argument("--train-dir", type=Path, default=None, help="Directory containing training .npy files.")
    parser.add_argument("--train-list", type=Path, default=None, help="Optional text/CSV list of training .npy files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--fold-index", type=int, default=None, help="0-based CV fold; uses matching fold NPY data and normalization parameters.")
    parser.add_argument("--fold-base-dir", type=Path, default=SCRIPT_DIR, help="Directory containing fold-specific NPY and normalization files.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train.")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=1.6e-3)
    parser.add_argument("--weight-decay", type=float, default=2.0e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--cosine-t0", type=int, default=10)
    parser.add_argument("--cosine-t-mult", type=int, default=2)
    parser.add_argument("--cosine-eta-min", type=float, default=0.01)
    parser.add_argument("--restart-decay", type=float, default=0.85)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="", help="cpu, cuda, cuda:0, etc. Default: cuda if available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    configure_fold(args)
    train(args)


if __name__ == "__main__":
    main()
