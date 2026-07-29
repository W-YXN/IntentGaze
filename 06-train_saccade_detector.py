#!/usr/bin/env python3
"""
Train the published IntentGaze saccade detector on a given training split.
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

WINDOW_SIZE = 7
INPUT_DIM = 18
KERNEL_SIZE = 3
HIDDEN_DIM = 16
DILATIONS = [1, 2]


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


def compute_receptive_field(kernel_size: int, dilations: list[int]) -> int:
    return 1 + (int(kernel_size) - 1) * int(sum(dilations))


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=int(kernel_size), dilation=int(dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class ResTCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(float(dropout))
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.drop(F.gelu(self.bn1(self.conv1(x))))
        h = self.drop(self.bn2(self.conv2(h)))
        return F.gelu(h + residual)


class SaccadeResTCN(nn.Module):
    def __init__(self, input_dim: int, dropout: float):
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = int(input_dim)
        for dilation in DILATIONS:
            blocks.append(ResTCNBlock(in_ch, HIDDEN_DIM, KERNEL_SIZE, int(dilation), float(dropout)))
            in_ch = HIDDEN_DIM
        self.backbone = nn.Sequential(*blocks)
        self.classifier = nn.Linear(HIDDEN_DIM, 1)
        self.input_dim = int(input_dim)
        self.hidden_dim = HIDDEN_DIM

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return h[:, :, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x)).squeeze(1)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().view(-1)
        logits = logits.view(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        return (alpha_t * ((1.0 - p_t) ** self.gamma) * bce).mean()


class GpuSaccadeWindowDataset:
    def __init__(self, files: list[Path], device: torch.device, window_size: int, use_padding_mask: bool):
        self.device = device
        self.window_size = int(window_size)
        self.use_padding_mask = bool(use_padding_mask)
        frames: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        starts: list[int] = []
        rows: list[dict[str, Any]] = []
        offset = 0
        input_dim: int | None = None
        for path in files:
            payload = np.load(path, allow_pickle=True).item()
            x = np.asarray(payload["input"], dtype=np.float32)
            if x.ndim != 2 or x.shape[1] != INPUT_DIM:
                raise ValueError(f"{path}: expected input shape [N,{INPUT_DIM}], got {x.shape}")
            cls = np.asarray(payload.get("target_cls", np.zeros(len(x), dtype=np.int64)), dtype=np.int64).reshape(-1)
            if len(cls) != len(x):
                raise ValueError(f"{path}: target_cls length mismatch")
            y = (cls == 2).astype(np.int64)
            original_frames = int(len(x))
            was_padded = False
            if len(x) < self.window_size:
                pad = self.window_size - len(x)
                x = np.pad(x, ((pad, 0), (0, 0)), mode="edge")
                y = np.pad(y, (pad, 0), mode="edge")
                was_padded = True
                if self.use_padding_mask:
                    mask = np.zeros((len(x), 1), dtype=np.float32)
                    mask[pad:] = 1.0
                    x = np.concatenate([x, mask], axis=1)
            elif self.use_padding_mask:
                x = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)

            if input_dim is None:
                input_dim = int(x.shape[1])
            elif int(x.shape[1]) != input_dim:
                raise ValueError(f"{path}: input dim changed from {input_dim} to {x.shape[1]}")

            local_starts = list(range(0, len(x) - self.window_size + 1))
            starts.extend(offset + s for s in local_starts)
            frames.append(np.ascontiguousarray(x))
            labels.append(np.ascontiguousarray(y))
            rows.append({
                "file": str(path),
                "frames": original_frames,
                "frames_after_padding": int(len(x)),
                "windows": int(len(local_starts)),
                "positive_frames": int(y.sum()),
                "was_padded": int(was_padded),
            })
            offset += len(x)

        if not starts or input_dim is None:
            raise RuntimeError("no trainable detector windows")
        self.frame_data = torch.from_numpy(np.concatenate(frames, axis=0)).to(device)
        frame_labels = torch.from_numpy(np.concatenate(labels, axis=0)).long().to(device)
        self.window_starts = torch.tensor(starts, dtype=torch.long, device=device)
        self.window_labels = frame_labels[self.window_starts + self.window_size - 1]
        self.offsets = torch.arange(self.window_size, device=device, dtype=torch.long)
        self.file_rows = rows
        self.input_dim = int(input_dim)

    def __len__(self) -> int:
        return int(self.window_starts.numel())

    def get_batch(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.window_starts[indices]
        idx = starts.unsqueeze(1) + self.offsets.unsqueeze(0)
        x = self.frame_data[idx].permute(0, 2, 1).contiguous()
        y = self.window_labels[indices]
        return x, y

    def summary(self) -> dict[str, Any]:
        counts = torch.bincount(self.window_labels, minlength=2).detach().cpu().numpy().astype(int)
        return {
            "device": str(self.device),
            "files": len(self.file_rows),
            "frames": int(self.frame_data.shape[0]),
            "windows": len(self),
            "input_dim": self.input_dim,
            "window_size": self.window_size,
            "positive_windows": int(counts[1]),
            "negative_windows": int(counts[0]),
            "positive_rate": float(counts[1] / max(counts.sum(), 1)),
            "use_padding_mask": self.use_padding_mask,
        }


def weighted_sample_indices(labels: torch.Tensor, n_samples: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=2).float().clamp_min_(1.0)
    weights = 1.0 / counts[labels]
    probs = weights / weights.sum()
    return torch.multinomial(probs, int(n_samples), replacement=True)


@torch.no_grad()
def train_metrics(model: nn.Module, dataset: GpuSaccadeWindowDataset, batch_size: int) -> dict[str, float]:
    model.eval()
    tp = fp = tn = fn = 0
    loss_sum = 0.0
    count = 0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for start in range(0, len(dataset), batch_size):
        idx = torch.arange(start, min(start + batch_size, len(dataset)), device=dataset.device)
        x, y = dataset.get_batch(idx)
        logits = model(x)
        prob = torch.sigmoid(logits)
        pred = prob >= 0.5
        truth = y.bool()
        tp += int((pred & truth).sum().item())
        fp += int((pred & ~truth).sum().item())
        tn += int((~pred & ~truth).sum().item())
        fn += int((~pred & truth).sum().item())
        loss_sum += float(criterion(logits, y.float()).item())
        count += int(y.numel())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "loss_bce": loss_sum / max(count, 1),
        "accuracy": (tp + tn) / max(count, 1),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


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


def read_train_list(path: Path) -> list[Path]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(text.splitlines()))
        out = []
        for row in rows:
            raw = row.get("path") or row.get("file") or row.get("npy") or row.get("output")
            if raw:
                out.append(Path(raw))
        return out
    return [Path(line.strip()) for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def resolve_train_files(main_dir: Path, transition_dir: Path, train_list: Path | None) -> list[Path]:
    if train_list:
        files = [p if p.is_absolute() else train_list.parent / p for p in read_train_list(train_list)]
    else:
        files = sorted(main_dir.rglob("*.npy")) + sorted(transition_dir.rglob("*.npy"))
    files = [p.resolve() for p in files if p.exists() and p.suffix.lower() == ".npy"]
    if not files:
        raise FileNotFoundError("no detector training npy files found")
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
        args.main_dir = args.main_dir or (SCRIPT_DIR / "npy_data")
        args.transition_dir = args.transition_dir or (SCRIPT_DIR / "npy_data_transition")
        args.output_dir = args.output_dir or (SCRIPT_DIR / "trained_saccade_detector")
        args.fold_info = None
        args.fold_norm_params = None
        return
    if args.main_dir is not None or args.transition_dir is not None:
        raise ValueError("--main-dir/--transition-dir cannot be combined with --fold-index; use --fold-base-dir instead")
    fold = fold_split.build_fold_split(args.fold_index)
    args.main_dir = args.fold_base_dir / f"npy_data_{fold['fold_name']}"
    args.transition_dir = args.fold_base_dir / f"npy_data_transition_{fold['fold_name']}"
    args.output_dir = args.output_dir or (args.fold_base_dir / f"trained_saccade_detector_{fold['fold_name']}")
    norm_path = args.fold_base_dir / f"global_norm_params_{fold['fold_name']}.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing fold-specific normalization parameters: {norm_path}")
    args.fold_info = fold
    args.fold_norm_params = json.loads(norm_path.read_text(encoding="utf-8"))
    args.fold_norm_path = norm_path


def train_stage(
    model: SaccadeResTCN,
    dataset: GpuSaccadeWindowDataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    out_dir: Path,
    config: dict[str, Any],
    epochs: int,
    batch_size: int,
    lr: float,
    args: argparse.Namespace,
    stage_name: str,
    weighted_sampler: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
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
            group["lr"] = float(lr) * lr_mult
        if weighted_sampler:
            order = weighted_sample_indices(dataset.window_labels, len(dataset))
        else:
            order = torch.randperm(len(dataset), device=dataset.device)

        loss_sum = torch.zeros((), device=dataset.device)
        count = 0
        last_grad_norm = 0.0
        for start in range(0, len(order), int(batch_size)):
            idx = order[start:start + int(batch_size)]
            x, y = dataset.get_batch(idx)
            logits = model(x)
            loss = criterion(logits, y.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()
            n = int(y.numel())
            loss_sum += loss.detach() * n
            count += n
            last_grad_norm = float(torch.as_tensor(grad_norm).detach().cpu().item())

        metrics = train_metrics(model, dataset, args.eval_batch_size)
        rec = {
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": float((loss_sum / max(count, 1)).item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "last_grad_norm": last_grad_norm,
            **{f"train_{k}": v for k, v in metrics.items()},
        }
        rows.append(rec)
        torch.save(
            {
                "stage": stage_name,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "train_metrics": rec,
            },
            out_dir / f"{stage_name}_epoch_{epoch:03d}.pt",
        )
        print(
            f"[{stage_name}] epoch {epoch:03d}/{epochs} "
            f"loss={rec['train_loss']:.6f} "
            f"bal_acc={rec['train_balanced_accuracy']:.4f} "
            f"recall={rec['train_recall']:.4f} "
            f"lr={rec['lr']:.3e}",
            flush=True,
        )
    return rows


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    files = resolve_train_files(args.main_dir, args.transition_dir, args.train_list)
    if args.fold_info is not None:
        files = validate_fold_training_files(files, args.fold_info, args.fold_norm_params)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = GpuSaccadeWindowDataset(files, device, args.window_size, args.use_padding_mask)
    model = SaccadeResTCN(
        input_dim=dataset.input_dim,
        dropout=args.dropout,
    ).to(device)

    config = {
        "script": Path(__file__).name,
        "task": "saccade_detection",
        "main_dir": str(args.main_dir),
        "transition_dir": str(args.transition_dir),
        "train_list": str(args.train_list) if args.train_list else "",
        "fold": args.fold_info,
        "fold_normalization_params": str(args.fold_norm_path) if args.fold_info else "",
        "output_dir": str(args.output_dir),
        "lr_schedule": {
            "type": "warmup_cosine_warm_restart",
            "warmup_epochs": args.warmup_epochs,
            "cosine_t0": args.cosine_t0,
            "cosine_t_mult": args.cosine_t_mult,
            "cosine_eta_min": args.cosine_eta_min,
            "restart_decay": args.restart_decay,
        },
        "model": {
            "type": "SaccadeResTCN",
            "input_dim": dataset.input_dim,
            "base_input_dim": INPUT_DIM,
            "padding_mask_channel": bool(args.use_padding_mask),
            "window_size": args.window_size,
            "kernel_size": KERNEL_SIZE,
            "dilations": DILATIONS,
            "receptive_field": compute_receptive_field(KERNEL_SIZE, DILATIONS),
            "hidden_dim": HIDDEN_DIM,
        },
        "dataset": dataset.summary(),
    }
    save_json(args.output_dir / "train_config.json", config)
    write_csv(args.output_dir / "train_files.csv", dataset.file_rows)

    stage1_opt = torch.optim.AdamW(model.parameters(), lr=args.stage1_lr, weight_decay=args.stage1_weight_decay)
    stage1_rows = train_stage(
        model=model,
        dataset=dataset,
        optimizer=stage1_opt,
        criterion=nn.BCEWithLogitsLoss(),
        out_dir=args.output_dir,
        config=config,
        epochs=args.stage1_epochs,
        batch_size=args.stage1_batch_size,
        lr=args.stage1_lr,
        args=args,
        stage_name="stage1",
        weighted_sampler=False,
    )
    torch.save(
        {
            "stage": "stage1_final",
            "model_state_dict": model.state_dict(),
            "config": config,
            "input_dim": dataset.input_dim,
            "window_size": args.window_size,
            "use_padding_mask": bool(args.use_padding_mask),
        },
        args.output_dir / "stage1_final_backbone.pt",
    )

    for p in model.backbone.parameters():
        p.requires_grad_(False)
    for p in model.classifier.parameters():
        p.requires_grad_(True)

    stage2_opt = torch.optim.AdamW(model.classifier.parameters(), lr=args.stage2_lr, weight_decay=args.stage2_weight_decay)
    stage2_rows = train_stage(
        model=model,
        dataset=dataset,
        optimizer=stage2_opt,
        criterion=FocalLoss(gamma=args.stage2_focal_gamma, alpha=args.stage2_focal_alpha),
        out_dir=args.output_dir,
        config=config,
        epochs=args.stage2_epochs,
        batch_size=args.stage2_batch_size,
        lr=args.stage2_lr,
        args=args,
        stage_name="stage2",
        weighted_sampler=True,
    )

    rows = stage1_rows + stage2_rows
    write_csv(args.output_dir / "train_epoch_metrics.csv", rows)
    final_metrics = train_metrics(model, dataset, args.eval_batch_size)
    save_json(args.output_dir / "train_summary.json", {
        "status": "ok",
        "final_train_metrics": final_metrics,
        "config": config,
    })
    print(f"Done. Per-epoch checkpoints saved to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the published IntentGaze saccade detector.")
    parser.add_argument("--main-dir", type=Path, default=None)
    parser.add_argument("--transition-dir", type=Path, default=None)
    parser.add_argument("--train-list", type=Path, default=None, help="Optional text/CSV list of training npy files.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fold-index", type=int, default=None, help="0-based CV fold; uses matching fold NPY data and normalization parameters.")
    parser.add_argument("--fold-base-dir", type=Path, default=SCRIPT_DIR, help="Directory containing fold-specific NPY and normalization files.")
    parser.add_argument("--stage1-epochs", type=int, required=True)
    parser.add_argument("--stage2-epochs", type=int, required=True)
    parser.add_argument("--stage1-batch-size", type=int, default=512)
    parser.add_argument("--stage2-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--stage1-weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage2-weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage2-focal-gamma", type=float, default=2.0)
    parser.add_argument("--stage2-focal-alpha", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--cosine-t0", type=int, default=10)
    parser.add_argument("--cosine-t-mult", type=int, default=2)
    parser.add_argument("--cosine-eta-min", type=float, default=0.01)
    parser.add_argument("--restart-decay", type=float, default=0.85)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-padding-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="", help="cpu, cuda, cuda:0, etc. Default: cuda if available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage1_epochs <= 0 or args.stage2_epochs <= 0:
        raise ValueError("--stage1-epochs and --stage2-epochs must be positive")
    configure_fold(args)
    train(args)


if __name__ == "__main__":
    main()
