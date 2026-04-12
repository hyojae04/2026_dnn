from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = (pred - target) ** 2
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def masked_bce_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    extra_state: Dict[str, object] | None = None,
) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "extra_state": extra_state or {},
    }
    torch.save(payload, path_obj)

    with open(path_obj.with_suffix(".json"), "w", encoding="utf-8") as fp:
        json.dump({"epoch": epoch, "metrics": metrics, "extra_state": extra_state or {}}, fp, indent=2)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def prepare_output_dir(output_dir: str, drive_output_dir: str = "") -> Path:
    # 로컬 폴더는 현재 세션용 작업대,
    # Drive 폴더는 전원이 꺼져도 남는 보관함이라고 생각하면 됩니다.
    target = Path(drive_output_dir) if drive_output_dir else Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    return target
