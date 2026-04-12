from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.skin_coach.config import (
    DEFAULT_CAUSE_COLUMNS,
    DEFAULT_IMAGE_TARGETS,
    DEFAULT_RISK_COLUMNS,
    DEFAULT_STATIC_COLUMNS,
    parse_columns,
)
from src.skin_coach.data import MultimodalSkinDataset
from src.skin_coach.models import MultimodalFusionModel
from src.skin_coach.utils import load_checkpoint, masked_bce_loss, masked_mse_loss, prepare_output_dir, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multimodal fusion skin model.")
    parser.add_argument("--multimodal-csv", type=str, required=True)
    parser.add_argument("--daily-logs-csv", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--score-columns", type=str, default=",".join(DEFAULT_IMAGE_TARGETS))
    parser.add_argument("--static-columns", type=str, default=",".join(DEFAULT_STATIC_COLUMNS))
    parser.add_argument("--risk-columns", type=str, default=",".join(DEFAULT_RISK_COLUMNS))
    parser.add_argument("--cause-columns", type=str, default=",".join(DEFAULT_CAUSE_COLUMNS))
    parser.add_argument("--change-columns", type=str, default="skin_score_delta_14d")
    parser.add_argument("--seq-len", type=int, default=14)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--backbone", type=str, default="efficientnet_b3")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts/multimodal_model")
    parser.add_argument("--drive-output-dir", type=str, default="")
    parser.add_argument("--resume-from", type=str, default="")
    return parser.parse_args()


def infer_temporal_input_dim(daily_logs_csv: str) -> int:
    df = pd.read_csv(daily_logs_csv, nrows=2)
    return len([column for column in df.columns if column not in {"user_id", "date"}])


def run_epoch(
    model: MultimodalFusionModel,
    loader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        sequence = batch["sequence"].to(device)
        sequence_mask = batch["sequence_mask"].to(device)
        static_features = batch["static_features"].to(device)
        score_targets = batch["score_targets"].to(device)
        score_mask = batch["score_mask"].to(device)
        risk_targets = batch["risk_targets"].to(device)
        risk_mask = batch["risk_mask"].to(device)
        cause_targets = batch["cause_targets"].to(device)
        cause_mask = batch["cause_mask"].to(device)
        change_targets = batch["change_targets"].to(device)
        change_mask = batch["change_mask"].to(device)

        outputs = model(images, sequence, sequence_mask, static_features)
        loss = torch.tensor(0.0, device=device)
        if "score_pred" in outputs:
            loss = loss + masked_mse_loss(outputs["score_pred"], score_targets, score_mask)
        if "risk_logits" in outputs:
            loss = loss + masked_bce_loss(outputs["risk_logits"], risk_targets, risk_mask)
        if "cause_logits" in outputs:
            loss = loss + masked_bce_loss(outputs["cause_logits"], cause_targets, cause_mask)
        if "change_pred" in outputs and change_targets.numel() > 0:
            loss = loss + masked_mse_loss(outputs["change_pred"], change_targets, change_mask)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    score_columns = parse_columns(args.score_columns)
    static_columns = parse_columns(args.static_columns)
    risk_columns = parse_columns(args.risk_columns)
    cause_columns = parse_columns(args.cause_columns)
    change_columns = parse_columns(args.change_columns)

    train_dataset = MultimodalSkinDataset(
        multimodal_csv=args.multimodal_csv,
        daily_logs_csv=args.daily_logs_csv,
        image_root=args.image_root,
        image_target_columns=score_columns,
        static_columns=static_columns,
        risk_columns=risk_columns,
        cause_columns=cause_columns,
        change_columns=change_columns,
        split="train",
        seq_len=args.seq_len,
        image_size=args.image_size,
        train=True,
    )
    val_dataset = MultimodalSkinDataset(
        multimodal_csv=args.multimodal_csv,
        daily_logs_csv=args.daily_logs_csv,
        image_root=args.image_root,
        image_target_columns=score_columns,
        static_columns=static_columns,
        risk_columns=risk_columns,
        cause_columns=cause_columns,
        change_columns=change_columns,
        split="val",
        seq_len=args.seq_len,
        image_size=args.image_size,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = MultimodalFusionModel(
        image_target_columns=score_columns,
        temporal_input_dim=infer_temporal_input_dim(args.daily_logs_csv),
        static_input_dim=len(static_columns),
        risk_columns=risk_columns,
        cause_columns=cause_columns,
        change_columns=change_columns,
        backbone_name=args.backbone,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    start_epoch = 1
    output_dir = prepare_output_dir(args.output_dir, args.drive_output_dir)

    if args.resume_from:
        checkpoint = load_checkpoint(args.resume_from, model=model, optimizer=optimizer, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("extra_state", {}).get("best_val_loss", checkpoint.get("metrics", {}).get("val_loss", float("inf"))))
        print(f"Resumed multimodal model from {args.resume_from} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, device)
        print(f"[Multimodal][Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        save_checkpoint(
            str(output_dir / "last.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"train_loss": train_loss, "val_loss": val_loss},
            extra_state={
                "score_columns": score_columns,
                "static_columns": static_columns,
                "risk_columns": risk_columns,
                "cause_columns": cause_columns,
                "change_columns": change_columns,
                "backbone": args.backbone,
                "best_val_loss": best_val,
            },
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                str(output_dir / "best.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train_loss": train_loss, "val_loss": val_loss},
                extra_state={
                    "score_columns": score_columns,
                    "static_columns": static_columns,
                    "risk_columns": risk_columns,
                    "cause_columns": cause_columns,
                    "change_columns": change_columns,
                    "backbone": args.backbone,
                    "best_val_loss": best_val,
                },
            )


if __name__ == "__main__":
    main()
