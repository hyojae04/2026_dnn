from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.skin_coach.config import (
    DEFAULT_CAUSE_COLUMNS,
    DEFAULT_RISK_COLUMNS,
    DEFAULT_TEMPORAL_FEATURES,
    parse_columns,
)
from src.skin_coach.data import SequenceTargetDataset
from src.skin_coach.models import TemporalCauseModel
from src.skin_coach.utils import load_checkpoint, masked_bce_loss, masked_mse_loss, prepare_output_dir, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the temporal cause analysis model.")
    parser.add_argument("--daily-logs-csv", type=str, required=True)
    parser.add_argument("--targets-csv", type=str, required=True)
    parser.add_argument("--feature-columns", type=str, default=",".join(DEFAULT_TEMPORAL_FEATURES))
    parser.add_argument("--risk-columns", type=str, default=",".join(DEFAULT_RISK_COLUMNS))
    parser.add_argument("--cause-columns", type=str, default=",".join(DEFAULT_CAUSE_COLUMNS))
    parser.add_argument("--delta-columns", type=str, default="skin_score_delta_14d")
    parser.add_argument("--seq-len", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts/temporal_model")
    parser.add_argument("--drive-output-dir", type=str, default="")
    parser.add_argument("--resume-from", type=str, default="")
    return parser.parse_args()


def run_epoch(
    model: TemporalCauseModel,
    loader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(loader, leave=False):
        sequence = batch["sequence"].to(device)
        sequence_mask = batch["sequence_mask"].to(device)
        risk_targets = batch["risk_targets"].to(device)
        risk_mask = batch["risk_mask"].to(device)
        cause_targets = batch["cause_targets"].to(device)
        cause_mask = batch["cause_mask"].to(device)
        delta_targets = batch["delta_targets"].to(device)
        delta_mask = batch["delta_mask"].to(device)

        outputs = model(sequence, sequence_mask)
        loss = torch.tensor(0.0, device=device)
        if "risk_logits" in outputs:
            loss = loss + masked_bce_loss(outputs["risk_logits"], risk_targets, risk_mask)
        if "cause_logits" in outputs:
            loss = loss + masked_bce_loss(outputs["cause_logits"], cause_targets, cause_mask)
        if "delta_pred" in outputs and delta_targets.numel() > 0:
            loss = loss + masked_mse_loss(outputs["delta_pred"], delta_targets, delta_mask)

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

    feature_columns = parse_columns(args.feature_columns)
    risk_columns = parse_columns(args.risk_columns)
    cause_columns = parse_columns(args.cause_columns)
    delta_columns = parse_columns(args.delta_columns)

    train_dataset = SequenceTargetDataset(
        daily_logs_csv=args.daily_logs_csv,
        targets_csv=args.targets_csv,
        feature_columns=feature_columns,
        risk_columns=risk_columns,
        cause_columns=cause_columns,
        delta_columns=delta_columns,
        split="train",
        seq_len=args.seq_len,
    )
    val_dataset = SequenceTargetDataset(
        daily_logs_csv=args.daily_logs_csv,
        targets_csv=args.targets_csv,
        feature_columns=feature_columns,
        risk_columns=risk_columns,
        cause_columns=cause_columns,
        delta_columns=delta_columns,
        split="val",
        seq_len=args.seq_len,
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

    model = TemporalCauseModel(
        input_dim=len(feature_columns),
        risk_columns=risk_columns,
        cause_columns=cause_columns,
        delta_columns=delta_columns,
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
        print(f"Resumed temporal model from {args.resume_from} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, device)
        print(f"[Temporal][Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        save_checkpoint(
            str(output_dir / "last.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"train_loss": train_loss, "val_loss": val_loss},
            extra_state={
                "feature_columns": feature_columns,
                "risk_columns": risk_columns,
                "cause_columns": cause_columns,
                "delta_columns": delta_columns,
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
                    "feature_columns": feature_columns,
                    "risk_columns": risk_columns,
                    "cause_columns": cause_columns,
                    "delta_columns": delta_columns,
                    "best_val_loss": best_val,
                },
            )


if __name__ == "__main__":
    main()
