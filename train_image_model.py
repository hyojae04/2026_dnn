from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.skin_coach.config import DEFAULT_IMAGE_TARGETS, parse_columns
from src.skin_coach.data import ImageMultiTaskDataset
from src.skin_coach.models import ImageMultiHeadModel
from src.skin_coach.utils import load_checkpoint, masked_mse_loss, prepare_output_dir, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the image multi-head skin analysis model.")
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, default="")
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--targets", type=str, default=",".join(DEFAULT_IMAGE_TARGETS))
    parser.add_argument("--backbone", type=str, default="efficientnet_b3")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts/image_model")
    parser.add_argument("--drive-output-dir", type=str, default="")
    parser.add_argument("--resume-from", type=str, default="")
    return parser.parse_args()


def run_epoch(
    model: ImageMultiHeadModel,
    loader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
    target_columns: list[str],
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        targets = batch["targets"].to(device)
        target_mask = batch["target_mask"].to(device)

        outputs = model(images)
        preds = torch.stack([outputs["scores"][target] for target in target_columns], dim=1)
        loss = masked_mse_loss(preds, targets, target_mask)

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
    target_columns = parse_columns(args.targets)

    train_csv = args.train_csv
    val_csv = args.val_csv or args.train_csv

    train_dataset = ImageMultiTaskDataset(
        csv_path=train_csv,
        image_root=args.image_root,
        target_columns=target_columns,
        split="train",
        image_size=args.image_size,
        train=True,
    )
    val_dataset = ImageMultiTaskDataset(
        csv_path=val_csv,
        image_root=args.image_root,
        target_columns=target_columns,
        split="val",
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

    model = ImageMultiHeadModel(target_columns=target_columns, backbone_name=args.backbone).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    start_epoch = 1
    output_dir = prepare_output_dir(args.output_dir, args.drive_output_dir)

    if args.resume_from:
        checkpoint = load_checkpoint(args.resume_from, model=model, optimizer=optimizer, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("extra_state", {}).get("best_val_loss", checkpoint.get("metrics", {}).get("val_loss", float("inf"))))
        print(f"Resumed image model from {args.resume_from} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, target_columns)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, None, device, target_columns)
        print(f"[Image][Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        save_checkpoint(
            str(output_dir / "last.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"train_loss": train_loss, "val_loss": val_loss},
            extra_state={
                "target_columns": target_columns,
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
                    "target_columns": target_columns,
                    "backbone": args.backbone,
                    "best_val_loss": best_val,
                },
            )


if __name__ == "__main__":
    main()
