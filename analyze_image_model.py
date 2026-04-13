from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from src.skin_coach.config import DEFAULT_IMAGE_TARGETS
from src.skin_coach.data import ImageMultiTaskDataset
from src.skin_coach.models import ImageMultiHeadModel
from src.skin_coach.utils import load_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the image multi-head model and generate visual reports.")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="reports/image_model_analysis")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_metadata(checkpoint_path: str, device: torch.device) -> tuple[ImageMultiHeadModel, Dict[str, object]]:
    raw = torch.load(checkpoint_path, map_location=device)
    extra_state = raw.get("extra_state", {})
    target_columns = extra_state.get("target_columns", DEFAULT_IMAGE_TARGETS)
    backbone = extra_state.get("backbone", "efficientnet_b3")

    model = ImageMultiHeadModel(target_columns=target_columns, backbone_name=backbone).to(device)
    load_checkpoint(checkpoint_path, model=model, optimizer=None, map_location=device)
    model.eval()
    return model, raw


def evaluate_split(
    model: ImageMultiHeadModel,
    csv_path: str,
    image_root: str,
    split: str,
    target_columns: List[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> pd.DataFrame:
    dataset = ImageMultiTaskDataset(
        csv_path=csv_path,
        image_root=image_root,
        target_columns=target_columns,
        split=split,
        image_size=image_size,
        train=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    all_rows: List[Dict[str, object]] = []
    running_index = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval-{split}", leave=False):
            images = batch["image"].to(device)
            targets = batch["targets"].cpu()
            masks = batch["target_mask"].cpu()
            outputs = model(images)
            preds = torch.stack([outputs["scores"][target] for target in target_columns], dim=1).cpu()

            batch_size_now = preds.shape[0]
            for i in range(batch_size_now):
                row: Dict[str, object] = {
                    "row_index": running_index,
                    "split": split,
                }
                for j, target_name in enumerate(target_columns):
                    row[f"{target_name}_pred"] = float(preds[i, j] * 100.0)
                    row[f"{target_name}_true"] = float(targets[i, j] * 100.0) if masks[i, j] > 0 else None
                    row[f"{target_name}_mask"] = int(masks[i, j].item())
                all_rows.append(row)
                running_index += 1

    return pd.DataFrame(all_rows)


def compute_metrics(pred_df: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
    rows = []
    for target_name in target_columns:
        mask_col = f"{target_name}_mask"
        true_col = f"{target_name}_true"
        pred_col = f"{target_name}_pred"
        target_df = pred_df[pred_df[mask_col] == 1].dropna(subset=[true_col, pred_col]).copy()
        if target_df.empty:
            continue

        y_true = target_df[true_col]
        y_pred = target_df[pred_col]
        rows.append(
            {
                "target": target_name,
                "count": int(len(target_df)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
                "r2": float(r2_score(y_true, y_pred)) if len(target_df) > 1 else None,
                "pred_mean": float(y_pred.mean()),
                "true_mean": float(y_true.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mae")


def plot_dataset_overview(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, y="source_dataset", order=df["source_dataset"].value_counts().index)
    plt.title("Dataset Source Counts")
    plt.xlabel("Count")
    plt.ylabel("Source Dataset")
    plt.tight_layout()
    plt.savefig(output_dir / "source_counts.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="split", order=["train", "val", "test"])
    plt.title("Split Counts")
    plt.tight_layout()
    plt.savefig(output_dir / "split_counts.png", dpi=180)
    plt.close()

    label_availability = df.groupby("source_dataset")[DEFAULT_IMAGE_TARGETS].apply(lambda x: x.notna().mean() * 100.0)
    plt.figure(figsize=(10, 5))
    sns.heatmap(label_availability, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Label Availability by Dataset (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "label_availability_heatmap.png", dpi=180)
    plt.close()


def plot_target_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for ax, target_name in zip(axes, DEFAULT_IMAGE_TARGETS):
        sns.histplot(df[target_name].dropna(), bins=20, ax=ax, kde=True)
        ax.set_title(target_name)
        ax.set_xlabel("Score")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distributions.png", dpi=180)
    plt.close()


def plot_prediction_scatter(pred_df: pd.DataFrame, target_columns: List[str], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for ax, target_name in zip(axes, target_columns):
        target_df = pred_df[pred_df[f"{target_name}_mask"] == 1].dropna(subset=[f"{target_name}_true", f"{target_name}_pred"])
        if target_df.empty:
            ax.set_title(f"{target_name}\n(no labels)")
            ax.axis("off")
            continue
        sns.scatterplot(
            data=target_df.sample(min(len(target_df), 1500), random_state=42),
            x=f"{target_name}_true",
            y=f"{target_name}_pred",
            s=14,
            alpha=0.5,
            ax=ax,
        )
        ax.plot([0, 100], [0, 100], linestyle="--", color="red", linewidth=1)
        ax.set_title(target_name)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_scatter.png", dpi=180)
    plt.close()


def plot_sample_images(df: pd.DataFrame, image_root: str, output_dir: Path) -> None:
    existing_df = df.copy()
    existing_df["abs_path"] = existing_df["image_path"].map(lambda p: str(Path(image_root) / str(p)))
    existing_df["exists"] = existing_df["abs_path"].map(lambda p: Path(p).exists())
    existing_df = existing_df[existing_df["exists"]].sample(min(12, len(existing_df)), random_state=42)
    if existing_df.empty:
        return

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()
    for ax, (_, row) in zip(axes, existing_df.iterrows()):
        image = Image.open(row["abs_path"]).convert("RGB")
        ax.imshow(image)
        ax.set_title(f"{row['source_dataset']}\n{Path(row['image_path']).name[:20]}", fontsize=9)
        ax.axis("off")
    for ax in axes[len(existing_df):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "sample_images.png", dpi=180)
    plt.close()


def save_model_summary(model: ImageMultiHeadModel, image_size: int, output_dir: Path) -> None:
    # 모델 요약은 "건물의 층별 안내도"처럼 구조를 빠르게 파악하게 해줍니다.
    model_summary = summary(model, input_size=(1, 3, image_size, image_size), depth=3, verbose=0)
    with open(output_dir / "model_summary.txt", "w", encoding="utf-8") as fp:
        fp.write(str(model))
        fp.write("\n\n")
        fp.write(str(model_summary))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    raw_df = pd.read_csv(args.csv_path)
    plot_dataset_overview(raw_df, output_dir)
    plot_target_distributions(raw_df, output_dir)
    plot_sample_images(raw_df, args.image_root, output_dir)

    model, checkpoint = load_model_and_metadata(args.checkpoint_path, device)
    extra_state = checkpoint.get("extra_state", {})
    target_columns = extra_state.get("target_columns", DEFAULT_IMAGE_TARGETS)

    save_model_summary(model, args.image_size, output_dir)

    split_values = set(raw_df["split"].astype(str).str.lower().unique())
    eval_splits = [split for split in ["val", "test", "train"] if split in split_values]
    all_predictions = []
    all_metrics = []

    for split in eval_splits:
        pred_df = evaluate_split(
            model=model,
            csv_path=args.csv_path,
            image_root=args.image_root,
            split=split,
            target_columns=target_columns,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        pred_df.to_csv(output_dir / f"predictions_{split}.csv", index=False)
        metrics_df = compute_metrics(pred_df, target_columns)
        metrics_df["split"] = split
        metrics_df.to_csv(output_dir / f"metrics_{split}.csv", index=False)
        all_predictions.append(pred_df)
        all_metrics.append(metrics_df)

    if all_predictions:
        merged_pred = pd.concat(all_predictions, ignore_index=True)
        plot_prediction_scatter(merged_pred, target_columns, output_dir)

    if all_metrics:
        merged_metrics = pd.concat(all_metrics, ignore_index=True)
        merged_metrics.to_csv(output_dir / "metrics_all_splits.csv", index=False)

    report = {
        "checkpoint_path": args.checkpoint_path,
        "csv_path": args.csv_path,
        "image_root": args.image_root,
        "device": str(device),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
        "checkpoint_extra_state": checkpoint.get("extra_state", {}),
        "row_count": int(len(raw_df)),
        "split_counts": raw_df["split"].value_counts(dropna=False).to_dict(),
        "source_counts": raw_df["source_dataset"].value_counts(dropna=False).to_dict() if "source_dataset" in raw_df.columns else {},
    }
    with open(output_dir / "analysis_report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    print(f"Analysis artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
