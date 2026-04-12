from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _canonical_split_name(split: str) -> str:
    split = split.lower().strip()
    if split in {"val", "valid", "validation"}:
        return "val"
    if split in {"train", "training"}:
        return "train"
    if split in {"test", "testing"}:
        return "test"
    return split


def build_image_transform(image_size: int = 320, train: bool = True) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def _to_float_tensor(values: List[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def _target_and_mask(row: pd.Series, columns: List[str], scale: float = 1.0) -> Dict[str, torch.Tensor]:
    values: List[float] = []
    mask: List[float] = []
    for column in columns:
        value = row.get(column, np.nan)
        if pd.isna(value):
            values.append(0.0)
            mask.append(0.0)
        else:
            values.append(float(value) / scale)
            mask.append(1.0)
    return {
        "values": _to_float_tensor(values),
        "mask": _to_float_tensor(mask),
    }


@dataclass
class SequenceSlice:
    features: torch.Tensor
    mask: torch.Tensor


class ImageMultiTaskDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        target_columns: List[str],
        split: str,
        image_size: int = 320,
        train: bool = True,
        target_scale: float = 100.0,
    ) -> None:
        self.image_root = Path(image_root)
        self.target_columns = target_columns
        self.target_scale = target_scale
        self.transform = build_image_transform(image_size=image_size, train=train)
        df = pd.read_csv(csv_path)
        df["split"] = df["split"].map(_canonical_split_name)
        self.df = df[df["split"] == _canonical_split_name(split)].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split='{split}' in {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        image_path = self.image_root / str(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        target_pack = _target_and_mask(row, self.target_columns, scale=self.target_scale)
        return {
            "image": self.transform(image),
            "targets": target_pack["values"],
            "target_mask": target_pack["mask"],
        }


class SequenceTargetDataset(Dataset):
    def __init__(
        self,
        daily_logs_csv: str,
        targets_csv: str,
        feature_columns: List[str],
        risk_columns: List[str],
        cause_columns: List[str],
        delta_columns: Optional[List[str]],
        split: str,
        seq_len: int = 14,
        target_scale: float = 100.0,
    ) -> None:
        self.feature_columns = feature_columns
        self.risk_columns = risk_columns
        self.cause_columns = cause_columns
        self.delta_columns = delta_columns or []
        self.seq_len = seq_len
        self.target_scale = target_scale

        targets_df = pd.read_csv(targets_csv)
        targets_df["split"] = targets_df["split"].map(_canonical_split_name)
        targets_df["anchor_date"] = pd.to_datetime(targets_df["anchor_date"])
        self.targets_df = targets_df[targets_df["split"] == _canonical_split_name(split)].reset_index(drop=True)
        if self.targets_df.empty:
            raise ValueError(f"No rows found for split='{split}' in {targets_csv}")

        logs_df = pd.read_csv(daily_logs_csv)
        logs_df["date"] = pd.to_datetime(logs_df["date"])
        self.logs_by_user = {
            user_id: user_df.sort_values("date").reset_index(drop=True)
            for user_id, user_df in logs_df.groupby("user_id")
        }

    def __len__(self) -> int:
        return len(self.targets_df)

    def _build_sequence(self, user_id: str, anchor_date: pd.Timestamp) -> SequenceSlice:
        feature_dim = len(self.feature_columns)
        values = np.zeros((self.seq_len, feature_dim), dtype=np.float32)
        mask = np.zeros((self.seq_len,), dtype=np.float32)

        user_logs = self.logs_by_user.get(user_id)
        if user_logs is None:
            return SequenceSlice(torch.from_numpy(values), torch.from_numpy(mask))

        start_date = anchor_date - timedelta(days=self.seq_len - 1)
        window = user_logs[(user_logs["date"] >= start_date) & (user_logs["date"] <= anchor_date)]
        window = window.tail(self.seq_len)

        offset = self.seq_len - len(window)
        for row_index, (_, log_row) in enumerate(window.iterrows(), start=offset):
            row_values = []
            for column in self.feature_columns:
                value = log_row.get(column, np.nan)
                row_values.append(0.0 if pd.isna(value) else float(value))
            values[row_index] = np.asarray(row_values, dtype=np.float32)
            mask[row_index] = 1.0

        return SequenceSlice(torch.from_numpy(values), torch.from_numpy(mask))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.targets_df.iloc[index]
        sequence = self._build_sequence(str(row["user_id"]), row["anchor_date"])
        risk_pack = _target_and_mask(row, self.risk_columns, scale=1.0)
        cause_pack = _target_and_mask(row, self.cause_columns, scale=1.0)
        delta_pack = _target_and_mask(row, self.delta_columns, scale=self.target_scale)
        return {
            "sequence": sequence.features,
            "sequence_mask": sequence.mask,
            "risk_targets": risk_pack["values"],
            "risk_mask": risk_pack["mask"],
            "cause_targets": cause_pack["values"],
            "cause_mask": cause_pack["mask"],
            "delta_targets": delta_pack["values"],
            "delta_mask": delta_pack["mask"],
        }


class MultimodalSkinDataset(Dataset):
    def __init__(
        self,
        multimodal_csv: str,
        daily_logs_csv: str,
        image_root: str,
        image_target_columns: List[str],
        static_columns: List[str],
        risk_columns: List[str],
        cause_columns: List[str],
        change_columns: Optional[List[str]],
        split: str,
        seq_len: int = 14,
        image_size: int = 320,
        train: bool = True,
        target_scale: float = 100.0,
    ) -> None:
        self.image_root = Path(image_root)
        self.image_target_columns = image_target_columns
        self.static_columns = static_columns
        self.risk_columns = risk_columns
        self.cause_columns = cause_columns
        self.change_columns = change_columns or []
        self.seq_len = seq_len
        self.target_scale = target_scale
        self.transform = build_image_transform(image_size=image_size, train=train)

        df = pd.read_csv(multimodal_csv)
        df["split"] = df["split"].map(_canonical_split_name)
        df["anchor_date"] = pd.to_datetime(df["anchor_date"])
        self.df = df[df["split"] == _canonical_split_name(split)].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split='{split}' in {multimodal_csv}")

        logs_df = pd.read_csv(daily_logs_csv)
        logs_df["date"] = pd.to_datetime(logs_df["date"])
        self.sequence_feature_columns = [column for column in logs_df.columns if column not in {"user_id", "date"}]
        self.logs_by_user = {
            user_id: user_df.sort_values("date").reset_index(drop=True)
            for user_id, user_df in logs_df.groupby("user_id")
        }

    def __len__(self) -> int:
        return len(self.df)

    def _build_sequence(self, user_id: str, anchor_date: pd.Timestamp) -> SequenceSlice:
        feature_dim = len(self.sequence_feature_columns)
        values = np.zeros((self.seq_len, feature_dim), dtype=np.float32)
        mask = np.zeros((self.seq_len,), dtype=np.float32)

        user_logs = self.logs_by_user.get(user_id)
        if user_logs is None:
            return SequenceSlice(torch.from_numpy(values), torch.from_numpy(mask))

        start_date = anchor_date - timedelta(days=self.seq_len - 1)
        window = user_logs[(user_logs["date"] >= start_date) & (user_logs["date"] <= anchor_date)]
        window = window.tail(self.seq_len)
        offset = self.seq_len - len(window)

        for row_index, (_, log_row) in enumerate(window.iterrows(), start=offset):
            row_values = []
            for column in self.sequence_feature_columns:
                value = log_row.get(column, np.nan)
                row_values.append(0.0 if pd.isna(value) else float(value))
            values[row_index] = np.asarray(row_values, dtype=np.float32)
            mask[row_index] = 1.0

        return SequenceSlice(torch.from_numpy(values), torch.from_numpy(mask))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        image = Image.open(self.image_root / str(row["image_path"])).convert("RGB")
        sequence = self._build_sequence(str(row["user_id"]), row["anchor_date"])

        static_values = [
            0.0 if pd.isna(row.get(column, np.nan)) else float(row[column])
            for column in self.static_columns
        ]

        image_pack = _target_and_mask(row, self.image_target_columns, scale=self.target_scale)
        risk_pack = _target_and_mask(row, self.risk_columns, scale=1.0)
        cause_pack = _target_and_mask(row, self.cause_columns, scale=1.0)
        change_pack = _target_and_mask(row, self.change_columns, scale=self.target_scale)

        return {
            "image": self.transform(image),
            "sequence": sequence.features,
            "sequence_mask": sequence.mask,
            "static_features": _to_float_tensor(static_values),
            "score_targets": image_pack["values"],
            "score_mask": image_pack["mask"],
            "risk_targets": risk_pack["values"],
            "risk_mask": risk_pack["mask"],
            "cause_targets": cause_pack["values"],
            "cause_mask": cause_pack["mask"],
            "change_targets": change_pack["values"],
            "change_mask": change_pack["mask"],
        }
