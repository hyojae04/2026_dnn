from __future__ import annotations

import json
from pathlib import Path
from typing import List


NOTEBOOK_DIR = Path(".")


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else line + "\n" for line in text.strip("\n").splitlines()],
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else line + "\n" for line in text.strip("\n").splitlines()],
    }


def notebook(cells: List[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
            "colab": {
                "provenance": [],
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: str, cells: List[dict]) -> None:
    target = NOTEBOOK_DIR / path
    target.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=2), encoding="utf-8")


def build_preprocessing_notebook() -> List[dict]:
    return [
        md_cell(
            """
# 01. Data Preprocessing

이 노트북은 여러 피부 데이터셋을 하나의 공통 학습 형식으로 합치는 단계입니다.

비유하면:

- 각 데이터셋은 서로 다른 학교에서 가져온 성적표
- 전처리 코드는 그 성적표를 같은 양식으로 다시 적는 교무실
- 최종 CSV는 모델이 읽는 공통 시험지
"""
        ),
        code_cell(
            """
!pip install -q -r requirements_colab.txt
"""
        ),
        code_cell(
            """
from google.colab import drive

# Google Drive를 연결하면 학습 중간 저장 파일이 세션 종료 후에도 남습니다.
# 비유하면 Colab 임시 책상 말고 개인 사물함에 저장하는 것입니다.
drive.mount("/content/drive")
"""
        ),
        code_cell(
            """
from pathlib import Path

import pandas as pd

from src.skin_coach.preprocessing import (
    DatasetSpec,
    build_multimodal_targets,
    build_temporal_targets,
    integrate_image_datasets,
    standardize_daily_logs,
    standardize_user_profiles,
    write_preprocessed_artifacts,
)

# 프로젝트 폴더와 데이터 폴더를 먼저 정합니다.
# 비유하면 "오늘 수업할 교실"과 "교과서 창고"를 정하는 단계입니다.
PROJECT_ROOT = Path("/content/2026_DNN")
DATA_ROOT = Path("/content/data")
OUTPUT_DIR = PROJECT_ROOT / "processed"
DRIVE_ROOT = Path("/content/drive/MyDrive/2026_DNN")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT, DATA_ROOT, OUTPUT_DIR, DRIVE_ROOT
"""
        ),
        code_cell(
            """
# 여기서는 어떤 데이터셋을 사용할지 목록을 적습니다.
# 비유하면 "어느 학교 성적표를 가져올지" 체크리스트를 만드는 셀입니다.
dataset_specs = [
    DatasetSpec(name="acne04", metadata_csv=str(DATA_ROOT / "acne04_metadata.csv"), image_root=str(DATA_ROOT / "images")),
    DatasetSpec(name="acne04v2", metadata_csv=str(DATA_ROOT / "acne04v2_metadata.csv"), image_root=str(DATA_ROOT / "images")),
    DatasetSpec(name="scin", metadata_csv=str(DATA_ROOT / "scin_metadata.csv"), image_root=str(DATA_ROOT / "images")),
    DatasetSpec(name="ddi", metadata_csv=str(DATA_ROOT / "ddi_metadata.csv"), image_root=str(DATA_ROOT / "images")),
    DatasetSpec(name="fitzpatrick17k", metadata_csv=str(DATA_ROOT / "fitzpatrick_metadata.csv"), image_root=str(DATA_ROOT / "images")),
    DatasetSpec(name="ffhq_wrinkle", metadata_csv=str(DATA_ROOT / "wrinkle_metadata.csv"), image_root=str(DATA_ROOT / "images")),
    DatasetSpec(name="custom", metadata_csv=str(DATA_ROOT / "custom_image_manifest.csv"), image_root=str(DATA_ROOT / "images")),
]

# 실제로 존재하는 CSV만 남깁니다.
dataset_specs = [spec for spec in dataset_specs if Path(spec.metadata_csv).exists()]
dataset_specs
"""
        ),
        code_cell(
            """
# 1단계: 이미지 데이터셋 통합
# 여러 성적표를 한 줄 표로 합치는 과정입니다.
image_labels_df = integrate_image_datasets(dataset_specs)

print("image_labels_df shape:", image_labels_df.shape)
display(image_labels_df.head())
"""
        ),
        code_cell(
            """
# 2단계: 생활습관 로그 정리
# 매일 쓴 일기장을 날짜 순서대로 반듯하게 정리하는 단계입니다.
daily_logs_csv = DATA_ROOT / "daily_logs.csv"
user_profiles_csv = DATA_ROOT / "user_profiles.csv"

daily_logs_df = standardize_daily_logs(str(daily_logs_csv), image_labels_df=image_labels_df)
user_profiles_df = standardize_user_profiles(str(user_profiles_csv) if user_profiles_csv.exists() else "")

print("daily_logs_df shape:", daily_logs_df.shape)
display(daily_logs_df.head())

print("user_profiles_df shape:", user_profiles_df.shape)
display(user_profiles_df.head())
"""
        ),
        code_cell(
            """
# 3단계: 시계열 타깃 생성
# "앞으로 피부가 나빠질까?" 같은 문제를 풀기 위해
# 과거 기록에서 미래 정답표를 만드는 단계입니다.
temporal_targets_df = build_temporal_targets(
    daily_logs_df=daily_logs_df,
    seq_len=14,
    worsening_threshold_points=5.0,
)

print("temporal_targets_df shape:", temporal_targets_df.shape)
display(temporal_targets_df.head())
"""
        ),
        code_cell(
            """
# 4단계: 멀티모달 타깃 생성
# 사진 성적표와 생활습관 일기장을 한 책상 위에 같이 놓는 느낌입니다.
multimodal_targets_df = build_multimodal_targets(
    image_labels_df=image_labels_df,
    daily_logs_df=daily_logs_df,
    temporal_targets_df=temporal_targets_df,
    user_profiles_df=user_profiles_df,
)

print("multimodal_targets_df shape:", multimodal_targets_df.shape)
display(multimodal_targets_df.head())
"""
        ),
        code_cell(
            """
# 5단계: 최종 CSV 저장
# 이제 모델이 읽을 수 있는 교과서 4권을 저장합니다.
artifacts = write_preprocessed_artifacts(
    output_dir=str(OUTPUT_DIR),
    image_labels_df=image_labels_df,
    daily_logs_df=daily_logs_df,
    temporal_targets_df=temporal_targets_df,
    multimodal_targets_df=multimodal_targets_df,
)

artifacts
"""
        ),
        md_cell(
            """
## 출력 파일 설명

- `image_labels.csv`: 이미지 모델이 읽는 성적표
- `daily_logs.csv`: 날짜별 생활습관 로그
- `temporal_targets.csv`: 시계열 모델의 정답표
- `multimodal_targets.csv`: 이미지 + 시계열 + 정적 정보가 합쳐진 최종 표
"""
        ),
    ]


def build_image_notebook() -> List[dict]:
    return [
        md_cell(
            """
# 02. Image Multi-Head Model

이 노트북은 얼굴 사진 한 장을 보고 여러 피부 점수를 동시에 예측하는 모델을 학습합니다.

비유하면:

- 사진 한 장은 시험지 한 장
- 백본 모델은 시험지를 읽는 선생님
- 멀티헤드는 국어/영어/수학처럼 여러 과목을 동시에 채점하는 채점창
"""
        ),
        code_cell("!pip install -q -r requirements_colab.txt"),
        code_cell(
            """
from google.colab import drive
drive.mount("/content/drive")
"""
        ),
        code_cell(
            """
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.skin_coach.config import DEFAULT_IMAGE_TARGETS
from src.skin_coach.data import ImageMultiTaskDataset
from src.skin_coach.models import ImageMultiHeadModel
from src.skin_coach.utils import load_checkpoint, masked_mse_loss, save_checkpoint, seed_everything

seed_everything(42)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
device
"""
        ),
        code_cell(
            """
PROJECT_ROOT = Path("/content/2026_DNN")
DATA_ROOT = PROJECT_ROOT / "processed"
IMAGE_ROOT = Path("/content/data/images")
DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/2026_DNN/checkpoints/image_model")
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMNS = DEFAULT_IMAGE_TARGETS
BATCH_SIZE = 16
IMAGE_SIZE = 320
EPOCHS = 10
LR = 3e-4
OUTPUT_DIR = DRIVE_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 이어 학습하려면 아래 경로를 last.pt 또는 best.pt로 바꾸면 됩니다.
RESUME_FROM = OUTPUT_DIR / "last.pt"
RESUME_FROM = RESUME_FROM if RESUME_FROM.exists() else None
"""
        ),
        code_cell(
            """
# 데이터셋을 불러옵니다.
# 이 단계는 시험지를 반별로 train/val로 나눠 책상에 올려두는 과정과 같습니다.
train_dataset = ImageMultiTaskDataset(
    csv_path=str(DATA_ROOT / "image_labels.csv"),
    image_root=str(IMAGE_ROOT),
    target_columns=TARGET_COLUMNS,
    split="train",
    image_size=IMAGE_SIZE,
    train=True,
)
val_dataset = ImageMultiTaskDataset(
    csv_path=str(DATA_ROOT / "image_labels.csv"),
    image_root=str(IMAGE_ROOT),
    target_columns=TARGET_COLUMNS,
    split="val",
    image_size=IMAGE_SIZE,
    train=False,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

sample = train_dataset[0]
print("image shape:", sample["image"].shape)
print("target shape:", sample["targets"].shape)
"""
        ),
        code_cell(
            """
model = ImageMultiHeadModel(
    target_columns=TARGET_COLUMNS,
    backbone_name="efficientnet_b3",
).to(device)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

best_val = float("inf")
start_epoch = 1
if RESUME_FROM is not None:
    checkpoint = load_checkpoint(str(RESUME_FROM), model=model, optimizer=optimizer, map_location=device)
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val = float(checkpoint.get("extra_state", {}).get("best_val_loss", checkpoint.get("metrics", {}).get("val_loss", float("inf"))))
    print("Resume from:", RESUME_FROM, "start_epoch:", start_epoch)
"""
        ),
        code_cell(
            """
def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        targets = batch["targets"].to(device)
        target_mask = batch["target_mask"].to(device)

        outputs = model(images)

        # 여러 head의 점수를 한 줄로 모읍니다.
        # 비유하면 과목별 점수를 성적표 한 줄에 적는 과정입니다.
        preds = torch.stack([outputs["scores"][target] for target in TARGET_COLUMNS], dim=1)
        loss = masked_mse_loss(preds, targets, target_mask)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)
"""
        ),
        code_cell(
            """
for epoch in range(start_epoch, EPOCHS + 1):
    train_loss = run_epoch(model, train_loader, optimizer)
    with torch.no_grad():
        val_loss = run_epoch(model, val_loader, optimizer=None)

    print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    save_checkpoint(
        path=str(OUTPUT_DIR / "last.pt"),
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics={"train_loss": train_loss, "val_loss": val_loss},
        extra_state={"target_columns": TARGET_COLUMNS, "backbone": "efficientnet_b3", "best_val_loss": best_val},
    )

    if val_loss < best_val:
        best_val = val_loss
        save_checkpoint(
            path=str(OUTPUT_DIR / "best.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"train_loss": train_loss, "val_loss": val_loss},
            extra_state={"target_columns": TARGET_COLUMNS, "backbone": "efficientnet_b3", "best_val_loss": best_val},
        )
"""
        ),
        md_cell(
            """
## 학습이 끝나면

- `best.pt`가 가장 성능이 좋았던 모델입니다.
- 이 모델은 사진만 보고 현재 피부 상태 점수를 예측합니다.
- 다음 노트북에서는 이 점수 흐름을 가지고 원인 분석을 합니다.
"""
        ),
    ]


def build_temporal_notebook() -> List[dict]:
    return [
        md_cell(
            """
# 03. Time-Series Cause Model

이 노트북은 최근 14일 생활습관과 피부 점수 흐름을 보고
앞으로 악화될지, 어떤 원인이 컸는지 추정하는 모델을 학습합니다.

비유하면:

- 하루하루 기록은 탐정 수첩
- GRU는 그 수첩을 시간순으로 읽는 탐정
- attention은 "어느 날이 특히 중요했는지" 형광펜으로 표시하는 기능
"""
        ),
        code_cell("!pip install -q -r requirements_colab.txt"),
        code_cell(
            """
from google.colab import drive
drive.mount("/content/drive")
"""
        ),
        code_cell(
            """
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.skin_coach.config import DEFAULT_CAUSE_COLUMNS, DEFAULT_RISK_COLUMNS, DEFAULT_TEMPORAL_FEATURES
from src.skin_coach.data import SequenceTargetDataset
from src.skin_coach.models import TemporalCauseModel
from src.skin_coach.utils import load_checkpoint, masked_bce_loss, masked_mse_loss, save_checkpoint, seed_everything

seed_everything(42)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
device
"""
        ),
        code_cell(
            """
PROJECT_ROOT = Path("/content/2026_DNN")
DATA_ROOT = PROJECT_ROOT / "processed"
DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/2026_DNN/checkpoints/temporal_model")
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = DEFAULT_TEMPORAL_FEATURES
RISK_COLUMNS = DEFAULT_RISK_COLUMNS
CAUSE_COLUMNS = DEFAULT_CAUSE_COLUMNS
DELTA_COLUMNS = ["skin_score_delta_14d"]
SEQ_LEN = 14
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4
OUTPUT_DIR = DRIVE_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESUME_FROM = OUTPUT_DIR / "last.pt"
RESUME_FROM = RESUME_FROM if RESUME_FROM.exists() else None
"""
        ),
        code_cell(
            """
train_dataset = SequenceTargetDataset(
    daily_logs_csv=str(DATA_ROOT / "daily_logs.csv"),
    targets_csv=str(DATA_ROOT / "temporal_targets.csv"),
    feature_columns=FEATURE_COLUMNS,
    risk_columns=RISK_COLUMNS,
    cause_columns=CAUSE_COLUMNS,
    delta_columns=DELTA_COLUMNS,
    split="train",
    seq_len=SEQ_LEN,
)
val_dataset = SequenceTargetDataset(
    daily_logs_csv=str(DATA_ROOT / "daily_logs.csv"),
    targets_csv=str(DATA_ROOT / "temporal_targets.csv"),
    feature_columns=FEATURE_COLUMNS,
    risk_columns=RISK_COLUMNS,
    cause_columns=CAUSE_COLUMNS,
    delta_columns=DELTA_COLUMNS,
    split="val",
    seq_len=SEQ_LEN,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

sample = train_dataset[0]
print("sequence shape:", sample["sequence"].shape)
print("risk target shape:", sample["risk_targets"].shape)
print("cause target shape:", sample["cause_targets"].shape)
"""
        ),
        code_cell(
            """
model = TemporalCauseModel(
    input_dim=len(FEATURE_COLUMNS),
    risk_columns=RISK_COLUMNS,
    cause_columns=CAUSE_COLUMNS,
    delta_columns=DELTA_COLUMNS,
    hidden_dim=128,
).to(device)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

best_val = float("inf")
start_epoch = 1
if RESUME_FROM is not None:
    checkpoint = load_checkpoint(str(RESUME_FROM), model=model, optimizer=optimizer, map_location=device)
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val = float(checkpoint.get("extra_state", {}).get("best_val_loss", checkpoint.get("metrics", {}).get("val_loss", float("inf"))))
    print("Resume from:", RESUME_FROM, "start_epoch:", start_epoch)
"""
        ),
        code_cell(
            """
def run_epoch(model, loader, optimizer=None):
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

        # 한 모델이 "악화 위험", "원인 기여도", "점수 변화량"을 같이 풀기 때문에
        # 손실도 세 문제의 점수를 더해서 계산합니다.
        loss = torch.tensor(0.0, device=device)
        if "risk_logits" in outputs:
            loss = loss + masked_bce_loss(outputs["risk_logits"], risk_targets, risk_mask)
        if "cause_logits" in outputs:
            loss = loss + masked_bce_loss(outputs["cause_logits"], cause_targets, cause_mask)
        if "delta_pred" in outputs:
            loss = loss + masked_mse_loss(outputs["delta_pred"], delta_targets, delta_mask)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)
"""
        ),
        code_cell(
            """
for epoch in range(start_epoch, EPOCHS + 1):
    train_loss = run_epoch(model, train_loader, optimizer)
    with torch.no_grad():
        val_loss = run_epoch(model, val_loader, optimizer=None)

    print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    save_checkpoint(
        path=str(OUTPUT_DIR / "last.pt"),
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics={"train_loss": train_loss, "val_loss": val_loss},
        extra_state={
            "feature_columns": FEATURE_COLUMNS,
            "risk_columns": RISK_COLUMNS,
            "cause_columns": CAUSE_COLUMNS,
            "delta_columns": DELTA_COLUMNS,
            "best_val_loss": best_val,
        },
    )

    if val_loss < best_val:
        best_val = val_loss
        save_checkpoint(
            path=str(OUTPUT_DIR / "best.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"train_loss": train_loss, "val_loss": val_loss},
            extra_state={
                "feature_columns": FEATURE_COLUMNS,
                "risk_columns": RISK_COLUMNS,
                "cause_columns": CAUSE_COLUMNS,
                "delta_columns": DELTA_COLUMNS,
                "best_val_loss": best_val,
            },
        )
"""
        ),
    ]


def build_multimodal_notebook() -> List[dict]:
    return [
        md_cell(
            """
# 04. Multimodal Fusion Model

이 노트북은 사진 점수와 생활습관 시계열을 함께 보고
피부 상태와 원인 기여도를 동시에 예측하는 최종 모델을 학습합니다.

비유하면:

- 이미지 모델은 사진 채점 선생님
- 시계열 모델은 생활습관 탐정
- 멀티모달 퓨전은 두 사람의 의견을 모아 최종 판정을 쓰는 담임선생님
"""
        ),
        code_cell("!pip install -q -r requirements_colab.txt"),
        code_cell(
            """
from google.colab import drive
drive.mount("/content/drive")
"""
        ),
        code_cell(
            """
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.skin_coach.config import DEFAULT_CAUSE_COLUMNS, DEFAULT_IMAGE_TARGETS, DEFAULT_RISK_COLUMNS, DEFAULT_STATIC_COLUMNS
from src.skin_coach.data import MultimodalSkinDataset
from src.skin_coach.models import MultimodalFusionModel
from src.skin_coach.utils import load_checkpoint, masked_bce_loss, masked_mse_loss, save_checkpoint, seed_everything

seed_everything(42)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
device
"""
        ),
        code_cell(
            """
PROJECT_ROOT = Path("/content/2026_DNN")
DATA_ROOT = PROJECT_ROOT / "processed"
IMAGE_ROOT = Path("/content/data/images")
DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/2026_DNN/checkpoints/multimodal_model")
DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCORE_COLUMNS = DEFAULT_IMAGE_TARGETS
STATIC_COLUMNS = DEFAULT_STATIC_COLUMNS
RISK_COLUMNS = DEFAULT_RISK_COLUMNS
CAUSE_COLUMNS = DEFAULT_CAUSE_COLUMNS
CHANGE_COLUMNS = ["skin_score_delta_14d"]
SEQ_LEN = 14
BATCH_SIZE = 12
EPOCHS = 15
LR = 2e-4
OUTPUT_DIR = DRIVE_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESUME_FROM = OUTPUT_DIR / "last.pt"
RESUME_FROM = RESUME_FROM if RESUME_FROM.exists() else None
"""
        ),
        code_cell(
            """
def infer_temporal_input_dim(csv_path):
    df = pd.read_csv(csv_path, nrows=2)
    return len([column for column in df.columns if column not in {"user_id", "date", "split"}])

train_dataset = MultimodalSkinDataset(
    multimodal_csv=str(DATA_ROOT / "multimodal_targets.csv"),
    daily_logs_csv=str(DATA_ROOT / "daily_logs.csv"),
    image_root=str(IMAGE_ROOT),
    image_target_columns=SCORE_COLUMNS,
    static_columns=STATIC_COLUMNS,
    risk_columns=RISK_COLUMNS,
    cause_columns=CAUSE_COLUMNS,
    change_columns=CHANGE_COLUMNS,
    split="train",
    seq_len=SEQ_LEN,
    image_size=320,
    train=True,
)
val_dataset = MultimodalSkinDataset(
    multimodal_csv=str(DATA_ROOT / "multimodal_targets.csv"),
    daily_logs_csv=str(DATA_ROOT / "daily_logs.csv"),
    image_root=str(IMAGE_ROOT),
    image_target_columns=SCORE_COLUMNS,
    static_columns=STATIC_COLUMNS,
    risk_columns=RISK_COLUMNS,
    cause_columns=CAUSE_COLUMNS,
    change_columns=CHANGE_COLUMNS,
    split="val",
    seq_len=SEQ_LEN,
    image_size=320,
    train=False,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

sample = train_dataset[0]
print("image shape:", sample["image"].shape)
print("sequence shape:", sample["sequence"].shape)
print("static shape:", sample["static_features"].shape)
"""
        ),
        code_cell(
            """
model = MultimodalFusionModel(
    image_target_columns=SCORE_COLUMNS,
    temporal_input_dim=infer_temporal_input_dim(str(DATA_ROOT / "daily_logs.csv")),
    static_input_dim=len(STATIC_COLUMNS),
    risk_columns=RISK_COLUMNS,
    cause_columns=CAUSE_COLUMNS,
    change_columns=CHANGE_COLUMNS,
    backbone_name="efficientnet_b3",
    hidden_dim=128,
).to(device)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

best_val = float("inf")
start_epoch = 1
if RESUME_FROM is not None:
    checkpoint = load_checkpoint(str(RESUME_FROM), model=model, optimizer=optimizer, map_location=device)
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val = float(checkpoint.get("extra_state", {}).get("best_val_loss", checkpoint.get("metrics", {}).get("val_loss", float("inf"))))
    print("Resume from:", RESUME_FROM, "start_epoch:", start_epoch)
"""
        ),
        code_cell(
            """
def run_epoch(model, loader, optimizer=None):
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

        # 이 모델은 "사진 증거"와 "생활습관 증거"를 같이 재판하는 판사 같은 역할입니다.
        # 그래서 점수 예측, 위험 예측, 원인 추정을 모두 더한 총점으로 학습합니다.
        loss = torch.tensor(0.0, device=device)
        if "score_pred" in outputs:
            loss = loss + masked_mse_loss(outputs["score_pred"], score_targets, score_mask)
        if "risk_logits" in outputs:
            loss = loss + masked_bce_loss(outputs["risk_logits"], risk_targets, risk_mask)
        if "cause_logits" in outputs:
            loss = loss + masked_bce_loss(outputs["cause_logits"], cause_targets, cause_mask)
        if "change_pred" in outputs:
            loss = loss + masked_mse_loss(outputs["change_pred"], change_targets, change_mask)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)
"""
        ),
        code_cell(
            """
for epoch in range(start_epoch, EPOCHS + 1):
    train_loss = run_epoch(model, train_loader, optimizer)
    with torch.no_grad():
        val_loss = run_epoch(model, val_loader, optimizer=None)

    print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    save_checkpoint(
        path=str(OUTPUT_DIR / "last.pt"),
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics={"train_loss": train_loss, "val_loss": val_loss},
        extra_state={
            "score_columns": SCORE_COLUMNS,
            "static_columns": STATIC_COLUMNS,
            "risk_columns": RISK_COLUMNS,
            "cause_columns": CAUSE_COLUMNS,
            "change_columns": CHANGE_COLUMNS,
            "best_val_loss": best_val,
        },
    )

    if val_loss < best_val:
        best_val = val_loss
        save_checkpoint(
            path=str(OUTPUT_DIR / "best.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"train_loss": train_loss, "val_loss": val_loss},
            extra_state={
                "score_columns": SCORE_COLUMNS,
                "static_columns": STATIC_COLUMNS,
                "risk_columns": RISK_COLUMNS,
                "cause_columns": CAUSE_COLUMNS,
                "change_columns": CHANGE_COLUMNS,
                "best_val_loss": best_val,
            },
        )
"""
        ),
    ]


def main() -> None:
    write_notebook("01_data_preprocessing.ipynb", build_preprocessing_notebook())
    write_notebook("02_image_multitask_model.ipynb", build_image_notebook())
    write_notebook("03_time_series_cause_model.ipynb", build_temporal_notebook())
    write_notebook("04_multimodal_fusion_model.ipynb", build_multimodal_notebook())
    print("Notebooks generated.")


if __name__ == "__main__":
    main()
