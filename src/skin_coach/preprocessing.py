from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_CAUSE_COLUMNS,
    DEFAULT_IMAGE_TARGETS,
    DEFAULT_STATIC_COLUMNS,
    DEFAULT_TEMPORAL_FEATURES,
    DEFAULT_TEMPORAL_SCORE_COLUMNS,
)


IMAGE_PATH_CANDIDATES = ["image_path", "filepath", "file_path", "path", "image", "img_path", "file_name", "filename"]
USER_ID_CANDIDATES = ["user_id", "subject_id", "patient_id", "case_id", "participant_id"]
DATE_CANDIDATES = ["capture_date", "date", "taken_at", "image_date", "anchor_date"]
SPLIT_CANDIDATES = ["split", "fold", "subset"]

IMAGE_SCORE_ALIASES: Dict[str, List[str]] = {
    "acne_score": ["acne_score", "acne_health_score", "acne_grade", "acne_severity", "severity_grade", "global_grade"],
    "redness_score": ["redness_score", "erythema_score", "erythema_severity", "redness_grade"],
    "dryness_score": ["dryness_score", "dry_score", "xerosis_score", "dryness_grade"],
    "wrinkle_score": ["wrinkle_score", "wrinkle_grade", "wrinkle_severity"],
    "pigmentation_score": ["pigmentation_score", "pigment_score", "hyperpigmentation_score", "spot_score"],
    "oiliness_score": ["oiliness_score", "sebum_score", "oil_score", "greasiness_score"],
    "pore_score": ["pore_score", "pores_score", "pore_grade", "pore_visibility_score"],
    "overall_skin_score": ["overall_skin_score", "skin_score", "global_skin_score"],
}

HIGHER_IS_WORSE_COLUMNS = {
    "acne_score",
    "redness_score",
    "dryness_score",
    "wrinkle_score",
    "pigmentation_score",
    "oiliness_score",
    "pore_score",
}

DAILY_LOG_ALIASES: Dict[str, List[str]] = {
    "sleep_hours": ["sleep_hours", "sleep_time", "sleep_duration"],
    "water_liters": ["water_liters", "water_intake_l", "water_ml"],
    "sugar_score": ["sugar_score", "sugar_intake", "sweets_score"],
    "dairy_score": ["dairy_score", "dairy_intake"],
    "spicy_food_score": ["spicy_food_score", "spicy_score"],
    "exercise_minutes": ["exercise_minutes", "workout_minutes"],
    "stress_score": ["stress_score", "stress_level"],
    "mask_hours": ["mask_hours", "mask_time"],
    "uv_exposure_hours": ["uv_exposure_hours", "sun_hours", "sun_exposure_hours"],
    "cleanser_use": ["cleanser_use", "washed_face", "cleanse_used"],
    "moisturizer_use": ["moisturizer_use", "moisturizer_used"],
    "product_changed": ["product_changed", "new_product", "routine_changed"],
    "temperature_c": ["temperature_c", "temp_c", "temperature"],
    "humidity": ["humidity", "humidity_pct"],
    "pm25": ["pm25", "fine_dust", "particulate_25"],
}

STATIC_ALIASES: Dict[str, List[str]] = {
    "age_scaled": ["age_scaled", "age_norm"],
    "sex_encoded": ["sex_encoded", "gender_encoded"],
    "family_history": ["family_history", "family_history_acne"],
    "sensitive_skin": ["sensitive_skin", "is_sensitive_skin"],
    "baseline_skin_type_encoded": ["baseline_skin_type_encoded", "skin_type_encoded"],
}


@dataclass
class DatasetSpec:
    name: str
    metadata_csv: str
    image_root: str


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _canonical_split_name(value: object) -> str:
    raw = str(value).lower().strip()
    if raw in {"train", "training"}:
        return "train"
    if raw in {"val", "valid", "validation"}:
        return "val"
    if raw in {"test", "testing"}:
        return "test"
    return raw


def _normalize_health_series(series: pd.Series, higher_is_worse: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return numeric

    finite = numeric.dropna()
    min_value = float(finite.min())
    max_value = float(finite.max())
    if max_value - min_value < 1e-8:
        normalized = pd.Series(np.where(numeric.notna(), 100.0, np.nan), index=series.index)
    else:
        normalized = (numeric - min_value) / (max_value - min_value) * 100.0

    if higher_is_worse:
        normalized = 100.0 - normalized
    return normalized.clip(0.0, 100.0)


def _coerce_binary(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return (numeric > 0).astype(float)


def _hash_bucket(text: str, seed: int = 42) -> float:
    digest = hashlib.md5(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _score_template_row() -> Dict[str, float]:
    return {column: np.nan for column in DEFAULT_IMAGE_TARGETS}


def _weak_label_from_condition_text(condition_text: object) -> Dict[str, float]:
    # 질환 이름을 우리 프로젝트의 피부 점수표로 번역하는 "간이 통역사"입니다.
    # 정확한 정답 라벨이 부족한 데이터셋에서 약한 힌트를 주는 용도로 사용합니다.
    text = str(condition_text).strip().lower()
    row = _score_template_row()
    if not text or text == "nan":
        return row

    if any(keyword in text for keyword in ["acne", "pimple", "comedone", "papule", "pustule"]):
        row["acne_score"] = 20.0
        row["redness_score"] = 45.0
        row["overall_skin_score"] = 35.0
    if any(keyword in text for keyword in ["rosacea", "erythema", "redness", "inflamed", "inflammation"]):
        row["redness_score"] = 20.0 if pd.isna(row["redness_score"]) else min(row["redness_score"], 20.0)
        row["overall_skin_score"] = 40.0 if pd.isna(row["overall_skin_score"]) else min(row["overall_skin_score"], 40.0)
    if any(keyword in text for keyword in ["eczema", "xerosis", "dry", "dermatitis", "atopic"]):
        row["dryness_score"] = 25.0
        row["redness_score"] = 40.0 if pd.isna(row["redness_score"]) else min(row["redness_score"], 40.0)
        row["overall_skin_score"] = 42.0 if pd.isna(row["overall_skin_score"]) else min(row["overall_skin_score"], 42.0)
    if any(keyword in text for keyword in ["melasma", "hyperpig", "pigment", "lentigo", "post-inflammatory"]):
        row["pigmentation_score"] = 20.0
        row["overall_skin_score"] = 48.0 if pd.isna(row["overall_skin_score"]) else min(row["overall_skin_score"], 48.0)
    if any(keyword in text for keyword in ["wrinkle", "rhytid", "aging", "photoaging"]):
        row["wrinkle_score"] = 25.0
        row["overall_skin_score"] = 55.0 if pd.isna(row["overall_skin_score"]) else min(row["overall_skin_score"], 55.0)
    if any(keyword in text for keyword in ["seborrhea", "oily", "oil", "sebum"]):
        row["oiliness_score"] = 30.0
        row["overall_skin_score"] = 55.0 if pd.isna(row["overall_skin_score"]) else min(row["overall_skin_score"], 55.0)
    if any(keyword in text for keyword in ["pore", "pores"]):
        row["pore_score"] = 35.0
        row["overall_skin_score"] = 60.0 if pd.isna(row["overall_skin_score"]) else min(row["overall_skin_score"], 60.0)
    return row


def _merge_score_columns(base_df: pd.DataFrame, score_dicts: List[Dict[str, float]], label_quality: str) -> pd.DataFrame:
    scores_df = pd.DataFrame(score_dicts)
    for column in DEFAULT_IMAGE_TARGETS:
        if column not in scores_df.columns:
            scores_df[column] = np.nan
        if column not in base_df.columns:
            base_df[column] = np.nan
        base_df[column] = base_df[column].fillna(scores_df[column])
    if "label_quality" not in base_df.columns:
        base_df["label_quality"] = label_quality
    else:
        base_df["label_quality"] = base_df["label_quality"].fillna(label_quality)
    return base_df


def _finalize_manifest(df: pd.DataFrame, keep_unlabeled: bool = True) -> pd.DataFrame:
    if df.empty:
        return df
    if "overall_skin_score" not in df.columns:
        df["overall_skin_score"] = np.nan
    per_item_scores = [column for column in DEFAULT_IMAGE_TARGETS if column != "overall_skin_score"]
    for column in DEFAULT_IMAGE_TARGETS:
        if column not in df.columns:
            df[column] = np.nan
    df["overall_skin_score"] = df["overall_skin_score"].fillna(df[per_item_scores].mean(axis=1))
    df["has_any_label"] = df[DEFAULT_IMAGE_TARGETS].notna().any(axis=1).astype(int)
    if not keep_unlabeled:
        df = df[df["has_any_label"] == 1].copy()
    return df.reset_index(drop=True)


def assign_group_splits(
    df: pd.DataFrame,
    group_column: str = "user_id",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.Series:
    # 같은 사용자를 train/val/test로 찢으면 답안을 미리 본 것과 비슷한 누수가 생깁니다.
    # 그래서 "같은 사람은 같은 반에만 배정"하는 규칙으로 split을 나눕니다.
    assignments: Dict[str, str] = {}
    for group_value in df[group_column].fillna("unknown").astype(str).unique():
        bucket = _hash_bucket(group_value, seed=seed)
        if bucket < test_ratio:
            assignments[group_value] = "test"
        elif bucket < test_ratio + val_ratio:
            assignments[group_value] = "val"
        else:
            assignments[group_value] = "train"
    return df[group_column].fillna("unknown").astype(str).map(assignments)


def _build_generic_manifest(
    dataset_name: str,
    df: pd.DataFrame,
    image_root: str,
    label_quality: str = "strong",
    keep_unlabeled: bool = True,
) -> pd.DataFrame:
    result = pd.DataFrame()
    image_col = _first_existing_column(df, IMAGE_PATH_CANDIDATES)
    if image_col is None:
        raise ValueError(f"{dataset_name}: image path column not found")

    user_col = _first_existing_column(df, USER_ID_CANDIDATES)
    date_col = _first_existing_column(df, DATE_CANDIDATES)
    split_col = _first_existing_column(df, SPLIT_CANDIDATES)

    result["image_path"] = df[image_col].astype(str)
    result["source_dataset"] = dataset_name
    result["source_image_root"] = image_root

    if user_col:
        result["user_id"] = df[user_col].astype(str)
    else:
        result["user_id"] = result["image_path"].map(lambda p: Path(p).parts[0] if len(Path(p).parts) > 1 else Path(p).stem)

    if date_col:
        result["capture_date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        result["capture_date"] = pd.NaT

    if split_col:
        result["split"] = df[split_col].map(_canonical_split_name)
    else:
        result["split"] = assign_group_splits(result, group_column="user_id")

    for score_column in DEFAULT_IMAGE_TARGETS:
        raw_col = _first_existing_column(df, IMAGE_SCORE_ALIASES.get(score_column, [score_column]))
        if raw_col is None:
            result[score_column] = np.nan
            continue
        higher_is_worse = score_column in HIGHER_IS_WORSE_COLUMNS
        result[score_column] = _normalize_health_series(df[raw_col], higher_is_worse=higher_is_worse)

    result["label_quality"] = label_quality
    return _finalize_manifest(result, keep_unlabeled=keep_unlabeled)


def standardize_acne04_metadata(metadata_csv: str, image_root: str, dataset_name: str = "acne04") -> pd.DataFrame:
    path = Path(metadata_csv)
    if path.exists() and path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        result = _build_generic_manifest(dataset_name, df, image_root=image_root, label_quality="strong", keep_unlabeled=False)
        count_col = _first_existing_column(df, ["lesion_count", "count", "acne_count", "num_lesions"])
        if count_col is not None:
            count_score = _normalize_health_series(df[count_col], higher_is_worse=True)
            result["acne_score"] = result["acne_score"].fillna(count_score)
            result["overall_skin_score"] = result["overall_skin_score"].fillna(count_score)
        return _finalize_manifest(result, keep_unlabeled=False)

    image_root_path = Path(image_root)
    image_paths = list(image_root_path.rglob("*.jpg")) + list(image_root_path.rglob("*.png")) + list(image_root_path.rglob("*.jpeg"))
    grade_map = {"acne0": 95.0, "acne1": 75.0, "acne2": 55.0, "acne3": 30.0}
    rows: List[Dict[str, object]] = []
    for image_path in image_paths:
        folder_name = image_path.parts[-2].lower() if len(image_path.parts) >= 2 else ""
        grade_score = next((score for prefix, score in grade_map.items() if folder_name.startswith(prefix)), np.nan)
        rows.append(
            {
                "image_path": str(image_path.relative_to(image_root_path)),
                "source_dataset": dataset_name,
                "source_image_root": image_root,
                "user_id": image_path.stem,
                "capture_date": pd.NaT,
                "split": "train",
                "acne_score": grade_score,
                "overall_skin_score": grade_score,
                "label_quality": "strong",
            }
        )
    return _finalize_manifest(pd.DataFrame(rows), keep_unlabeled=False)


def standardize_acne04v2_metadata(annotation_path: str, image_root: str, dataset_name: str = "acne04v2") -> pd.DataFrame:
    path = Path(annotation_path)
    if path.exists() and path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)

        images_df = pd.DataFrame(payload.get("images", []))
        anns_df = pd.DataFrame(payload.get("annotations", []))
        if images_df.empty:
            return pd.DataFrame(columns=["image_path", "user_id", "capture_date", "split", "source_dataset", "source_image_root", *DEFAULT_IMAGE_TARGETS])

        lesion_counts = anns_df.groupby("image_id").size() if not anns_df.empty else pd.Series(dtype=float)
        result = pd.DataFrame()
        result["image_path"] = images_df["file_name"].astype(str)
        result["source_dataset"] = dataset_name
        result["source_image_root"] = image_root
        result["user_id"] = images_df["file_name"].astype(str).map(lambda p: Path(p).stem)
        result["capture_date"] = pd.NaT
        result["split"] = assign_group_splits(result, group_column="user_id")
        result["lesion_count"] = images_df["id"].map(lesion_counts).fillna(0).astype(float)
        result["acne_score"] = _normalize_health_series(result["lesion_count"], higher_is_worse=True)
        result["overall_skin_score"] = result["acne_score"]
        result["label_quality"] = "strong"
        return _finalize_manifest(result, keep_unlabeled=False)

    return standardize_acne04_metadata(annotation_path, image_root=image_root, dataset_name=dataset_name)


def standardize_scin_metadata(metadata_csv: str, image_root: str, dataset_name: str = "scin") -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    result = _build_generic_manifest(dataset_name, df, image_root=image_root, label_quality="weak", keep_unlabeled=True)
    condition_col = _first_existing_column(
        df,
        [
            "dermatologist_skin_condition_on_label_name",
            "skin_condition_label",
            "condition_label",
            "dermatologist_condition_label",
            "label_name",
            "condition",
        ],
    )
    if condition_col is not None:
        result = _merge_score_columns(result, [_weak_label_from_condition_text(value) for value in df[condition_col]], label_quality="weak")
    return _finalize_manifest(result, keep_unlabeled=True)


def standardize_ddi_metadata(metadata_csv: str, image_root: str, dataset_name: str = "ddi") -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    result = _build_generic_manifest(dataset_name, df, image_root=image_root, label_quality="weak", keep_unlabeled=True)

    condition_col = _first_existing_column(df, ["disease", "disease_label", "label", "diagnosis", "class_name", "malignant_benign"])
    if condition_col is not None:
        result = _merge_score_columns(result, [_weak_label_from_condition_text(value) for value in df[condition_col]], label_quality="weak")

    image_col = _first_existing_column(df, IMAGE_PATH_CANDIDATES)
    if image_col is None:
        file_col = _first_existing_column(df, ["DDI_file", "file", "image_id"])
        if file_col is not None:
            result["image_path"] = df[file_col].astype(str).map(lambda name: f"images/{name}")

    return _finalize_manifest(result, keep_unlabeled=True)


def standardize_fitzpatrick17k_metadata(metadata_csv: str, image_root: str, dataset_name: str = "fitzpatrick17k") -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    result = _build_generic_manifest(dataset_name, df, image_root=image_root, label_quality="weak", keep_unlabeled=True)

    condition_col = _first_existing_column(df, ["label", "three_partition_label", "nine_partition_label", "disease", "diagnosis"])
    if condition_col is not None:
        result = _merge_score_columns(result, [_weak_label_from_condition_text(value) for value in df[condition_col]], label_quality="weak")

    image_col = _first_existing_column(df, IMAGE_PATH_CANDIDATES)
    if image_col is None:
        md5_col = _first_existing_column(df, ["md5hash", "md5"])
        url_col = _first_existing_column(df, ["url", "image_url"])
        if md5_col is not None:
            result["image_path"] = df[md5_col].astype(str) + ".jpg"
        elif url_col is not None:
            result["image_path"] = df[url_col].astype(str).map(lambda url: Path(str(url)).name)

    return _finalize_manifest(result, keep_unlabeled=True)


def standardize_ffhq_wrinkle_metadata(metadata_csv: str, image_root: str, dataset_name: str = "ffhq_wrinkle") -> pd.DataFrame:
    path = Path(metadata_csv)
    if path.exists() and path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        result = _build_generic_manifest(dataset_name, df, image_root=image_root, label_quality="strong", keep_unlabeled=False)
        mask_col = _first_existing_column(df, ["wrinkle_mask_ratio", "wrinkle_area_ratio", "mask_coverage"])
        if mask_col is not None:
            result["wrinkle_score"] = _normalize_health_series(df[mask_col], higher_is_worse=True)
            result["overall_skin_score"] = result["overall_skin_score"].fillna(result["wrinkle_score"])
        return _finalize_manifest(result, keep_unlabeled=False)

    base = Path(image_root)
    face_image_dir = base / "masked_face_images"
    image_base = base / "images1024x1024"
    manual_mask_dir = base / "manual_wrinkle_masks"
    mask_paths = list(manual_mask_dir.rglob("*.png")) if manual_mask_dir.exists() else []
    rows: List[Dict[str, object]] = []
    ratios: List[float] = []

    for mask_path in mask_paths:
        stem = mask_path.stem
        face_image = face_image_dir / f"{stem}.png"
        if not face_image.exists():
            bucket_dir = f"{int(stem) // 1000 * 1000:05d}" if stem.isdigit() else stem[:5]
            candidate = image_base / bucket_dir / f"{stem}.png"
            if candidate.exists():
                face_image = candidate

        image_path = str(face_image.relative_to(base)) if face_image.exists() else str(mask_path.relative_to(base))
        rows.append(
            {
                "image_path": image_path,
                "user_id": stem,
                "capture_date": pd.NaT,
                "split": "val" if _hash_bucket(stem) < 0.15 else "train",
                "source_dataset": dataset_name,
                "source_image_root": image_root,
                "label_quality": "strong",
            }
        )

        try:
            from PIL import Image

            mask_img = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
            ratios.append(float((mask_img > 0.1).mean()))
        except Exception:
            ratios.append(np.nan)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["wrinkle_area_ratio"] = ratios
    result["wrinkle_score"] = _normalize_health_series(pd.Series(ratios), higher_is_worse=True)
    result["overall_skin_score"] = result["wrinkle_score"]
    return _finalize_manifest(result, keep_unlabeled=False)


def standardize_image_manifest(dataset_name: str, metadata_csv: str, image_root: str) -> pd.DataFrame:
    dataset_key = dataset_name.lower().strip()
    if dataset_key == "acne04":
        return standardize_acne04_metadata(metadata_csv, image_root=image_root, dataset_name=dataset_name)
    if dataset_key == "acne04v2":
        return standardize_acne04v2_metadata(metadata_csv, image_root=image_root, dataset_name=dataset_name)
    if dataset_key == "scin":
        return standardize_scin_metadata(metadata_csv, image_root=image_root, dataset_name=dataset_name)
    if dataset_key == "ddi":
        return standardize_ddi_metadata(metadata_csv, image_root=image_root, dataset_name=dataset_name)
    if dataset_key == "fitzpatrick17k":
        return standardize_fitzpatrick17k_metadata(metadata_csv, image_root=image_root, dataset_name=dataset_name)
    if dataset_key == "ffhq_wrinkle":
        return standardize_ffhq_wrinkle_metadata(metadata_csv, image_root=image_root, dataset_name=dataset_name)

    df = pd.read_csv(metadata_csv)
    return _build_generic_manifest(dataset_name, df, image_root=image_root, label_quality="strong", keep_unlabeled=True)


def integrate_image_datasets(dataset_specs: Sequence[DatasetSpec]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for spec in dataset_specs:
        if not spec.metadata_csv and spec.name.lower() != "acne04":
            continue
        frames.append(standardize_image_manifest(spec.name, spec.metadata_csv, spec.image_root))
    if not frames:
        return pd.DataFrame(columns=["image_path", "user_id", "capture_date", "split", "source_dataset", "source_image_root", *DEFAULT_IMAGE_TARGETS])
    combined = pd.concat(frames, ignore_index=True)
    combined["capture_date"] = pd.to_datetime(combined["capture_date"], errors="coerce")
    return combined


def standardize_user_profiles(user_profiles_csv: Optional[str]) -> pd.DataFrame:
    if not user_profiles_csv:
        return pd.DataFrame(columns=["user_id", *DEFAULT_STATIC_COLUMNS])

    df = pd.read_csv(user_profiles_csv)
    user_col = _first_existing_column(df, USER_ID_CANDIDATES)
    if user_col is None:
        raise ValueError("user_profiles_csv must include a user id column")

    result = pd.DataFrame()
    result["user_id"] = df[user_col].astype(str)

    for target_col in DEFAULT_STATIC_COLUMNS:
        raw_col = _first_existing_column(df, STATIC_ALIASES.get(target_col, [target_col]))
        if raw_col is None:
            result[target_col] = 0.0
            continue
        series = pd.to_numeric(df[raw_col], errors="coerce")
        if target_col == "age_scaled":
            finite = series.dropna()
            if finite.empty:
                result[target_col] = 0.0
            else:
                min_value = float(finite.min())
                max_value = float(finite.max())
                if max_value - min_value < 1e-8:
                    result[target_col] = 0.0
                else:
                    result[target_col] = ((series - min_value) / (max_value - min_value)).fillna(0.0)
        else:
            result[target_col] = series.fillna(0.0)

    return result.drop_duplicates(subset=["user_id"]).reset_index(drop=True)


def standardize_daily_logs(daily_logs_csv: str, image_labels_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = pd.read_csv(daily_logs_csv)
    user_col = _first_existing_column(df, USER_ID_CANDIDATES)
    date_col = _first_existing_column(df, DATE_CANDIDATES)
    split_col = _first_existing_column(df, SPLIT_CANDIDATES)
    if user_col is None or date_col is None:
        raise ValueError("daily_logs_csv must include user_id and date columns")

    result = pd.DataFrame()
    result["user_id"] = df[user_col].astype(str)
    result["date"] = pd.to_datetime(df[date_col], errors="coerce")

    if split_col:
        result["split"] = df[split_col].map(_canonical_split_name)
    else:
        result["split"] = assign_group_splits(result, group_column="user_id")

    for feature_column in DEFAULT_TEMPORAL_FEATURES:
        raw_col = _first_existing_column(df, DAILY_LOG_ALIASES.get(feature_column, [feature_column]))
        if raw_col is None:
            result[feature_column] = 0.0
            continue
        raw_series = df[raw_col]
        if feature_column in {"cleanser_use", "moisturizer_use", "product_changed"}:
            result[feature_column] = _coerce_binary(raw_series)
        else:
            numeric = pd.to_numeric(raw_series, errors="coerce").fillna(0.0)
            if feature_column == "water_liters" and numeric.max() > 20:
                numeric = numeric / 1000.0
            result[feature_column] = numeric

    for score_column in DEFAULT_TEMPORAL_SCORE_COLUMNS:
        raw_col = _first_existing_column(df, IMAGE_SCORE_ALIASES.get(score_column, [score_column]))
        if raw_col is None:
            result[score_column] = np.nan
            continue
        numeric = pd.to_numeric(df[raw_col], errors="coerce")
        if numeric.dropna().empty:
            result[score_column] = np.nan
        else:
            result[score_column] = numeric.clip(0.0, 100.0)

    if image_labels_df is not None and not image_labels_df.empty:
        image_daily = image_labels_df.copy()
        image_daily["date"] = pd.to_datetime(image_daily["capture_date"], errors="coerce")
        image_daily = image_daily.dropna(subset=["date"])
        image_daily = image_daily.groupby(["user_id", "date"], as_index=False)[DEFAULT_TEMPORAL_SCORE_COLUMNS].mean()
        result = result.merge(image_daily, on=["user_id", "date"], how="left", suffixes=("", "_from_image"))
        for score_column in DEFAULT_TEMPORAL_SCORE_COLUMNS:
            image_col = f"{score_column}_from_image"
            if image_col in result.columns:
                result[score_column] = result[score_column].fillna(result[image_col])
                result = result.drop(columns=[image_col])

    result["overall_skin_score"] = result["overall_skin_score"].fillna(
        result[[column for column in DEFAULT_TEMPORAL_SCORE_COLUMNS if column != "overall_skin_score"]].mean(axis=1)
    )
    return result.sort_values(["user_id", "date"]).reset_index(drop=True)


def _future_value_lookup(user_df: pd.DataFrame, anchor_date: pd.Timestamp, horizon_days: int, tolerance_days: int = 3) -> float:
    target_date = anchor_date + pd.Timedelta(days=horizon_days)
    candidates = user_df[(user_df["date"] >= target_date) & (user_df["date"] <= target_date + pd.Timedelta(days=tolerance_days))]
    if candidates.empty:
        return np.nan
    return float(candidates.iloc[0]["overall_skin_score"])


def _recent_window(user_df: pd.DataFrame, anchor_date: pd.Timestamp, days: int = 7) -> pd.DataFrame:
    start = anchor_date - pd.Timedelta(days=days - 1)
    return user_df[(user_df["date"] >= start) & (user_df["date"] <= anchor_date)]


def _weak_cause_scores(window_df: pd.DataFrame) -> Dict[str, float]:
    # 정답 원인 라벨이 없을 때 쓰는 약한 힌트입니다.
    # 완벽한 진실이라기보다, 초반 탐정에게 주는 참고 메모라고 보면 됩니다.
    if window_df.empty:
        return {column: 0.0 for column in DEFAULT_CAUSE_COLUMNS}

    sleep_mean = window_df["sleep_hours"].mean()
    stress_mean = window_df["stress_score"].mean()
    product_recent = window_df["product_changed"].max()
    uv_mean = window_df["uv_exposure_hours"].mean()
    diet_mean = window_df[["sugar_score", "dairy_score", "spicy_food_score"]].mean(axis=1).mean()
    mask_mean = window_df["mask_hours"].mean()

    return {
        "cause_sleep": float(np.clip((7.0 - sleep_mean) / 4.0, 0.0, 1.0)),
        "cause_stress": float(np.clip(stress_mean / 10.0, 0.0, 1.0)),
        "cause_product_change": float(np.clip(product_recent, 0.0, 1.0)),
        "cause_uv": float(np.clip(uv_mean / 4.0, 0.0, 1.0)),
        "cause_diet": float(np.clip(diet_mean / 5.0, 0.0, 1.0)),
        "cause_mask": float(np.clip(mask_mean / 10.0, 0.0, 1.0)),
    }


def build_temporal_targets(
    daily_logs_df: pd.DataFrame,
    seq_len: int = 14,
    worsening_threshold_points: float = 5.0,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for user_id, user_df in daily_logs_df.groupby("user_id"):
        user_df = user_df.sort_values("date").reset_index(drop=True)
        for _, row in user_df.iterrows():
            anchor_date = row["date"]
            if pd.isna(anchor_date) or pd.isna(row["overall_skin_score"]):
                continue

            history_window = _recent_window(user_df, anchor_date, days=seq_len)
            if len(history_window) < max(3, seq_len // 2):
                continue

            future_7 = _future_value_lookup(user_df, anchor_date, horizon_days=7)
            future_14 = _future_value_lookup(user_df, anchor_date, horizon_days=14)
            if np.isnan(future_14):
                continue

            delta_14 = float(future_14 - row["overall_skin_score"])
            weak_causes = _weak_cause_scores(history_window)
            rows.append(
                {
                    "user_id": user_id,
                    "anchor_date": anchor_date,
                    "split": row["split"],
                    "future_worsening_7d": np.nan if np.isnan(future_7) else float(future_7 <= row["overall_skin_score"] - worsening_threshold_points),
                    "future_worsening_14d": float(future_14 <= row["overall_skin_score"] - worsening_threshold_points),
                    "skin_score_delta_14d": delta_14,
                    **weak_causes,
                }
            )
    return pd.DataFrame(rows)


def build_multimodal_targets(
    image_labels_df: pd.DataFrame,
    daily_logs_df: pd.DataFrame,
    temporal_targets_df: pd.DataFrame,
    user_profiles_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if image_labels_df.empty:
        return pd.DataFrame()

    images = image_labels_df.copy()
    images["anchor_date"] = pd.to_datetime(images["capture_date"], errors="coerce")
    images = images.dropna(subset=["anchor_date"]).sort_values(["user_id", "anchor_date"]).reset_index(drop=True)

    temporal = temporal_targets_df.copy()
    temporal["anchor_date"] = pd.to_datetime(temporal["anchor_date"], errors="coerce")
    temporal = temporal.sort_values(["user_id", "anchor_date"]).reset_index(drop=True)

    merged = pd.merge_asof(
        images.sort_values("anchor_date"),
        temporal.sort_values("anchor_date"),
        by="user_id",
        on="anchor_date",
        direction="nearest",
        tolerance=pd.Timedelta(days=3),
        suffixes=("", "_temporal"),
    )
    merged = merged.dropna(subset=["future_worsening_14d"]).reset_index(drop=True)

    if user_profiles_df is not None and not user_profiles_df.empty:
        merged = merged.merge(user_profiles_df, on="user_id", how="left")
    else:
        for column in DEFAULT_STATIC_COLUMNS:
            merged[column] = 0.0

    for column in DEFAULT_STATIC_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    if "split_temporal" in merged.columns:
        merged["split"] = merged["split"].fillna(merged["split_temporal"])
        merged = merged.drop(columns=["split_temporal"])

    keep_columns = [
        "user_id",
        "image_path",
        "anchor_date",
        "split",
        *DEFAULT_STATIC_COLUMNS,
        *DEFAULT_IMAGE_TARGETS,
        "future_worsening_7d",
        "future_worsening_14d",
        *DEFAULT_CAUSE_COLUMNS,
        "skin_score_delta_14d",
    ]
    return merged[keep_columns].reset_index(drop=True)


def write_preprocessed_artifacts(
    output_dir: str,
    image_labels_df: pd.DataFrame,
    daily_logs_df: pd.DataFrame,
    temporal_targets_df: pd.DataFrame,
    multimodal_targets_df: pd.DataFrame,
) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_path = output_path / "image_labels.csv"
    daily_path = output_path / "daily_logs.csv"
    temporal_path = output_path / "temporal_targets.csv"
    multimodal_path = output_path / "multimodal_targets.csv"
    report_path = output_path / "preprocessing_report.json"

    image_labels_df.to_csv(image_path, index=False)
    daily_logs_df.to_csv(daily_path, index=False)
    temporal_targets_df.to_csv(temporal_path, index=False)
    multimodal_targets_df.to_csv(multimodal_path, index=False)

    report = {
        "image_rows": int(len(image_labels_df)),
        "daily_log_rows": int(len(daily_logs_df)),
        "temporal_target_rows": int(len(temporal_targets_df)),
        "multimodal_target_rows": int(len(multimodal_targets_df)),
        "image_sources": image_labels_df["source_dataset"].value_counts().to_dict() if "source_dataset" in image_labels_df.columns else {},
        "strong_label_rows": int(image_labels_df.get("label_quality", pd.Series(dtype=str)).eq("strong").sum()) if not image_labels_df.empty else 0,
        "weak_label_rows": int(image_labels_df.get("label_quality", pd.Series(dtype=str)).eq("weak").sum()) if not image_labels_df.empty else 0,
    }
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    return {
        "image_labels_csv": str(image_path),
        "daily_logs_csv": str(daily_path),
        "temporal_targets_csv": str(temporal_path),
        "multimodal_targets_csv": str(multimodal_path),
        "report_json": str(report_path),
    }
