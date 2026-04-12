from __future__ import annotations

from typing import List


DEFAULT_IMAGE_TARGETS = [
    "acne_score",
    "redness_score",
    "dryness_score",
    "wrinkle_score",
    "pigmentation_score",
    "oiliness_score",
    "pore_score",
    "overall_skin_score",
]

DEFAULT_TEMPORAL_FEATURES = [
    "sleep_hours",
    "water_liters",
    "sugar_score",
    "dairy_score",
    "spicy_food_score",
    "exercise_minutes",
    "stress_score",
    "mask_hours",
    "uv_exposure_hours",
    "cleanser_use",
    "moisturizer_use",
    "product_changed",
    "temperature_c",
    "humidity",
    "pm25",
]

DEFAULT_TEMPORAL_SCORE_COLUMNS = [
    "acne_score",
    "redness_score",
    "dryness_score",
    "wrinkle_score",
    "pigmentation_score",
    "oiliness_score",
    "pore_score",
    "overall_skin_score",
]

DEFAULT_RISK_COLUMNS = [
    "future_worsening_7d",
    "future_worsening_14d",
]

DEFAULT_CAUSE_COLUMNS = [
    "cause_sleep",
    "cause_stress",
    "cause_product_change",
    "cause_uv",
    "cause_diet",
    "cause_mask",
]

DEFAULT_STATIC_COLUMNS = [
    "age_scaled",
    "sex_encoded",
    "family_history",
    "sensitive_skin",
    "baseline_skin_type_encoded",
]


def parse_columns(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]
