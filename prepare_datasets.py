from __future__ import annotations

import argparse

from src.skin_coach.preprocessing import (
    DatasetSpec,
    build_multimodal_targets,
    build_temporal_targets,
    integrate_image_datasets,
    standardize_daily_logs,
    standardize_user_profiles,
    write_preprocessed_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate multiple skin datasets into one training-ready format.")

    parser.add_argument("--acne04-csv", type=str, default="")
    parser.add_argument("--acne04-root", type=str, default="")
    parser.add_argument("--acne04v2-csv", type=str, default="")
    parser.add_argument("--acne04v2-root", type=str, default="")
    parser.add_argument("--scin-csv", type=str, default="")
    parser.add_argument("--scin-root", type=str, default="")
    parser.add_argument("--ddi-csv", type=str, default="")
    parser.add_argument("--ddi-root", type=str, default="")
    parser.add_argument("--fitzpatrick-csv", type=str, default="")
    parser.add_argument("--fitzpatrick-root", type=str, default="")
    parser.add_argument("--wrinkle-csv", type=str, default="")
    parser.add_argument("--wrinkle-root", type=str, default="")
    parser.add_argument("--custom-image-csv", type=str, default="")
    parser.add_argument("--custom-image-root", type=str, default="")

    parser.add_argument("--daily-logs-csv", type=str, required=True)
    parser.add_argument("--user-profiles-csv", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="processed")
    parser.add_argument("--seq-len", type=int, default=14)
    parser.add_argument("--worsening-threshold-points", type=float, default=5.0)
    return parser.parse_args()


def make_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    # 여러 학교에서 온 성적표를 한 상자에 모으는 느낌으로 dataset spec을 만듭니다.
    candidates = [
        ("acne04", args.acne04_csv, args.acne04_root),
        ("acne04v2", args.acne04v2_csv, args.acne04v2_root),
        ("scin", args.scin_csv, args.scin_root),
        ("ddi", args.ddi_csv, args.ddi_root),
        ("fitzpatrick17k", args.fitzpatrick_csv, args.fitzpatrick_root),
        ("ffhq_wrinkle", args.wrinkle_csv, args.wrinkle_root),
        ("custom", args.custom_image_csv, args.custom_image_root),
    ]
    specs = [
        DatasetSpec(name=name, metadata_csv=csv_path, image_root=image_root)
        for name, csv_path, image_root in candidates
        if csv_path or image_root
    ]
    return specs


def main() -> None:
    args = parse_args()
    dataset_specs = make_specs(args)

    image_labels_df = integrate_image_datasets(dataset_specs)
    daily_logs_df = standardize_daily_logs(args.daily_logs_csv, image_labels_df=image_labels_df)
    user_profiles_df = standardize_user_profiles(args.user_profiles_csv)
    temporal_targets_df = build_temporal_targets(
        daily_logs_df,
        seq_len=args.seq_len,
        worsening_threshold_points=args.worsening_threshold_points,
    )
    multimodal_targets_df = build_multimodal_targets(
        image_labels_df=image_labels_df,
        daily_logs_df=daily_logs_df,
        temporal_targets_df=temporal_targets_df,
        user_profiles_df=user_profiles_df,
    )

    artifacts = write_preprocessed_artifacts(
        output_dir=args.output_dir,
        image_labels_df=image_labels_df,
        daily_logs_df=daily_logs_df,
        temporal_targets_df=temporal_targets_df,
        multimodal_targets_df=multimodal_targets_df,
    )

    print("Preprocessing complete.")
    for key, value in artifacts.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
