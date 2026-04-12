"""
Colab quickstart example

1. Upload this repo to Google Drive or GitHub.
2. In Colab:

    !git clone <your-repo-url>
    %cd 2026_DNN
    !pip install -q torch torchvision pandas pillow tqdm

3. Train each model:

    !python train_image_model.py \
        --train-csv /content/data/image_labels.csv \
        --image-root /content/data/images

    !python train_temporal_model.py \
        --daily-logs-csv /content/data/daily_logs.csv \
        --targets-csv /content/data/temporal_targets.csv

    !python train_multimodal_model.py \
        --multimodal-csv /content/data/multimodal_targets.csv \
        --daily-logs-csv /content/data/daily_logs.csv \
        --image-root /content/data/images
"""

