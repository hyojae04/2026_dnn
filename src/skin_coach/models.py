from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision import models


def _build_backbone(backbone_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    name = backbone_name.lower()
    if name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)
        feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        return backbone, feature_dim
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        feature_dim = backbone.classifier[2].in_features
        backbone.classifier = nn.Identity()
        return backbone, feature_dim
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim
    raise ValueError(f"Unsupported backbone: {backbone_name}")


class ImageEncoder(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b3", pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        self.backbone, feature_dim = _build_backbone(backbone_name, pretrained=pretrained)
        self.feature_dim = feature_dim
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.projection(features)


class ImageMultiHeadModel(nn.Module):
    def __init__(
        self,
        target_columns: List[str],
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.target_columns = target_columns
        self.encoder = ImageEncoder(backbone_name=backbone_name, pretrained=pretrained, dropout=dropout)
        self.heads = nn.ModuleDict(
            {
                target: nn.Sequential(
                    nn.Linear(self.encoder.feature_dim, 128),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, 1),
                )
                for target in target_columns
            }
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.encoder(images)
        scores = {target: torch.sigmoid(head(embedding)).squeeze(-1) for target, head in self.heads.items()}
        return {"embedding": embedding, "scores": scores}


class TemporalAttentionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = nn.Linear(self.output_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(sequence)
        encoded, _ = self.rnn(x)
        encoded = self.dropout(encoded)

        attn_logits = self.attention(encoded).squeeze(-1)
        attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        pooled = torch.sum(encoded * attn_weights.unsqueeze(-1), dim=1)
        return pooled


class TemporalCauseModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        risk_columns: List[str],
        cause_columns: List[str],
        delta_columns: List[str],
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.risk_columns = risk_columns
        self.cause_columns = cause_columns
        self.delta_columns = delta_columns
        self.encoder = TemporalAttentionEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.risk_head = nn.Linear(self.encoder.output_dim, len(risk_columns)) if risk_columns else None
        self.cause_head = nn.Linear(self.encoder.output_dim, len(cause_columns)) if cause_columns else None
        self.delta_head = nn.Linear(self.encoder.output_dim, len(delta_columns)) if delta_columns else None

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.encoder(sequence, mask)
        outputs: Dict[str, torch.Tensor] = {"embedding": embedding}
        if self.risk_head is not None:
            outputs["risk_logits"] = self.risk_head(embedding)
        if self.cause_head is not None:
            outputs["cause_logits"] = self.cause_head(embedding)
        if self.delta_head is not None:
            outputs["delta_pred"] = torch.tanh(self.delta_head(embedding))
        return outputs


class MultimodalFusionModel(nn.Module):
    def __init__(
        self,
        image_target_columns: List[str],
        temporal_input_dim: int,
        static_input_dim: int,
        risk_columns: List[str],
        cause_columns: List[str],
        change_columns: List[str],
        backbone_name: str = "efficientnet_b3",
        hidden_dim: int = 128,
        dropout: float = 0.2,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.image_target_columns = image_target_columns
        self.risk_columns = risk_columns
        self.cause_columns = cause_columns
        self.change_columns = change_columns

        self.image_encoder = ImageEncoder(backbone_name=backbone_name, pretrained=pretrained_backbone, dropout=dropout)
        self.temporal_encoder = TemporalAttentionEncoder(
            input_dim=temporal_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(static_input_dim),
            nn.Linear(static_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fusion_dim = self.image_encoder.feature_dim + self.temporal_encoder.output_dim + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Linear(256, len(image_target_columns)) if image_target_columns else None
        self.risk_head = nn.Linear(256, len(risk_columns)) if risk_columns else None
        self.cause_head = nn.Linear(256, len(cause_columns)) if cause_columns else None
        self.change_head = nn.Linear(256, len(change_columns)) if change_columns else None

    def forward(
        self,
        images: torch.Tensor,
        sequence: torch.Tensor,
        sequence_mask: torch.Tensor,
        static_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        image_embedding = self.image_encoder(images)
        temporal_embedding = self.temporal_encoder(sequence, sequence_mask)
        static_embedding = self.static_encoder(static_features)
        fused = self.fusion(torch.cat([image_embedding, temporal_embedding, static_embedding], dim=-1))

        outputs: Dict[str, torch.Tensor] = {"embedding": fused}
        if self.score_head is not None:
            outputs["score_pred"] = torch.sigmoid(self.score_head(fused))
        if self.risk_head is not None:
            outputs["risk_logits"] = self.risk_head(fused)
        if self.cause_head is not None:
            outputs["cause_logits"] = self.cause_head(fused)
        if self.change_head is not None:
            outputs["change_pred"] = torch.tanh(self.change_head(fused))
        return outputs
