"""Fusion model: dual-embedding classifier for BirdNET + PERCH.

This module contains only the model architecture and forward pass
needed for inference.  Training code, dataset classes, and evaluation
utilities have been removed.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualEmbeddingClassifier(nn.Module):
    """Fusion classifier combining BirdNET and PERCH embeddings.

    Takes a concatenated ``[birdnet_dim | perch_dim]`` input, projects
    each half through separate projection heads, concatenates the
    projections, and passes through fusion layers to a classification
    head.
    """

    def __init__(
        self,
        n_species: int,
        birdnet_dim: int = 87,
        perch_dim: int = 87,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.n_species = n_species
        self.birdnet_dim = birdnet_dim
        self.perch_dim = perch_dim

        # BirdNET projection
        self.birdnet_projection = nn.Sequential(
            nn.Linear(birdnet_dim, hidden_dims[0] // 2),
            nn.BatchNorm1d(hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # PERCH projection
        self.perch_projection = nn.Sequential(
            nn.Linear(perch_dim, hidden_dims[0] // 2),
            nn.BatchNorm1d(hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion layers
        layers: list[nn.Module] = []
        in_dim = hidden_dims[0]
        for h in hidden_dims[1:]:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        self.fusion_layers = nn.Sequential(*layers)

        # Output
        self.classifier = nn.Linear(hidden_dims[-1], n_species)

    def forward(self, combined_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            combined_embeddings: ``(batch, birdnet_dim + perch_dim)``

        Returns:
            Logits ``(batch, n_species)``
        """
        birdnet_emb = combined_embeddings[:, : self.birdnet_dim]
        perch_emb = combined_embeddings[:, self.birdnet_dim :]

        birdnet_proj = self.birdnet_projection(birdnet_emb)
        perch_proj = self.perch_projection(perch_emb)

        fused = torch.cat([birdnet_proj, perch_proj], dim=1)
        features = self.fusion_layers(fused)
        return self.classifier(features)


def load_fusion_model(
    weights_path: str,
    config_path: str,
    device: torch.device | None = None,
) -> tuple[DualEmbeddingClassifier, dict]:
    """Load a trained fusion model from weights and architecture config.

    Args:
        weights_path: Path to ``.pth`` file.
        config_path: Path to ``architecture_info_public.json``.
        device: Torch device.  Defaults to CUDA if available.

    Returns:
        ``(model, config_dict)`` with model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    arch = config["architecture"]
    model = DualEmbeddingClassifier(
        n_species=config["n_species"],
        birdnet_dim=arch["birdnet_dim"],
        perch_dim=arch["perch_dim"],
        hidden_dims=arch["hidden_dims"],
        dropout=0.3,
    )

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Handle both checkpoint format and raw state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config
