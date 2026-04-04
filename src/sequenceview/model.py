from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_TO_INDEX = {amino_acid: index + 1 for index, amino_acid in enumerate(AA_ALPHABET)}
UNKNOWN_INDEX = len(AMINO_ACID_TO_INDEX) + 1
PAD_INDEX = 0


class ProteinClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 192, dropout: float = 0.25) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_INDEX)
        self.encoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mask = inputs != PAD_INDEX
        embeddings = self.embedding(inputs)
        outputs, _ = self.encoder(embeddings)
        outputs = outputs * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        mean_pool = outputs.sum(dim=1) / lengths
        max_pool = outputs.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        features = torch.cat([mean_pool, max_pool], dim=1)
        return self.head(features)


def clean_sequence(sequence: str) -> str:
    allowed = set(AA_ALPHABET)
    return "".join(character if character in allowed else "X" for character in sequence.upper())


def encode_sequence(sequence: str, max_length: int) -> torch.Tensor:
    encoded = torch.full((max_length,), PAD_INDEX, dtype=torch.long)
    cleaned = clean_sequence(sequence)[:max_length]
    for position, character in enumerate(cleaned):
        encoded[position] = AMINO_ACID_TO_INDEX.get(character, UNKNOWN_INDEX)
    return encoded


class ProteinModelService:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: ProteinClassifier | None = None
        self.max_length = 768
        self.metadata: dict[str, Any] = {}
        self.model_loaded = False
        self.confidence_threshold = float(os.getenv("SEQUENCEVIEW_CONFIDENCE_THRESHOLD", "0.70"))

    def load(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        vocab_size = int(checkpoint.get("vocab_size", len(AMINO_ACID_TO_INDEX) + 2))
        self.max_length = int(checkpoint.get("max_length", 768))
        self.metadata = dict(checkpoint.get("metadata", {}))

        model = ProteinClassifier(vocab_size=vocab_size)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model
        self.model_loaded = True

    def predict(self, sequence: str) -> dict[str, Any]:
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Model is not loaded")

        input_tensor = encode_sequence(sequence, self.max_length).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(input_tensor), dim=1)[0].cpu()

        predicted_index = int(probabilities.argmax().item())
        predicted_class = "oxidoreductase" if predicted_index == 1 else "hydrolase"
        confidence = float(probabilities[predicted_index].item())
        hydrolase_probability = float(probabilities[0].item())
        oxidoreductase_probability = float(probabilities[1].item())
        margin = float(abs(oxidoreductase_probability - hydrolase_probability))
        is_uncertain = confidence < self.confidence_threshold

        return {
            "label": predicted_index,
            "class_name": "uncertain" if is_uncertain else predicted_class,
            "predicted_class": predicted_class,
            "is_uncertain": is_uncertain,
            "confidence": confidence,
            "confidence_threshold": self.confidence_threshold,
            "margin": margin,
            "probabilities": {
                "hydrolase": hydrolase_probability,
                "oxidoreductase": oxidoreductase_probability,
            },
            "max_length": self.max_length,
            "metadata": self.metadata,
        }
