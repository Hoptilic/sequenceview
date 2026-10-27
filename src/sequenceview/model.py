from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_TO_INDEX = {amino_acid: index + 1 for index, amino_acid in enumerate(AA_ALPHABET)}
UNKNOWN_INDEX = len(AMINO_ACID_TO_INDEX) + 1
PAD_INDEX = 0
CLASS_NAMES = ("hydrolase", "oxidoreductase")


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
        return self.classify(self.embedding(inputs), inputs != PAD_INDEX)

    def classify(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.encoder(embeddings)
        outputs = outputs * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        mean_pool = outputs.sum(dim=1) / lengths
        max_pool = outputs.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        features = torch.cat([mean_pool, max_pool], dim=1)
        return self.head(features)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        self.temperature = 1.0
        self.checkpoint_sha256 = ""
        self.model_loaded = False
        self.confidence_threshold = float(os.getenv("SEQUENCEVIEW_CONFIDENCE_THRESHOLD", "0.70"))

    def load(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.checkpoint_sha256 = file_sha256(self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        vocab_size = int(checkpoint.get("vocab_size", len(AMINO_ACID_TO_INDEX) + 2))
        self.max_length = int(checkpoint.get("max_length", 768))
        self.metadata = dict(checkpoint.get("metadata", {}))
        self.temperature = float(checkpoint.get("temperature", 1.0))

        state_dict = checkpoint["state_dict"]
        model = ProteinClassifier(
            vocab_size=vocab_size,
            embedding_dim=int(state_dict["embedding.weight"].shape[1]),
            hidden_dim=int(state_dict["encoder.weight_hh_l0"].shape[1]),
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.model = model
        self.model_loaded = True

    def predict(self, sequence: str) -> dict[str, Any]:
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Model is not loaded")

        input_tensor = encode_sequence(sequence, self.max_length).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor) / self.temperature
            probabilities = torch.softmax(logits, dim=1)[0].cpu()

        predicted_index = int(probabilities.argmax().item())
        predicted_class = CLASS_NAMES[predicted_index]
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
            "calibrated": self.temperature != 1.0,
            "temperature": self.temperature,
            "max_length": self.max_length,
        }

    def attribute(self, sequence: str) -> dict[str, Any]:
        """Gradient x input saliency for the predicted class, one score per residue in [-1, 1]."""
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Model is not loaded")

        tokens = encode_sequence(sequence, self.max_length).unsqueeze(0).to(self.device)
        mask = tokens != PAD_INDEX
        length = min(len(sequence), self.max_length)

        # cuDNN refuses to backpropagate through RNNs in eval mode.
        with torch.backends.cudnn.flags(enabled=False):
            embeddings = self.model.embedding(tokens).detach().requires_grad_(True)
            logits = self.model.classify(embeddings, mask)[0]
            target = int(logits.argmax().item())
            (logits[target] - logits[1 - target]).backward()

        scores = (embeddings.grad * embeddings).sum(dim=-1)[0, :length].detach().cpu()
        peak = float(scores.abs().max().item()) if length else 0.0
        if peak > 0:
            scores = scores / peak

        return {
            "method": "gradient x input",
            "target": CLASS_NAMES[target],
            "length": length,
            "scores": [round(float(score), 4) for score in scores],
        }

    def info(self) -> dict[str, Any]:
        return {
            "loaded": self.model_loaded,
            "version": self.checkpoint_sha256[:12],
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_path": str(self.checkpoint_path),
            "classes": list(CLASS_NAMES),
            "max_length": self.max_length,
            "temperature": self.temperature,
            "confidence_threshold": self.confidence_threshold,
            "device": str(self.device),
            "metadata": self.metadata,
        }
