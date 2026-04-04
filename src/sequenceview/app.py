from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify, request

from sequenceview.model import ProteinModelService, clean_sequence
from sequenceview.sequence_analysis import analyze_sequence, find_invalid_residues, normalize_sequence, sanitize_for_analysis


def create_app() -> Flask:
    app = Flask(__name__)

    checkpoint_path = os.getenv("SEQUENCEVIEW_CHECKPOINT_PATH", "protein_classifier.pt")
    model_service = ProteinModelService(checkpoint_path=checkpoint_path)

    try:
        model_service.load()
    except Exception as error:  # pylint: disable=broad-exception-caught
        app.logger.warning("Model not loaded during startup: %s", error)

    @app.get("/health")
    def health() -> tuple[Any, int]:
        return jsonify({
            "status": "ok",
            "model_loaded": model_service.model_loaded,
            "checkpoint_path": str(model_service.checkpoint_path),
        }), 200

    @app.post("/api/predict")
    def predict() -> tuple[Any, int]:
        payload = request.get_json(silent=True) or {}
        raw_sequence = payload.get("sequence", "")

        if not isinstance(raw_sequence, str) or not raw_sequence.strip():
            return jsonify({"error": "'sequence' is required and must be a non-empty string."}), 400

        normalized = normalize_sequence(raw_sequence)
        if not normalized:
            return jsonify({"error": "Sequence is empty after normalization."}), 400

        invalid_residues = find_invalid_residues(normalized)
        analysis_sequence = sanitize_for_analysis(normalized)
        model_sequence = clean_sequence(normalized)

        if not analysis_sequence:
            return jsonify({
                "error": "Sequence has no standard amino acids after normalization.",
                "invalid_residues": invalid_residues,
                "allowed": "ACDEFGHIKLMNPQRSTVWY",
            }), 400

        try:
            sequence_data = analyze_sequence(analysis_sequence)
        except Exception as error:  # pylint: disable=broad-exception-caught
            return jsonify({"error": f"Failed to parse/analyze sequence: {error}"}), 400

        if not model_service.model_loaded:
            return jsonify({
                "error": "Model checkpoint is not loaded.",
                "checkpoint_path": str(model_service.checkpoint_path),
            }), 503

        try:
            prediction = model_service.predict(model_sequence)
        except Exception as error:  # pylint: disable=broad-exception-caught
            return jsonify({"error": f"Model prediction failed: {error}"}), 500

        sequence_data["invalid_residues"] = invalid_residues
        sequence_data["analysis_sequence"] = analysis_sequence
        sequence_data["prediction_sequence"] = model_sequence

        return jsonify({
            "input": {"sequence": raw_sequence},
            "analysis": sequence_data,
            "prediction": prediction,
        }), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
