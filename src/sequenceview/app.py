from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify, request, send_from_directory

from sequenceview.comparison import MAX_COMPARE_LENGTH, align_sequences
from sequenceview.model import AA_ALPHABET, ProteinModelService, clean_sequence
from sequenceview.motif_library import scan_motifs
from sequenceview.projection import pca_2d
from sequenceview.sequence_analysis import (
    analyze_sequence,
    find_invalid_residues,
    parse_fasta_records,
    sanitize_for_analysis,
    sequence_warnings,
)

MAX_BATCH_RECORDS = 500
MAX_SCAN_REGION = 250
MIN_PROJECTION_RECORDS = 3
BATCH_SUMMARY_FIELDS = ("molecular_weight", "isoelectric_point", "instability_index", "gravy")


class RequestError(Exception):
    def __init__(self, message: str, status: int = 400, **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.body = {"error": message, **extra}


def create_app() -> Flask:
    app = Flask(__name__)
    project_root = Path(__file__).resolve().parents[2]
    frontend_dist = Path(os.getenv("SEQUENCEVIEW_FRONTEND_DIST", project_root / "frontend" / "dist"))

    checkpoint_path_raw = Path(os.getenv("SEQUENCEVIEW_CHECKPOINT_PATH", "protein_classifier.pt"))
    checkpoint_path = checkpoint_path_raw if checkpoint_path_raw.is_absolute() else project_root / checkpoint_path_raw
    model_service = ProteinModelService(checkpoint_path=checkpoint_path)

    try:
        model_service.load()
    except Exception as error:  # pylint: disable=broad-exception-caught
        app.logger.warning("Model not loaded during startup: %s", error)

    def model_stamp() -> dict[str, Any]:
        return {
            "version": model_service.checkpoint_sha256[:12],
            "checkpoint_sha256": model_service.checkpoint_sha256,
            "max_length": model_service.max_length,
            "temperature": model_service.temperature,
            "trained_on": {
                key: model_service.metadata.get(key)
                for key in ("oxidoreductase_count", "hydrolase_count")
            },
        }

    def analyze_record(sequence: str, with_attribution: bool) -> dict[str, Any]:
        invalid_residues = find_invalid_residues(sequence)
        analysis_sequence = sanitize_for_analysis(sequence)
        if not analysis_sequence:
            raise RequestError(
                "Sequence has no standard amino acids after normalization.",
                invalid_residues=invalid_residues,
                allowed=AA_ALPHABET,
            )

        if not model_service.model_loaded:
            raise RequestError(
                "Model checkpoint is not loaded.",
                status=503,
                checkpoint_path=str(model_service.checkpoint_path),
            )

        try:
            analysis = analyze_sequence(analysis_sequence)
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise RequestError(f"Failed to parse/analyze sequence: {error}") from error

        model_sequence = clean_sequence(sequence)
        try:
            prediction = model_service.predict(model_sequence)
            attribution = model_service.attribute(model_sequence) if with_attribution else None
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise RequestError(f"Model prediction failed: {error}", status=500) from error

        analysis["invalid_residues"] = invalid_residues
        analysis["analysis_sequence"] = analysis_sequence
        analysis["prediction_sequence"] = model_sequence

        return {
            "analysis": analysis,
            "prediction": prediction,
            "attribution": attribution,
            "warnings": sequence_warnings(sequence, analysis_sequence, model_service.max_length),
            "motifs": scan_motifs(model_sequence),
            "model": model_stamp(),
        }

    def read_records() -> list[tuple[str, str]]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise RequestError("Request body must be a JSON object.")

        records_field = payload.get("records")
        if records_field is not None:
            if not isinstance(records_field, list):
                raise RequestError("'records' must be a list of {id, sequence} objects.")
            records = []
            for index, item in enumerate(records_field, start=1):
                if not isinstance(item, dict) or not isinstance(item.get("sequence"), str):
                    raise RequestError(f"Record {index} must be an object with a string 'sequence'.")
                for identifier, sequence in parse_fasta_records(item["sequence"]):
                    records.append((str(item.get("id") or identifier), sequence))
            return records

        raw_sequence = payload.get("sequence", "")
        if not isinstance(raw_sequence, str) or not raw_sequence.strip():
            raise RequestError("'sequence' is required and must be a non-empty string.")

        records = parse_fasta_records(raw_sequence)
        if not records:
            raise RequestError("Sequence is empty after normalization.")
        return records

    @app.errorhandler(RequestError)
    def handle_request_error(error: RequestError) -> tuple[Any, int]:
        return jsonify(error.body), error.status

    @app.get("/health")
    def health() -> tuple[Any, int]:
        return jsonify({
            "status": "ok",
            "model_loaded": model_service.model_loaded,
            "checkpoint_path": str(model_service.checkpoint_path),
            "model_version": model_service.checkpoint_sha256[:12],
        }), 200

    @app.get("/api/model")
    def model_info() -> tuple[Any, int]:
        return jsonify(model_service.info()), 200

    @app.post("/api/predict")
    def predict() -> tuple[Any, int]:
        records = read_records()
        if len(records) > 1:
            raise RequestError(
                f"Input contains {len(records)} records; use /api/predict/batch for multi-record input."
            )

        identifier, sequence = records[0]
        result = analyze_record(sequence, with_attribution=True)
        result["input"] = {"id": identifier, "sequence": sequence}
        return jsonify(result), 200

    @app.post("/api/predict/batch")
    def predict_batch() -> tuple[Any, int]:
        records = read_records()
        if len(records) > MAX_BATCH_RECORDS:
            raise RequestError(f"At most {MAX_BATCH_RECORDS} records per request ({len(records)} sent).")
        if not model_service.model_loaded:
            raise RequestError(
                "Model checkpoint is not loaded.",
                status=503,
                checkpoint_path=str(model_service.checkpoint_path),
            )

        results = []
        embeddings: list[tuple[int, list[float]]] = []
        for index, (identifier, sequence) in enumerate(records, start=1):
            entry: dict[str, Any] = {"index": index, "id": identifier, "sequence": sequence}
            try:
                detail = analyze_record(sequence, with_attribution=False)
            except RequestError as error:
                entry["error"] = error.body["error"]
            else:
                prediction = detail["prediction"]
                entry.update({
                    "length": detail["analysis"]["length"],
                    "class_name": prediction["class_name"],
                    "predicted_class": prediction["predicted_class"],
                    "is_uncertain": prediction["is_uncertain"],
                    "confidence": prediction["confidence"],
                    "margin": prediction["margin"],
                    "probabilities": prediction["probabilities"],
                    "warnings": detail["warnings"],
                    **{field: detail["analysis"][field] for field in BATCH_SUMMARY_FIELDS},
                })
                embeddings.append((index, model_service.embed(clean_sequence(sequence))))
            results.append(entry)

        projection = None
        if len(embeddings) >= MIN_PROJECTION_RECORDS:
            coordinates, explained = pca_2d([vector for _, vector in embeddings])
            projection = {
                "method": "PCA of pooled BiGRU features",
                "explained_variance": explained,
                "points": [
                    {"index": index, "x": x, "y": y}
                    for (index, _), (x, y) in zip(embeddings, coordinates)
                ],
            }

        return jsonify({"count": len(results), "results": results, "projection": projection, "model": model_stamp()}), 200

    def read_single_record(text: Any, label: str) -> tuple[str, str]:
        if not isinstance(text, str) or not text.strip():
            raise RequestError(f"'{label}' is required and must be a non-empty string.")
        records = parse_fasta_records(text)
        if len(records) != 1:
            raise RequestError(f"'{label}' must contain exactly one sequence ({len(records)} found).")
        return records[0]

    @app.post("/api/mutscan")
    def mutation_scan() -> tuple[Any, int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise RequestError("Request body must be a JSON object.")
        if not model_service.model_loaded:
            raise RequestError("Model checkpoint is not loaded.", status=503)

        identifier, sequence = read_single_record(payload.get("sequence"), "sequence")
        cleaned = clean_sequence(sequence)[: model_service.max_length]

        try:
            start = int(payload.get("start", 0))
            end = int(payload.get("end", min(len(cleaned), MAX_SCAN_REGION)))
        except (TypeError, ValueError) as error:
            raise RequestError("'start' and 'end' must be integers (0-based, end exclusive).") from error
        if start < 0 or end <= start or start >= len(cleaned):
            raise RequestError("Scan region is empty or outside the sequence.")
        if end - start > MAX_SCAN_REGION:
            raise RequestError(f"Scan region is limited to {MAX_SCAN_REGION} residues.")

        try:
            scan = model_service.scan_mutations(cleaned, start, end)
        except Exception as error:  # pylint: disable=broad-exception-caught
            raise RequestError(f"Mutation scan failed: {error}", status=500) from error

        scan["id"] = identifier
        scan["sequence_length"] = len(cleaned)
        scan["max_region"] = MAX_SCAN_REGION
        scan["model"] = model_stamp()
        return jsonify(scan), 200

    @app.post("/api/compare")
    def compare() -> tuple[Any, int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise RequestError("Request body must be a JSON object.")
        if not model_service.model_loaded:
            raise RequestError("Model checkpoint is not loaded.", status=503)

        sides = {}
        for label in ("a", "b"):
            identifier, sequence = read_single_record(payload.get(label), label)
            cleaned = clean_sequence(sequence)
            if len(cleaned) > MAX_COMPARE_LENGTH:
                raise RequestError(f"Sequence '{label}' is longer than {MAX_COMPARE_LENGTH} residues.")
            sides[label] = {"id": identifier, "sequence": cleaned, "prediction": model_service.predict(cleaned)}

        alignment = align_sequences(sides["a"]["sequence"], sides["b"]["sequence"])
        first, second = sides["a"]["prediction"], sides["b"]["prediction"]
        return jsonify({
            "a": sides["a"],
            "b": sides["b"],
            "alignment": alignment,
            "delta_oxidoreductase_probability": round(
                second["probabilities"]["oxidoreductase"] - first["probabilities"]["oxidoreductase"], 4
            ),
            "model": model_stamp(),
        }), 200

    def frontend_missing() -> tuple[Any, int]:
        return jsonify({
            "message": "Frontend build not found. Build React app in frontend/ and retry.",
            "expected_path": str(frontend_dist),
        }), 404

    @app.get("/")
    def index() -> Any:
        if (frontend_dist / "index.html").exists():
            return send_from_directory(frontend_dist, "index.html")
        return frontend_missing()

    @app.get("/<path:asset_path>")
    def frontend_assets(asset_path: str) -> Any:
        if asset_path.startswith("api/"):
            return jsonify({"error": "Not found."}), 404
        asset_file = frontend_dist / asset_path
        if asset_file.is_file():
            return send_from_directory(frontend_dist, asset_path)
        if (frontend_dist / "index.html").exists():
            return send_from_directory(frontend_dist, "index.html")
        return frontend_missing()

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG") == "1",
    )
