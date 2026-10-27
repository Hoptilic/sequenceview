from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify, request, send_from_directory

from sequenceview.model import AA_ALPHABET, ProteinModelService, clean_sequence
from sequenceview.sequence_analysis import (
    analyze_sequence,
    find_invalid_residues,
    parse_fasta_records,
    sanitize_for_analysis,
    sequence_warnings,
)

MAX_BATCH_RECORDS = 500
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
            results.append(entry)

        return jsonify({"count": len(results), "results": results, "model": model_stamp()}), 200

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
