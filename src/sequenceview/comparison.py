from __future__ import annotations

from typing import Any

from Bio.Align import PairwiseAligner, substitution_matrices

MAX_COMPARE_LENGTH = 2000


def align_sequences(first: str, second: str) -> dict[str, Any]:
    """Global BLOSUM62 alignment with affine gaps; returns gapped strings and identity statistics."""
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.mode = "global"

    alignment = aligner.align(first, second)[0]
    row_a, row_b = alignment[0], alignment[1]

    aligned_columns = [(a, b) for a, b in zip(row_a, row_b) if a != "-" and b != "-"]
    identical = sum(1 for a, b in aligned_columns if a == b)
    similar = sum(1 for a, b in aligned_columns if a == b or aligner.substitution_matrix[a, b] > 0)
    shorter = min(len(first), len(second))

    return {
        "score": float(alignment.score),
        "aligned_a": row_a,
        "aligned_b": row_b,
        "aligned_columns": len(aligned_columns),
        "identity": identical / len(aligned_columns) if aligned_columns else 0.0,
        "similarity": similar / len(aligned_columns) if aligned_columns else 0.0,
        "identity_over_alignment": identical / len(row_a) if row_a else 0.0,
        "identity_over_shorter": identical / shorter if shorter else 0.0,
        "gaps": len(row_a) - len(aligned_columns),
    }
