from __future__ import annotations

from collections import Counter
from textwrap import wrap
from typing import Any

from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from sequenceview.model import AA_ALPHABET

VALID_AMINO_ACIDS = set(AA_ALPHABET)


def normalize_sequence(raw_sequence: str) -> str:
    lines = [line.strip() for line in raw_sequence.splitlines() if line.strip()]
    if any(line.startswith(">") for line in lines):
        # FASTA headers start with '>' and should never be part of the sequence.
        lines = [line for line in lines if not line.startswith(">")]
    sequence = "".join(lines).replace(" ", "").upper()
    return sequence


def find_invalid_residues(sequence: str) -> list[str]:
    invalid = sorted({character for character in sequence if character not in VALID_AMINO_ACIDS})
    return invalid


def sanitize_for_analysis(sequence: str) -> str:
    return "".join(character for character in sequence if character in VALID_AMINO_ACIDS)


def format_sequence(sequence: str, line_width: int = 60) -> str:
    return "\n".join(wrap(sequence, line_width))


def amino_acid_counts(sequence: str) -> dict[str, int]:
    counts = Counter(sequence)
    return {aa: int(counts.get(aa, 0)) for aa in AA_ALPHABET}


def analyze_sequence(sequence: str) -> dict[str, Any]:
    protein_seq = Seq(sequence)
    analysis = ProteinAnalysis(str(protein_seq))

    # Biopython changed this from method to property in newer versions.
    aa_percent = (
        analysis.amino_acids_percent
        if hasattr(analysis, "amino_acids_percent")
        else analysis.get_amino_acids_percent()
    )
    aa_frequency = {aa: round(float(aa_percent.get(aa, 0.0)), 6) for aa in AA_ALPHABET}

    return {
        "normalized_sequence": str(protein_seq),
        "formatted_sequence": format_sequence(str(protein_seq)),
        "length": len(protein_seq),
        "molecular_weight": round(float(analysis.molecular_weight()), 4),
        "aromaticity": round(float(analysis.aromaticity()), 6),
        "instability_index": round(float(analysis.instability_index()), 6),
        "isoelectric_point": round(float(analysis.isoelectric_point()), 6),
        "gravy": round(float(analysis.gravy()), 6),
        "amino_acid_counts": amino_acid_counts(str(protein_seq)),
        "amino_acid_frequency": aa_frequency,
    }
