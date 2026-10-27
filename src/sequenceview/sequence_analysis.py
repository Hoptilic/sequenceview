from __future__ import annotations

from collections import Counter
from textwrap import wrap
from typing import Any

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.SeqUtils.ProtParamData import kd

from sequenceview.model import AA_ALPHABET

VALID_AMINO_ACIDS = set(AA_ALPHABET)
TRAIN_MIN_LENGTH = 50
TRAIN_MAX_LENGTH = 1000
MAX_UNKNOWN_FRACTION = 0.02
HYDROPATHY_WINDOW = 9
CHARGE_CURVE_PH = [step / 2 for step in range(2, 29)]


def parse_fasta_records(raw_sequence: str) -> list[tuple[str, str]]:
    """Split plain or FASTA text into (identifier, sequence) pairs."""
    records: list[tuple[str, list[str]]] = []
    for line in raw_sequence.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            records.append((line[1:].strip() or f"record_{len(records) + 1}", []))
        else:
            if not records:
                records.append(("sequence", []))
            records[-1][1].append(line)

    parsed = []
    for identifier, parts in records:
        sequence = "".join(parts).replace(" ", "").upper()
        if sequence:
            parsed.append((identifier, sequence))
    return parsed


def normalize_sequence(raw_sequence: str) -> str:
    return "".join(sequence for _, sequence in parse_fasta_records(raw_sequence))


def find_invalid_residues(sequence: str) -> list[str]:
    return sorted({character for character in sequence if character not in VALID_AMINO_ACIDS})


def sanitize_for_analysis(sequence: str) -> str:
    return "".join(character for character in sequence if character in VALID_AMINO_ACIDS)


def format_sequence(sequence: str, line_width: int = 60) -> str:
    return "\n".join(wrap(sequence, line_width))


def amino_acid_counts(sequence: str) -> dict[str, int]:
    counts = Counter(sequence)
    return {aa: int(counts.get(aa, 0)) for aa in AA_ALPHABET}


def sequence_warnings(normalized: str, analysis_sequence: str, max_length: int) -> list[dict[str, str]]:
    """Flag inputs that fall outside what the classifier was trained on."""
    warnings = []
    length = len(analysis_sequence)

    if length < TRAIN_MIN_LENGTH:
        warnings.append({
            "code": "short_sequence",
            "message": f"Sequence has {length} residues; the model was trained on {TRAIN_MIN_LENGTH} to {TRAIN_MAX_LENGTH}.",
        })
    if length > TRAIN_MAX_LENGTH:
        warnings.append({
            "code": "long_sequence",
            "message": f"Sequence has {length} residues; the model was trained on {TRAIN_MIN_LENGTH} to {TRAIN_MAX_LENGTH}.",
        })
    if len(normalized) > max_length:
        warnings.append({
            "code": "truncated",
            "message": f"Only the first {max_length} residues are seen by the model.",
        })
    if normalized:
        unknown_fraction = 1 - length / len(normalized)
        if unknown_fraction > MAX_UNKNOWN_FRACTION:
            warnings.append({
                "code": "non_standard_residues",
                "message": f"{unknown_fraction:.1%} of residues are non-standard (training data allowed up to {MAX_UNKNOWN_FRACTION:.0%}).",
            })
    return warnings


def analyze_sequence(sequence: str) -> dict[str, Any]:
    analysis = ProteinAnalysis(sequence)

    counts = amino_acid_counts(sequence)
    aa_frequency = {aa: round(counts[aa] / len(sequence), 6) for aa in AA_ALPHABET}

    helix, turn, sheet = analysis.secondary_structure_fraction()
    extinction_reduced, extinction_oxidized = analysis.molar_extinction_coefficient()

    if len(sequence) >= HYDROPATHY_WINDOW:
        hydropathy = [round(float(value), 4) for value in analysis.protein_scale(kd, HYDROPATHY_WINDOW, 1.0)]
    else:
        hydropathy = []

    return {
        "normalized_sequence": sequence,
        "formatted_sequence": format_sequence(sequence),
        "length": len(sequence),
        "molecular_weight": round(float(analysis.molecular_weight()), 4),
        "aromaticity": round(float(analysis.aromaticity()), 6),
        "instability_index": round(float(analysis.instability_index()), 6),
        "isoelectric_point": round(float(analysis.isoelectric_point()), 6),
        "gravy": round(float(analysis.gravy()), 6),
        "charge_at_ph7": round(float(analysis.charge_at_pH(7.0)), 4),
        "extinction_coefficient_reduced": int(extinction_reduced),
        "extinction_coefficient_oxidized": int(extinction_oxidized),
        "secondary_structure_fraction": {
            "helix": round(float(helix), 6),
            "turn": round(float(turn), 6),
            "sheet": round(float(sheet), 6),
        },
        "hydropathy_profile": {"window": HYDROPATHY_WINDOW, "values": hydropathy},
        "charge_curve": [[ph, round(float(analysis.charge_at_pH(ph)), 4)] for ph in CHARGE_CURVE_PH],
        "amino_acid_counts": counts,
        "amino_acid_frequency": aa_frequency,
    }
