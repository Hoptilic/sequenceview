import pytest

from sequenceview.sequence_analysis import (
    analyze_sequence,
    find_invalid_residues,
    parse_fasta_records,
    sanitize_for_analysis,
    sequence_warnings,
)


def test_parse_plain_sequence():
    assert parse_fasta_records("mkv lla\n  gh") == [("sequence", "MKVLLAGH")]


def test_parse_multiple_fasta_records():
    text = ">first protein\nMKV\nLLA\n\n>second\nGHK\n"
    assert parse_fasta_records(text) == [("first protein", "MKVLLA"), ("second", "GHK")]


def test_parse_skips_empty_records_and_names_blank_headers():
    assert parse_fasta_records(">a\n>\nMKV") == [("record_2", "MKV")]


def test_invalid_residue_handling():
    assert find_invalid_residues("MKXB*Z") == ["*", "B", "X", "Z"]
    assert sanitize_for_analysis("MKXB*Z") == "MK"


def test_analysis_fields(lysozyme):
    result = analyze_sequence(lysozyme)
    assert result["length"] == len(lysozyme)
    assert result["molecular_weight"] == pytest.approx(14313.1, abs=1.0)
    assert sum(result["amino_acid_counts"].values()) == len(lysozyme)
    assert 0 < sum(result["secondary_structure_fraction"].values()) <= 1.0 + 1e-3
    assert len(result["hydropathy_profile"]["values"]) == len(lysozyme) - 8
    assert result["charge_curve"][0][0] < result["charge_curve"][-1][0]
    assert result["charge_curve"][0][1] > result["charge_curve"][-1][1]
    assert result["extinction_coefficient_reduced"] <= result["extinction_coefficient_oxidized"]


def test_short_sequence_has_empty_hydropathy_profile():
    assert analyze_sequence("MKVL")["hydropathy_profile"]["values"] == []


def test_warnings(lysozyme):
    assert sequence_warnings(lysozyme, lysozyme, 768) == []

    codes = {w["code"] for w in sequence_warnings("MKV", "MKV", 768)}
    assert codes == {"short_sequence"}

    long_sequence = "A" * 1200
    codes = {w["code"] for w in sequence_warnings(long_sequence, long_sequence, 768)}
    assert codes == {"long_sequence", "truncated"}

    noisy = "A" * 100 + "X" * 10
    codes = {w["code"] for w in sequence_warnings(noisy, "A" * 100, 768)}
    assert codes == {"non_standard_residues"}
