import re

import pytest

from sequenceview.motif_library import MOTIFS, prosite_to_regex, scan_motifs


def test_conversion_handles_each_syntax_element():
    assert prosite_to_regex("[LIVM]-{KG}-x(2)-G-x(8,9)-C.") == "[LIVM][^KG].{2}G.{8,9}C"
    assert prosite_to_regex("<M-x-S>") == "^M.S$"


def test_every_library_pattern_compiles_and_is_well_formed():
    for motif in MOTIFS:
        re.compile(prosite_to_regex(motif["pattern"]))
        assert motif["class"] in {"hydrolase", "oxidoreductase"}
        assert motif["id"] == "" or re.fullmatch(r"PS\d{5}", motif["id"])


def test_known_signatures_are_found_with_positions():
    trypsin_histidine = "AAAALTAAHCAAAA"
    hits = {hit["id"]: hit for hit in scan_motifs(trypsin_histidine)}
    assert hits["PS00134"]["matches"][0] == {"start": 4, "end": 10, "match": "LTAAHC"}
    assert hits["PS00134"]["class"] == "hydrolase"

    zinc_adh = "MKGHEAAGIVESIGEGVTTVRPG"
    assert "PS00059" in {hit["id"] for hit in scan_motifs(zinc_adh)}


def test_overlapping_matches_are_all_reported():
    hits = [hit for hit in scan_motifs("GASAGASAG") if hit["name"].startswith("GxSxG")]
    starts = [match["start"] for match in hits[0]["matches"]]
    assert starts == [0, 4]


def test_no_hits_on_unrelated_sequence():
    assert scan_motifs("A" * 60) == []


@pytest.mark.parametrize("sequence", ["", "M", "XXXX"])
def test_degenerate_input_is_safe(sequence):
    assert scan_motifs(sequence) == []
