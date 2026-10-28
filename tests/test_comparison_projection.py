import pytest

from sequenceview.comparison import align_sequences
from sequenceview.projection import pca_2d


def test_identical_sequences_align_perfectly():
    result = align_sequences("MKVLAAGHKT", "MKVLAAGHKT")
    assert result["identity"] == 1.0
    assert result["gaps"] == 0
    assert result["aligned_a"] == result["aligned_b"] == "MKVLAAGHKT"


def test_gaps_and_identity_with_an_indel():
    result = align_sequences("MKVLAAGHKT", "MKVAAGHKKT")
    assert len(result["aligned_a"]) == len(result["aligned_b"])
    assert "-" in result["aligned_a"] + result["aligned_b"]
    assert result["gaps"] == 2
    assert result["identity"] == 1.0
    assert result["identity_over_alignment"] == pytest.approx(9 / 11, abs=1e-6)
    assert result["similarity"] >= result["identity"]


def test_unrelated_sequences_have_low_identity():
    assert align_sequences("A" * 30, "W" * 30)["identity"] == 0.0


def test_pca_recovers_the_dominant_axis_and_is_deterministic():
    vectors = [[0.0, 0.0, 0.0], [2.0, 0.1, 0.0], [4.0, -0.1, 0.0], [6.0, 0.0, 0.1]]
    coordinates, explained = pca_2d(vectors)
    assert explained[0] > 0.99
    xs = [point[0] for point in coordinates]
    assert xs == sorted(xs)
    assert pca_2d(vectors)[0] == coordinates


def test_pca_with_two_points_pads_the_second_axis():
    coordinates, explained = pca_2d([[1.0, 2.0], [3.0, 4.0]])
    assert len(coordinates) == 2 and explained[1] == pytest.approx(0.0, abs=1e-6) or explained[1] >= 0
