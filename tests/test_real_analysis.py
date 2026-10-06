from __future__ import annotations

import math

import pytest

from robotics_vision.real_analysis import distribution, spearman, summarize_real


def test_spearman_handles_ties_constant_and_negative_associations():
    assert spearman([1, 1, 2, 3], [10, 10, 30, 40]) == pytest.approx(1)
    assert spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])


def test_distribution_preserves_negative_seeds_and_uses_sample_sd():
    values = distribution([-1.0, 0.0, 2.0])
    assert values["individual_values"] == [-1.0, 0.0, 2.0]
    assert values["mean"] == pytest.approx(1 / 3)
    assert values["sample_sd"] == pytest.approx(math.sqrt(7 / 3))
    assert distribution([2.0])["sample_sd"] is None


def test_analysis_refuses_to_present_incomplete_matrix_as_aggregate():
    with pytest.raises(ValueError, match="entire"):
        summarize_real({"complete": False, "records": []})
