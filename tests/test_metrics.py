"""Coefficient-error metrics used by the experiment reports."""

import pytest

from lsi.metrics import coefficient_error_metrics, coefficient_errors


def test_coefficient_errors_include_leakage_into_zero_truth_modes():
    fitted = {4: 0.21, 7: -0.30, 9: 0.08}

    errors = coefficient_errors(fitted, [4, 7], [0.20, -0.25])
    metrics = coefficient_error_metrics(fitted, [4, 7], [0.20, -0.25])

    assert errors == pytest.approx({4: 0.01, 7: -0.05, 9: 0.08})
    assert metrics == pytest.approx({
        "max_error_all_modes": 0.08,
        "max_error_nonzero_truth_modes": 0.05,
        "max_leakage_into_zero_modes": 0.08,
    })


def test_coefficient_error_metrics_can_exclude_gauge_modes():
    fitted = {2: 10.0, 4: 0.21, 9: 0.08}

    metrics = coefficient_error_metrics(
        fitted,
        [4],
        [0.20],
        exclude_indices=(2, 3),
    )

    assert metrics["max_error_all_modes"] == pytest.approx(0.08)
    assert metrics["max_leakage_into_zero_modes"] == pytest.approx(0.08)


def test_coefficient_errors_reject_invalid_truth_tables():
    with pytest.raises(ValueError, match="same length"):
        coefficient_errors({4: 0.2}, [4, 7], [0.2])
    with pytest.raises(ValueError, match="duplicates"):
        coefficient_errors({4: 0.2}, [4, 4], [0.2, 0.3])
