"""波前质量指标与系数误差对比（单位：waves）。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence

import numpy as np

__all__ = [
    "pv",
    "rms",
    "wavefront_error",
    "coefficient_errors",
    "coefficient_error_metrics",
]


def pv(W: np.ndarray, pupil: np.ndarray | None = None) -> float:
    """峰谷值（忽略 NaN）。"""
    v = np.asarray(W, dtype=float)
    if pupil is not None:
        v = v[pupil]
    v = v[np.isfinite(v)]
    return float(v.max() - v.min()) if v.size else float("nan")


def rms(W: np.ndarray, pupil: np.ndarray | None = None) -> float:
    """去平移后的均方根。"""
    v = np.asarray(W, dtype=float)
    if pupil is not None:
        v = v[pupil]
    v = v[np.isfinite(v)]
    if not v.size:
        return float("nan")
    return float(np.sqrt(np.mean((v - v.mean()) ** 2)))


def wavefront_error(
    W_test: np.ndarray,
    W_ref: np.ndarray,
    pupil: np.ndarray | None = None,
) -> dict[str, float]:
    """差值指标（先在有效区域内去平移）。"""
    a, b = np.asarray(W_test, float), np.asarray(W_ref, float)
    valid = np.isfinite(a) & np.isfinite(b)
    if pupil is not None:
        valid &= pupil
    d = a[valid] - b[valid]
    d = d - d.mean()
    return {
        "rms": float(np.sqrt(np.mean(d**2))),
        "pv": float(d.max() - d.min()),
        "max_abs": float(np.max(np.abs(d))),
    }


def coefficient_errors(
    fitted: Mapping[int, float],
    truth_indices: Sequence[int],
    truth_coeffs: Sequence[float],
) -> dict[int, float]:
    """每个拟合模式的误差；真值中缺省的模式按 0 处理。"""
    truth = dict(zip((int(j) for j in truth_indices), (float(c) for c in truth_coeffs)))
    return {int(j): float(v) - truth.get(int(j), 0.0) for j, v in fitted.items()}


def coefficient_error_metrics(
    fitted: Mapping[int, float],
    truth_indices: Sequence[int],
    truth_coeffs: Sequence[float],
    *,
    exclude_indices: Sequence[int] = (),
) -> dict[str, float]:
    """汇总拟合模式误差，含向零真值模式的泄漏。"""
    excluded = {int(j) for j in exclude_indices}
    selected = {int(j): float(v) for j, v in fitted.items() if int(j) not in excluded}
    errors = coefficient_errors(selected, truth_indices, truth_coeffs)
    truth = {
        int(j): float(c)
        for j, c in zip(truth_indices, truth_coeffs)
        if int(j) not in excluded
    }
    signal = [abs(e) for j, e in errors.items() if truth.get(j, 0.0) != 0.0]
    leakage = [abs(e) for j, e in errors.items() if truth.get(j, 0.0) == 0.0]
    return {
        "max_error_all_modes": max(map(abs, errors.values()), default=0.0),
        "max_error_nonzero_truth_modes": max(signal, default=0.0),
        "max_leakage_into_zero_modes": max(leakage, default=0.0),
    }
