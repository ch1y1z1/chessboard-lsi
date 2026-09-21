"""Wavefront quality metrics and comparison helpers (units: waves)."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "pv",
    "rms",
    "wavefront_error",
    "coefficient_comparison",
    "coefficient_errors",
    "coefficient_error_metrics",
    "summary_table",
    "dump_json",
]


def pv(W: np.ndarray, pupil: np.ndarray | None = None) -> float:
    """Peak-to-valley of a wavefront (NaNs ignored)."""
    v = np.asarray(W, dtype=float)
    if pupil is not None:
        v = v[pupil]
    v = v[np.isfinite(v)]
    return float(v.max() - v.min()) if v.size else float("nan")


def rms(W: np.ndarray, pupil: np.ndarray | None = None) -> float:
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
    """Difference metrics after removing piston over the valid region."""
    a = np.asarray(W_test, dtype=float)
    b = np.asarray(W_ref, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if pupil is not None:
        valid &= pupil
    d = a[valid] - b[valid]
    if not d.size:
        return {"rms": float("nan"), "pv": float("nan"), "max_abs": float("nan")}
    d = d - d.mean()
    return {
        "rms": float(np.sqrt(np.mean(d**2))),
        "pv": float(d.max() - d.min()),
        "max_abs": float(np.max(np.abs(d))),
    }


def coefficient_comparison(
    indices: Sequence[int],
    c_test: Sequence[float],
    c_ref: Sequence[float],
) -> list[dict[str, float]]:
    indices = list(indices)
    c_test = list(c_test)
    c_ref = list(c_ref)
    lengths = (len(indices), len(c_test), len(c_ref))
    if len(set(lengths)) != 1:
        raise ValueError(
            "indices, c_test and c_ref must have the same length, got "
            f"{lengths}"
        )
    out = []
    for j, a, b in zip(indices, c_test, c_ref):
        out.append(
            {"j": int(j), "fitted": float(a), "true": float(b), "error": float(a - b)}
        )
    return out


def coefficient_errors(
    fitted: Mapping[int, float],
    truth_indices: Sequence[int],
    truth_coeffs: Sequence[float],
) -> dict[int, float]:
    """Error for every fitted mode, with omitted truth modes treated as zero."""
    truth_indices = [int(j) for j in truth_indices]
    truth_coeffs = [float(c) for c in truth_coeffs]
    if len(truth_indices) != len(truth_coeffs):
        raise ValueError(
            "truth_indices and truth_coeffs must have the same length, got "
            f"{len(truth_indices)} and {len(truth_coeffs)}"
        )
    if len(set(truth_indices)) != len(truth_indices):
        raise ValueError("truth_indices must not contain duplicates")
    truth = dict(zip(truth_indices, truth_coeffs))
    return {
        int(j): float(value) - truth.get(int(j), 0.0)
        for j, value in fitted.items()
    }


def coefficient_error_metrics(
    fitted: Mapping[int, float],
    truth_indices: Sequence[int],
    truth_coeffs: Sequence[float],
    *,
    exclude_indices: Sequence[int] = (),
) -> dict[str, float]:
    """Summarize fitted-mode error, including leakage into zero-truth modes."""
    excluded = {int(j) for j in exclude_indices}
    selected = {
        int(j): float(value)
        for j, value in fitted.items()
        if int(j) not in excluded
    }
    errors = coefficient_errors(selected, truth_indices, truth_coeffs)
    truth = {
        int(j): float(c)
        for j, c in zip(truth_indices, truth_coeffs)
        if int(j) not in excluded
    }
    signal_errors = [
        abs(error)
        for j, error in errors.items()
        if truth.get(j, 0.0) != 0.0
    ]
    leakage = [
        abs(error)
        for j, error in errors.items()
        if truth.get(j, 0.0) == 0.0
    ]
    return {
        "max_error_all_modes": max(map(abs, errors.values()), default=0.0),
        "max_error_nonzero_truth_modes": max(signal_errors, default=0.0),
        "max_leakage_into_zero_modes": max(leakage, default=0.0),
    }


def summary_table(rows: dict[str, dict[str, float]], title: str = "") -> str:
    """Render a small fixed-width table for the console."""
    keys = sorted({k for r in rows.values() for k in r})
    name_w = max(len("case"), *(len(k) for k in rows)) if rows else 6
    head = f"{'case':<{name_w}} " + " ".join(f"{k:>14}" for k in keys)
    lines = [title, head, "-" * len(head)] if title else [head, "-" * len(head)]
    for name, vals in rows.items():
        lines.append(
            f"{name:<{name_w}} "
            + " ".join(f"{vals.get(k, float('nan')):>14.6g}" for k in keys)
        )
    return "\n".join(lines)


def dump_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, complex):
            return [o.real, o.imag]
        raise TypeError(f"not serializable: {type(o)}")

    path.write_text(json.dumps(data, indent=2, default=default))