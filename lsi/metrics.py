"""仿真评价层：真值只在这里使用，不能参与求解/多初值解选择。"""
from __future__ import annotations

import numpy as np


def recovery_metrics(coeffs, truth, indices, pupil_basis, shear):
    coeffs, truth = np.asarray(coeffs), np.asarray(truth)

    def aligned(candidate):
        errors = [(pupil_basis @ (candidate - sign * truth)).std()
                  for sign in (1, -1)]
        sign = (1, -1)[int(np.argmin(errors))]
        return {"wavefront_rms_error": float(min(errors)),
                "max_coeff_error": float(np.max(np.abs(candidate - sign * truth))),
                "truth_sign": sign}

    raw = aligned(coeffs)
    # 对每个整体符号分别选择倾斜的周期代表，再比较完整波前误差。
    candidates = []
    tilt = np.isin(indices, [2, 3])
    for sign in (1, -1):
        delta = coeffs - sign * truth
        periods = np.zeros(len(coeffs), dtype=int)
        periods[tilt] = np.rint(delta[tilt] * shear).astype(int)
        adjusted = coeffs - periods / shear
        candidates.append((float((pupil_basis @ (adjusted-sign*truth)).std()),
                           float(np.max(np.abs(adjusted-sign*truth))), sign, periods))
    best = min(candidates, key=lambda item: item[0])
    raw.update({
        "equivalent_wavefront_rms_error": best[0],
        "equivalent_max_coeff_error": best[1],
        "equivalent_truth_sign": best[2],
        "tilt_alias": bool(np.any(best[3])),
        "historical_coeff_error_pm": float(min(np.max(np.abs(coeffs-truth)),
                                                 np.max(np.abs(coeffs+truth)))),
    })
    return raw
