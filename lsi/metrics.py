"""仿真评价层：真值只在这里使用，不能参与求解/多初值解选择。"""
from __future__ import annotations

import numpy as np


def recovery_metrics(coeffs, truth, indices, pupil_basis, shear, *, sign="free"):
    """系数/波前恢复指标。返回的 dict 可直接并入 runs.csv 行。

    符号策略 ``sign``：

    * ``"free"``   旧协议：允许一个整体 ± 变号取最优——无载频观测下符号
                   物理不可观测。
    * ``"fixed"``  载频协议：符号固定为 +1——空间载频（及相移）观测打破
                   ± 简并，符号翻转即物理错误而非等价解。
    * ``±1``       显式指定固定符号。

    指标分三个互不替代的域，别名解不被默默排除也不计入 raw 成功：

    * raw（物理域，主指标）``wavefront_rms_error`` 为光瞳上差分波前的
      采样 std——只去 piston（减去采样均值，见 ``piston_removed``），
      保留 tilt；``max_coeff_error`` 为逐系数最大误差。
    * alias（倾斜周期等价域）``equivalent_*`` 允许 Z2/Z3 按 1/shear 的
      整数周期平移后取最优；``tilt_alias`` 标记是否动用了周期等价。
    * pm（历史兼容列）``historical_coeff_error_pm`` 始终按自由 ± 变号
      计算，仅用于与旧输出对照，不作评价依据。
    """
    coeffs = np.asarray(coeffs, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if coeffs.shape != truth.shape:
        raise ValueError("coeffs 与 truth 形状必须一致")
    if isinstance(sign, str):
        if sign not in ("free", "fixed"):
            raise ValueError("sign 必须是 'free'、'fixed' 或 ±1")
        signs = (1, -1) if sign == "free" else (1,)
    elif sign in (1, -1):
        signs = (int(sign),)
    else:
        raise ValueError("sign 必须是 'free'、'fixed' 或 ±1")

    # 对每个允许的整体符号计算 raw 域误差，取最优。
    raw = []
    for s in signs:
        diff = pupil_basis @ (coeffs - s * truth)
        raw.append((float(diff.std()), float(np.max(np.abs(coeffs - s * truth))),
                    float(diff.mean()), s))
    raw_err, raw_max, piston, truth_sign = min(raw, key=lambda item: item[0])

    # 对每个允许的整体符号分别选择倾斜的周期代表，再比较完整波前误差。
    tilt = np.isin(np.asarray(indices), [2, 3])
    candidates = []
    for s in signs:
        delta = coeffs - s * truth
        periods = np.zeros(len(coeffs), dtype=int)
        periods[tilt] = np.rint(delta[tilt] * shear).astype(int)
        adjusted = coeffs - periods / shear
        candidates.append((float((pupil_basis @ (adjusted - s * truth)).std()),
                           float(np.max(np.abs(adjusted - s * truth))), s, periods))
    best = min(candidates, key=lambda item: item[0])
    return {
        "wavefront_rms_error": raw_err,
        "max_coeff_error": raw_max,
        "truth_sign": truth_sign,
        "piston_removed": piston,
        "sign_policy": "free" if len(signs) == 2 else "fixed",
        "equivalent_wavefront_rms_error": best[0],
        "equivalent_max_coeff_error": best[1],
        "equivalent_truth_sign": best[2],
        "tilt_alias": bool(np.any(best[3])),
        "historical_coeff_error_pm": float(min(np.max(np.abs(coeffs - truth)),
                                               np.max(np.abs(coeffs + truth)))),
    }
