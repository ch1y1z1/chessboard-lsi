"""45° 旋转振幅棋盘光栅的衍射级次振幅（论文 2.1、4.1.1）。

探测器坐标 (a, b) 与光栅坐标 (m, n) 的关系（45° 旋转）：

    a = (m + n) / 2,   b = (n - m) / 2       <=>   m = a - b, n = a + b

理想 50% 占空比（论文表 2-3）：

    A_00 = 1/2,   A_ab = -2 / (pi^2 (a^2 - b^2))   (a + b 为奇数),  其余为 0

即 |A_±1| = 2/pi^2 = 0.2026（效率 4.11%），Parseval: sum |A|^2 = 1/2。

占空比误差（论文 4.1.1）提供两种模型：

``"complementary"``（默认）
    光栅坐标下 t = (1 + s(u) s(v)) / 2，s 为占空比 d 的 ±1 方波：
    一个透明子格为 d×d、对角另一格为 (1-d)×(1-d)，开孔率恒为 1/2。
    A_mn = S_m S_n / 2，S_k = (1 - e^{-i 2 pi k d}) / (i pi k)。
    d = 1/2 时精确退化为表 2-3 的实振幅。

``"enlarged"``
    论文式 (4-1)：两个透明方块 [0,w]^2 与 [1/2, 1/2+w]^2 同步长到 w = d，
    两孔径线性叠加：
    A_mn = 2 w^2 sinc(mw) sinc(nw) e^{-i pi w (m+n)}   (m+n 偶),  0 (m+n 奇)
    w = 1/2 时同样回到表 2-3。此模型下偶数级以 O(delta) 出现，可与表 4-1 对照。
"""

from __future__ import annotations

import numpy as np

__all__ = ["chessboard_orders", "diffraction_efficiency"]


def chessboard_orders(
    max_index: int = 3, duty: float = 0.5, model: str = "complementary"
) -> dict[tuple[float, float], complex]:
    """探测器坐标下的级次振幅表 ``{(a, b): A_ab}``（含 (0, 0) 级）。

    ``max_index`` 保留 max(|a|, |b|) <= max_index 的级次。
    """
    if not 0.0 < duty < 1.0:
        raise ValueError("duty 必须在 (0, 1) 内")
    if model not in ("complementary", "enlarged"):
        raise ValueError("model 必须是 'complementary' 或 'enlarged'")

    def square_wave(k: int) -> complex:
        if k == 0:
            return complex(2.0 * duty - 1.0)
        return -np.expm1(-2j * np.pi * k * duty) / (1j * np.pi * k)

    def amplitude(m: int, n: int) -> complex:
        if model == "enlarged":
            if (m + n) % 2:
                return 0.0j
            return (
                2.0 * duty**2 * np.sinc(m * duty) * np.sinc(n * duty)
                * np.exp(-1j * np.pi * duty * (m + n))
            )
        if duty == 0.5:
            if m % 2 == 0 or n % 2 == 0:
                return 0.0j
            return -2.0 / (np.pi**2 * m * n)
        return 0.5 * square_wave(m) * square_wave(n)

    orders: dict[tuple[float, float], complex] = {}
    for m in range(-2 * max_index, 2 * max_index + 1):
        for n in range(-2 * max_index, 2 * max_index + 1):
            if (m, n) == (0, 0):
                continue
            a, b = (m + n) / 2.0, (n - m) / 2.0
            if max(abs(a), abs(b)) > max_index:
                continue
            amp = amplitude(m, n)
            if abs(amp) > 1e-13:
                orders[(a, b)] = amp
    orders[(0.0, 0.0)] = (
        2.0 * duty**2 if model == "enlarged" else (1.0 + (2.0 * duty - 1.0) ** 2) / 2.0
    )
    return dict(sorted(orders.items()))


def diffraction_efficiency(orders: dict) -> dict[str, float | dict]:
    """各衍射级的效率 |A|^2 及常用汇总。"""
    eff = {k: float(abs(v) ** 2) for k, v in orders.items()}
    dc = eff.get((0.0, 0.0), 0.0)
    first = sum(
        eff.get(k, 0.0) for k in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    )
    return {"dc": dc, "first_order_total": first, "all": eff}
