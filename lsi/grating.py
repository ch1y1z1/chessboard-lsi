"""45° 旋转振幅棋盘光栅的衍射级次振幅（论文 2.1，表 2-3）。

探测器坐标 (a, b) 与光栅坐标 (m, n) 的关系（45° 旋转）：

    a = (m + n) / 2,   b = (n - m) / 2       <=>   m = a - b, n = a + b

理想 50% 占空比：只有 m、n 同为奇数的级次非零，

    A_00 = 1/2,   A_mn = -2 / (pi^2 m n)   (m, n 均为奇数)

即 |A_±1| = 2/pi^2 = 0.2026（效率 4.11%），Parseval: sum |A|^2 = 1/2。
"""

from __future__ import annotations

import numpy as np


def chessboard_orders(max_index: int = 3) -> dict[tuple[float, float], complex]:
    """探测器坐标下的级次振幅表 ``{(a, b): A_ab}``（含 (0, 0) 级）。

    保留 max(|a|, |b|) <= max_index 的级次。
    """
    orders: dict[tuple[float, float], complex] = {}
    for m in range(-2 * max_index, 2 * max_index + 1):
        for n in range(-2 * max_index, 2 * max_index + 1):
            if m % 2 == 0 or n % 2 == 0:
                continue
            a, b = (m + n) / 2.0, (n - m) / 2.0
            if max(abs(a), abs(b)) <= max_index:
                orders[(a, b)] = -2.0 / (np.pi**2 * m * n)
    orders[(0.0, 0.0)] = 0.5
    return dict(sorted(orders.items()))


def diffraction_efficiency(orders: dict) -> dict[str, float | dict]:
    """各衍射级的效率 |A|^2 及常用汇总。"""
    eff = {k: float(abs(v) ** 2) for k, v in orders.items()}
    first = sum(
        eff.get(k, 0.0) for k in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    )
    return {"dc": eff[(0.0, 0.0)], "first_order_total": first, "all": eff}
