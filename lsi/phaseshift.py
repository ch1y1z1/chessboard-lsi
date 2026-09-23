"""相移干涉图的相位解算与剪切区域判定（论文 2.3.3 / 3.1.1）。

N 步最小二乘相移（论文式 2-20）：逐像素拟合

    I_i = B + C cos(delta_i) + S sin(delta_i)
    psi = atan2(-S, C)     M = hypot(C, S)     B = 截距

均匀步长 delta_i = 2 pi i / N 时 cos/sin 基正交，退化为闭式

    psi = atan2( -sum_i I_i sin delta_i , sum_i I_i cos delta_i )

这里直接用线性最小二乘，因此非标定步长（deltas 参数）也严格成立。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Grid


@dataclass
class PhaseShiftResult:
    phase: np.ndarray        # 缠绕相位 (rad)，[-pi, pi)
    modulation: np.ndarray   # 调制度 M（论文式 3-1）
    background: np.ndarray   # 背景项 B


def lsq_phase_shift(
    frames: np.ndarray, deltas: np.ndarray | None = None
) -> PhaseShiftResult:
    """N 步最小二乘相移解调；``deltas`` 缺省为 2 pi i / N。"""
    frames = np.asarray(frames, dtype=float)
    n = frames.shape[0]
    if deltas is None:
        deltas = 2.0 * np.pi * np.arange(n) / n
    design = np.column_stack([np.ones(n), np.cos(deltas), np.sin(deltas)])
    coef = np.linalg.pinv(design) @ frames.reshape(n, -1)
    shape = frames.shape[1:]
    background = coef[0].reshape(shape)
    c_cos = coef[1].reshape(shape)
    s_sin = coef[2].reshape(shape)
    return PhaseShiftResult(
        phase=np.arctan2(-s_sin, c_cos),
        modulation=np.hypot(c_cos, s_sin),
        background=background,
    )


def shear_regions(grid: Grid, s: float) -> dict[str, np.ndarray]:
    """四个剪切干涉区域（论文图 2-8/2-9），光瞳取单位圆。

    x 剪切区 = 零级光瞳与 (±s, 0) 移位光瞳的三重交；"center5" 为
    中央五光束区（再交上 (0, ±s) 移位光瞳）。
    """
    x, y = grid.coords()

    def pupil(dx: float, dy: float) -> np.ndarray:
        return (x - dx) ** 2 + (y - dy) ** 2 <= 1.0

    region_x = pupil(0.0, 0.0) & pupil(s, 0.0) & pupil(-s, 0.0)
    region_y = pupil(0.0, 0.0) & pupil(0.0, s) & pupil(0.0, -s)
    return {"x": region_x, "y": region_y, "center5": region_x & region_y}
