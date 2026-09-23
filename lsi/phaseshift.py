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

__all__ = [
    "PhaseShiftResult",
    "lsq_phase_shift",
    "find_pupil_circle",
    "circle_fit",
    "shear_regions",
]


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
    if n < 3:
        raise ValueError("相移解调至少需要 3 帧")
    if deltas is None:
        deltas = 2.0 * np.pi * np.arange(n) / n
    deltas = np.asarray(deltas, dtype=float)

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


# --------------------------------------------------------------------------- #
# 剪切区域判定（论文 3.1.1，式 3-1 ... 3-5）
# --------------------------------------------------------------------------- #
def circle_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Kasa 代数最小二乘圆拟合，返回 (cx, cy, r)。"""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    return float(cx), float(cy), float(np.sqrt(sol[2] + cx**2 + cy**2))


def find_pupil_circle(
    mod: np.ndarray, grid: Grid, threshold_frac: float = 0.4
) -> tuple[float, float, float]:
    """调制度图 -> 阈值 -> 最大连通域外缘 -> 圆拟合，得零级光瞳 (cx, cy, r)。"""
    from scipy import ndimage

    mask = mod > threshold_frac * mod.max()
    label, n_lab = ndimage.label(mask)
    sizes = ndimage.sum(mask, label, index=np.arange(1, n_lab + 1))
    blob = ndimage.binary_fill_holes(label == int(np.argmax(sizes)) + 1)
    edge = blob & ~ndimage.binary_erosion(blob)
    x, y = grid.coords()
    return circle_fit(x[edge], y[edge])


def shear_regions(
    grid: Grid,
    s: float,
    cx: float = 0.0,
    cy: float = 0.0,
    r: float = 1.0,
) -> dict[str, np.ndarray]:
    """四个剪切干涉区域（论文图 2-8/2-9）。

    x 剪切区 = 零级光瞳与 (±s, 0) 移位光瞳的三重交；"center5" 为
    中央五光束区（再交上 (0, ±s) 移位光瞳）。
    """
    x, y = grid.coords()

    def pupil(dx: float, dy: float) -> np.ndarray:
        return (x - cx - dx) ** 2 + (y - cy - dy) ** 2 <= r * r

    zero = pupil(0.0, 0.0)
    region_x = zero & pupil(s, 0.0) & pupil(-s, 0.0)
    region_y = zero & pupil(0.0, s) & pupil(0.0, -s)
    center5 = region_x & region_y
    return {
        "zero": zero,
        "region_x": region_x,
        "region_y": region_y,
        "center5": center5,
        "region_x_only": region_x & ~center5,
        "region_y_only": region_y & ~center5,
    }
