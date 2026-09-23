"""端到端论文流程：

    I(x, y) --(相移 / 傅里叶变换)--> dW --(差分 Zernike 最小二乘)--> W

两条路线共用同一个重构级，只有 I 的解调不同。

物理前提（本实现依赖、不再运行时检查）：
  * 0 级与 ±1 级对称成对且等幅 —— 解调出的频率-1 相位才等于
    pi * [W(x+s) - W(x-s)] 的双边差分（缺一边时是单边差分 2[W(x+s)-W(x)]）；
  * 解包裹掩膜单连通 —— 多个连通分量各自携带不可观测的 2pi 规范；
  * 波前足够小（演示用 <= ~2 波长）—— 否则缠绕相位出现涡旋，
    调制度 cos(Gamma) 在剪切区内变号（论文 2.3.2）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .forward import ForwardModel
from .ftmode import LobeResult, demodulate_lobe
from .phaseshift import lsq_phase_shift, shear_regions
from .reconstruct import ZernikeFit, fit_differential_zernike
from .unwrap import unwrap_poisson, wrap

DEFAULT_INDICES = tuple(range(2, 17))


@dataclass
class DiffPhase:
    """两个剪切方向的解调差分数据。

    ``dW`` 为差分波前（waves）；``mask`` 为剪切区域；``confidence`` 为
    路线相关的信号强度（相移调制度 / 载频瓣幅值），供加权重构。
    """

    dW: dict[str, np.ndarray]
    mask: dict[str, np.ndarray]
    confidence: dict[str, np.ndarray]
    phase: dict[str, np.ndarray] = field(default_factory=dict)
    wrapped_phase: dict[str, np.ndarray] = field(default_factory=dict)


def _unwrap_in_region(wrapped: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """掩膜内 Poisson 解包裹，并把相位锚定到掩膜中心像素的缠绕值。"""
    phase = unwrap_poisson(wrapped, mask)
    ys, xs = np.nonzero(mask)
    k = int(np.argmin(
        (ys - wrapped.shape[0] / 2.0) ** 2 + (xs - wrapped.shape[1] / 2.0) ** 2
    ))
    return phase + (wrapped[ys[k], xs[k]] - phase[ys[k], xs[k]])


# --------------------------------------------------------------------------- #
def demodulate_phase_shift(
    fm: ForwardModel,
    frames_x: np.ndarray,
    frames_y: np.ndarray,
) -> DiffPhase:
    """N 步相移解调 + 剪切区提取，返回两个方向的 dW（waves）。

    解调相位先减去光栅常数项再解包裹：理想棋盘的 x 对偏移恰为 pi，
    直接在 ±pi 分支切线附近解包裹会产生整帧 2 pi 抖动。
    """
    res = {"x": lsq_phase_shift(frames_x), "y": lsq_phase_shift(frames_y)}
    masks = shear_regions(fm.grid, fm.s)

    dW, phase, wrapped_phase, mod = {}, {}, {}, {}
    for direction in ("x", "y"):
        mask = masks[direction]
        offset = fm.demodulation_offset(direction)
        wrapped = wrap(res[direction].phase - offset)
        ph = _unwrap_in_region(wrapped, mask)
        dW[direction] = ph / np.pi          # 相位 = pi * 双边差分（waves）
        phase[direction] = ph
        wrapped_phase[direction] = wrapped
        mod[direction] = res[direction].modulation
    return DiffPhase(
        dW=dW,
        mask={"x": masks["x"], "y": masks["y"]},
        confidence=mod,
        phase=phase,
        wrapped_phase=wrapped_phase,
    )


def demodulate_fourier(
    fm: ForwardModel,
    image: np.ndarray,
    *,
    direction: str = "x",
    f0: float | None = None,
    threshold_frac: float = 0.5,
    erode_px: int = 4,
    window_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray, LobeResult]:
    """单帧载频解调一个方向，返回 (dW, mask, lobe)。

    mask = 载频瓣幅值超过峰值 threshold_frac 且落在物理剪切支撑
    （0 级与 ±1 级光瞳的交）内；``erode_px`` 把掩膜边缘腐蚀掉若干像素，
    去掉滤波核混入暗区造成的边界畸变。
    """
    from scipy import ndimage

    offset = fm.demodulation_offset(direction)
    lobe = demodulate_lobe(
        image, fm.grid, direction,
        f0=fm.config.carrier_f0 if f0 is None else f0,
        phase_offset=offset,
        window_radius=window_radius,
    )
    a, b = (1.0, 0.0) if direction == "x" else (0.0, 1.0)
    support = fm.order_support(0.0, 0.0) & fm.order_support(a, b) & fm.order_support(-a, -b)
    mask = (lobe.amplitude > threshold_frac * lobe.amplitude[support].max()) & support
    if erode_px > 0:
        mask = ndimage.binary_erosion(mask, iterations=erode_px)
    phase = _unwrap_in_region(lobe.phase, mask)
    return phase / np.pi, mask, lobe


# --------------------------------------------------------------------------- #
def reconstruct(
    fm: ForwardModel,
    diff: DiffPhase,
    *,
    indices: Sequence[int] = DEFAULT_INDICES,
) -> ZernikeFit:
    """对解调差分数据做调制度加权的差分 Zernike 最小二乘（式 2-28 ... 2-31）。"""
    x, y = fm.grid.coords()
    return fit_differential_zernike(
        diff.dW["x"], diff.dW["y"], diff.mask["x"], diff.mask["y"],
        fm.s, x, y,
        indices=indices,
        weights_x=np.clip(diff.confidence["x"], 1e-6, None),
        weights_y=np.clip(diff.confidence["y"], 1e-6, None),
    )


def phase_shift_to_wavefront(
    fm: ForwardModel,
    frames_x,
    frames_y,
    *,
    indices: Sequence[int] = DEFAULT_INDICES,
) -> tuple[ZernikeFit, DiffPhase]:
    diff = demodulate_phase_shift(fm, frames_x, frames_y)
    return reconstruct(fm, diff, indices=indices), diff


def fourier_to_wavefront(
    fm: ForwardModel,
    image,
    *,
    indices: Sequence[int] = DEFAULT_INDICES,
    **demod_kwargs,
) -> tuple[ZernikeFit, DiffPhase]:
    dWx, mask_x, lobe_x = demodulate_fourier(fm, image, direction="x", **demod_kwargs)
    dWy, mask_y, lobe_y = demodulate_fourier(fm, image, direction="y", **demod_kwargs)
    diff = DiffPhase(
        dW={"x": dWx, "y": dWy},
        mask={"x": mask_x, "y": mask_y},
        confidence={"x": lobe_x.amplitude, "y": lobe_y.amplitude},
        phase={"x": dWx * np.pi, "y": dWy * np.pi},
        wrapped_phase={"x": lobe_x.phase, "y": lobe_y.phase},
    )
    return reconstruct(fm, diff, indices=indices), diff
