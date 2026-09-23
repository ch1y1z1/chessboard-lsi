"""端到端论文流程：

    I(x, y) --(相移 / 傅里叶变换)--> dW --(差分 Zernike 最小二乘)--> W

两条路线共用同一个重构级，只有 I 的解调不同。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .forward import ForwardModel
from .ftmode import LobeResult, demodulate_lobe
from .phaseshift import find_pupil_circle, lsq_phase_shift, shear_regions
from .reconstruct import ZernikeFit, fit_differential_zernike
from .unwrap import unwrap_poisson, wrap

__all__ = [
    "DiffPhase",
    "demodulate_phase_shift",
    "demodulate_fourier",
    "reconstruct",
    "phase_shift_to_wavefront",
    "fourier_to_wavefront",
]

DEFAULT_INDICES = tuple(range(2, 17))


@dataclass
class DiffPhase:
    """两个剪切方向的解调差分数据。

    ``dW`` 为差分波前（waves）；``mask`` 为剪切区域；``confidence`` 为
    路线相关的信号强度（相移调制度 / 载频瓣幅值），供加权重构；
    ``offset_removed`` 记录各方向是否已扣除光栅模型的常数相位。
    """

    dW: dict[str, np.ndarray]
    mask: dict[str, np.ndarray]
    confidence: dict[str, np.ndarray]
    offset_removed: dict[str, bool]
    phase: dict[str, np.ndarray] = field(default_factory=dict)
    wrapped_phase: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _require_symmetric_pair(fm: ForwardModel, direction: str) -> None:
    """two-sided 差分解读的物理前提：±1 级对都存在且与 0 级的 beat 等幅。

    解调出的频率-1 相位等于 pi * [W(x+s) - W(x-s)] 仅当两个对称 beat
    系数模相等；缺一边时实际是单边差分 2[W(x+s) - W(x)]，不能除以 pi。
    """
    if direction not in ("x", "y"):
        raise ValueError(f"direction 必须是 'x' 或 'y'，得到 {direction!r}")
    a, b = (1.0, 0.0) if direction == "x" else (0.0, 1.0)
    amps = dict(zip(fm.order_list, fm.amplitudes))
    ap, am = amps.get((a, b), 0.0), amps.get((-a, -b), 0.0)
    if amps.get((0.0, 0.0), 0.0) == 0.0:
        raise ValueError("缺少 (0,0) 级：无法构成 ±1 拍频")
    if ap == 0.0 or am == 0.0:
        raise ValueError(f"{direction} 方向缺少 ±1 级对的一边，无法按双边差分解释")
    if not np.isclose(abs(ap), abs(am), rtol=1e-3, atol=0.0):
        raise ValueError(
            f"±1 级振幅不对称：|A+| = {abs(ap):.4g}, |A-| = {abs(am):.4g}"
        )


def _unwrap_in_region(wrapped: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """掩膜内 Poisson 解包裹，并把相位锚定到掩膜中心像素的缠绕值。

    掩膜必须单连通：每个连通分量各自携带一个不可观测的 2pi 规范，
    多分量时它们之间的整数相位差会被重构当成真实像差。
    """
    from scipy import ndimage

    _, n_comp = ndimage.label(mask)
    if n_comp == 0:
        raise ValueError("解包裹掩膜为空")
    if n_comp > 1:
        raise ValueError(
            f"解包裹掩膜有 {n_comp} 个连通分量：分量间相位规范不可观测"
        )
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
    *,
    region_mode: str = "analytic",
    threshold_frac: float = 0.4,
    remove_offset: bool = True,
) -> DiffPhase:
    """N 步相移解调 + 剪切区提取，返回两个方向的 dW（waves）。

    ``region_mode="analytic"`` 用已知光瞳（单位圆）与 ±s 移位圆的交；
    ``"modulation"`` 走论文实验做法：调制度阈值 -> 边缘 -> 圆拟合 ->
    按剪切量平移（式 3-1 ... 3-5）。

    解调相位先减去光栅常数项再解包裹：理想棋盘的 x 对偏移恰为 pi，
    直接在 ±pi 分支切线附近解包裹会产生整帧 2 pi 抖动。
    """
    cfg = fm.config
    if region_mode not in ("analytic", "modulation"):
        raise ValueError(
            f"region_mode 必须是 'analytic'/'modulation'，得到 {region_mode!r}"
        )
    _require_symmetric_pair(fm, "x")
    _require_symmetric_pair(fm, "y")
    for name, frames in (("frames_x", frames_x), ("frames_y", frames_y)):
        if np.ndim(frames) != 3 or frames.shape[1:] != fm.shape:
            raise ValueError(
                f"{name} 形状必须是 (N, {fm.shape[0]}, {fm.shape[1]})，"
                f"得到 {np.shape(frames)}"
            )
    res = {"x": lsq_phase_shift(frames_x), "y": lsq_phase_shift(frames_y)}

    if region_mode == "modulation":
        cx, cy, r = find_pupil_circle(
            res["x"].modulation, cfg.grid, threshold_frac=threshold_frac
        )
        masks = shear_regions(cfg.grid, cfg.s, cx, cy, r)
        meta = {"circle": (cx, cy, r)}
    else:
        masks = shear_regions(cfg.grid, cfg.s)
        meta = {}

    dW, phase, wrapped_phase, mod = {}, {}, {}, {}
    for direction in ("x", "y"):
        mask = masks[f"region_{direction}"]
        offset = fm.demodulation_offset(direction)
        wrapped = wrap(res[direction].phase - offset)
        ph = _unwrap_in_region(wrapped, mask)
        if not remove_offset:
            ph = ph + offset   # 常数在解包裹后加回，避免选错缠绕分支
        dW[direction] = ph / np.pi          # 相位 = pi * 双边差分（waves）
        phase[direction] = ph
        wrapped_phase[direction] = wrapped
        mod[direction] = res[direction].modulation
    return DiffPhase(
        dW=dW,
        mask={"x": masks["region_x"], "y": masks["region_y"]},
        confidence=mod,
        offset_removed={"x": remove_offset, "y": remove_offset},
        phase=phase,
        wrapped_phase=wrapped_phase,
        meta={**meta, "route": "phase_shift", "n_steps": int(frames_x.shape[0])},
    )


def demodulate_fourier(
    fm: ForwardModel,
    image: np.ndarray,
    *,
    direction: str = "x",
    f0: float | None = None,
    threshold_frac: float = 0.5,
    erode_px: int = 4,
    remove_offset: bool = True,
    window_radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray, LobeResult]:
    """单帧载频解调一个方向，返回 (dW, mask, lobe)。

    mask = 载频瓣幅值超过峰值 threshold_frac 且落在物理剪切支撑
    （0 级与 ±1 级光瞳的交）内；``erode_px`` 把掩膜边缘腐蚀掉若干像素，
    去掉滤波核混入暗区造成的边界畸变。
    """
    from scipy import ndimage

    if direction not in ("x", "y"):
        raise ValueError(f"direction 必须是 'x' 或 'y'，得到 {direction!r}")
    if not 0.0 <= threshold_frac < 1.0:
        raise ValueError("threshold_frac 必须在 [0, 1) 内")
    _require_symmetric_pair(fm, direction)
    offset = fm.demodulation_offset(direction)
    lobe = demodulate_lobe(
        image, fm.grid, direction,
        f0=fm.config.carrier_f0 if f0 is None else f0,
        phase_offset=offset,
        window_radius=window_radius,
    )
    a, b = (1.0, 0.0) if direction == "x" else (0.0, 1.0)
    support = fm.order_support(0.0, 0.0) & fm.order_support(a, b) & fm.order_support(-a, -b)
    if not np.any(support):
        raise ValueError("剪切支撑为空：0 级与 ±1 级光瞳没有交集")
    mask = (lobe.amplitude > threshold_frac * lobe.amplitude[support].max()) & support
    if erode_px > 0:
        mask = ndimage.binary_erosion(mask, iterations=erode_px)
    if not mask.any():
        raise ValueError("掩膜为空：降低 threshold_frac 或 erode_px")
    phase = _unwrap_in_region(lobe.phase, mask)
    if not remove_offset:
        phase = phase + offset
    return phase / np.pi, mask, lobe


# --------------------------------------------------------------------------- #
def reconstruct(
    fm: ForwardModel,
    diff: DiffPhase,
    *,
    indices: Sequence[int] = DEFAULT_INDICES,
    offset_mode: str = "none",
    weight_by_confidence: bool = True,
    resolve_tilt_gauge: bool = True,
) -> ZernikeFit:
    """对解调差分数据做差分 Zernike 最小二乘（式 2-28 ... 2-31）。

    ``offset_mode``：
      ``"none"``     解调时已扣除模型常数（默认）；
      ``"model"``    在此减去模型预测常数（配合 remove_offset=False）；
      ``"estimate"`` 每方向估计一个自由常数（与 tilt 列共线，仅在振幅
                     未知时有意义）。

    ``resolve_tilt_gauge``：修正解包裹的整数波规范。解包裹把种子像素钉到
    其缠绕值，若真实相位距缠绕值超过半条纹，整张 dW 图差整数个波长；
    该常数被 tilt 列（dZx(2) = 2s 为常数）吸收，表现为虚假 tilt。
    开启后把拟合 tilt 乘列常数、四舍五入到整数 k，非零则扣除后重拟合。
    相当于先验 |Z2|, |Z3| < 1/(4s)；大 tilt 波前请关闭。
    """
    if offset_mode not in ("none", "model", "estimate"):
        raise ValueError(
            f"offset_mode 必须是 'none'/'model'/'estimate'，得到 {offset_mode!r}"
        )
    if offset_mode == "none" and not all(diff.offset_removed.values()):
        raise ValueError(
            "offset_mode='none' 但解调未扣除光栅常数：请用 'model' 或 'estimate'"
        )
    if offset_mode == "model" and all(diff.offset_removed.values()):
        raise ValueError(
            "offset_mode='model' 会重复扣除：DiffPhase 记录常数已移除"
        )
    x, y = fm.grid.coords()
    wx = wy = None
    if weight_by_confidence and diff.confidence:
        wx = np.clip(diff.confidence["x"], 1e-6, None)
        wy = np.clip(diff.confidence["y"], 1e-6, None)
    known = None
    if offset_mode == "model":
        known = {
            d: 0.0 if diff.offset_removed[d]
            else fm.demodulation_offset(d) / np.pi
            for d in ("x", "y")
        }

    def _fit(dwx, dwy):
        return fit_differential_zernike(
            dwx, dwy, diff.mask["x"], diff.mask["y"], fm.s, x, y,
            indices=indices,
            fit_offsets=(offset_mode == "estimate"),
            known_offsets=known,
            weights_x=wx, weights_y=wy,
        )

    fit = _fit(diff.dW["x"], diff.dW["y"])
    k_x = k_y = 0
    if resolve_tilt_gauge and offset_mode != "estimate":
        idx = list(indices)
        # 规范误差恰为整数个波长；小数部分远离整数说明常数来自真实
        # tilt 或噪声，此时取整会注入整波误差，故拒绝。
        if 2 in idx:
            est = fit.coeffs[idx.index(2)] * 2.0 * fm.s   # dZx(2) = 2s 为常数
            if abs(est - np.rint(est)) <= 0.25:
                k_x = int(np.rint(est))
        if 3 in idx:
            est = fit.coeffs[idx.index(3)] * 2.0 * fm.s   # dZy(3) = 2s
            if abs(est - np.rint(est)) <= 0.25:
                k_y = int(np.rint(est))
        if k_x or k_y:
            fit = _fit(diff.dW["x"] - k_x, diff.dW["y"] - k_y)
    diff.meta["tilt_gauge"] = {"k_x": k_x, "k_y": k_y}
    return fit


def phase_shift_to_wavefront(
    fm: ForwardModel,
    frames_x,
    frames_y,
    *,
    indices: Sequence[int] = DEFAULT_INDICES,
    region_mode: str = "analytic",
    remove_offset: bool = True,
    offset_mode: str = "none",
    resolve_tilt_gauge: bool = True,
    weight_by_confidence: bool = True,
) -> tuple[ZernikeFit, DiffPhase]:
    diff = demodulate_phase_shift(
        fm, frames_x, frames_y,
        region_mode=region_mode, remove_offset=remove_offset,
    )
    fit = reconstruct(
        fm, diff, indices=indices, offset_mode=offset_mode,
        resolve_tilt_gauge=resolve_tilt_gauge,
        weight_by_confidence=weight_by_confidence,
    )
    return fit, diff


def fourier_to_wavefront(
    fm: ForwardModel,
    image,
    *,
    indices: Sequence[int] = DEFAULT_INDICES,
    remove_offset: bool = True,
    offset_mode: str = "none",
    resolve_tilt_gauge: bool = True,
    weight_by_confidence: bool = True,
    **demod_kwargs,
) -> tuple[ZernikeFit, DiffPhase]:
    dWx, mask_x, lobe_x = demodulate_fourier(
        fm, image, direction="x", remove_offset=remove_offset, **demod_kwargs
    )
    dWy, mask_y, lobe_y = demodulate_fourier(
        fm, image, direction="y", remove_offset=remove_offset, **demod_kwargs
    )
    diff = DiffPhase(
        dW={"x": dWx, "y": dWy},
        mask={"x": mask_x, "y": mask_y},
        confidence={"x": lobe_x.amplitude, "y": lobe_y.amplitude},
        offset_removed={"x": remove_offset, "y": remove_offset},
        # 解包裹相位（弧度）= dW * pi
        phase={"x": dWx * np.pi, "y": dWy * np.pi},
        wrapped_phase={"x": lobe_x.phase, "y": lobe_y.phase},
        meta={"route": "fourier", "lobes": {"x": lobe_x, "y": lobe_y}},
    )
    fit = reconstruct(
        fm, diff, indices=indices, offset_mode=offset_mode,
        resolve_tilt_gauge=resolve_tilt_gauge,
        weight_by_confidence=weight_by_confidence,
    )
    return fit, diff
