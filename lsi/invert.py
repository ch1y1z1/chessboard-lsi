"""论文反演链：I(x, y) --(相移 / 傅里叶变换)--> dW --(差分 Zernike)--> W。

按计算链条自上而下：

    1. 相移解调     N 步闭式 psi = atan2(-Σ I sinδ, Σ I cosδ)（式 2-20）
    2. 载频解调     二维 FFT 提取 +f0 瓣 -> arg c(x,y)（2.4 节）
    3. 解包裹       掩膜内 Poisson 最小二乘（Ghiglia–Romero）
    4. 差分重构     ΔW = ΔZ c 最小二乘（式 2-28~2-31）
    5. 端到端       phase_shift_to_wavefront / fourier_to_wavefront

物理前提（本实现依赖，不运行时检查）：
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
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

from .model import ForwardModel, Grid, differential_zernike_matrix, wavefront


# --------------------------------------------------------------------------- #
# 1. N 步相移解调与剪切区（论文 2.3.3 / 3.1.1）
# --------------------------------------------------------------------------- #
def lsq_phase_shift(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """N 步均匀相移解调，返回 (缠绕相位 psi, 调制度 M)。

    第 i 步相移 delta_i = 2 pi i / N，cos/sin 基正交，式 (2-20) 退化为闭式

        C = (2/N) sum_i I_i cos(delta_i),   S = (2/N) sum_i I_i sin(delta_i)
        psi = atan2(-S, C),   M = hypot(C, S)
    """
    frames = np.asarray(frames, dtype=float)
    n = frames.shape[0]
    deltas = 2.0 * np.pi * np.arange(n) / n
    c_cos = 2.0 / n * np.tensordot(np.cos(deltas), frames, axes=1)
    s_sin = 2.0 / n * np.tensordot(np.sin(deltas), frames, axes=1)
    return np.arctan2(-s_sin, c_cos), np.hypot(c_cos, s_sin)


def shear_regions(grid: Grid, s: float) -> dict[str, np.ndarray]:
    """两个方向的剪切干涉区（论文图 2-8/2-9），光瞳取单位圆。

    x 剪切区 = 零级光瞳与 (±s, 0) 移位光瞳的三重交；y 同理。
    """
    x, y = grid.coords()

    def pupil(dx: float, dy: float) -> np.ndarray:
        return (x - dx) ** 2 + (y - dy) ** 2 <= 1.0

    return {
        "x": pupil(0.0, 0.0) & pupil(s, 0.0) & pupil(-s, 0.0),
        "y": pupil(0.0, 0.0) & pupil(0.0, s) & pupil(0.0, -s),
    }


# --------------------------------------------------------------------------- #
# 2. 单帧载频（傅里叶变换模式）解调（论文 2.4）
# --------------------------------------------------------------------------- #
# 离焦引入空间载频 f0（f0 = m/2s），使 ±1 级干涉项移到 (±f0, 0) 与 (0, ±f0)。
# 处理流程（论文 2.4.3）：提取 +f0 瓣 -> 移回基带 -> arg -> 差分相位。
#
# 这里用"逐像素去载频 + 带限低通"实现移频：c = IFFT( FFT(I e^{-i 2 pi f0 u})
# * W )。+f0 瓣含 E_+ E_0* 与 E_0 E_-* 两个拍频，对对称棋盘系数合成为
#
#     C_{+f0} = 2 A_0 A_1 cos(Gamma) * exp{ i pi [W(x+s,y) - W(x-s,y)] }
#
# 即论文的双边差分（式 2-42），条件是 cos(Gamma) 不变号（2.4.2 的调制度约束）。

#: 低通滤波半径 = 4L（L 为网格宽度，频谱像素）；幅值掩膜阈值与边缘腐蚀量。
_LOBE_WINDOW_RADIUS_FACTOR = 4.0
_LOBE_MASK_THRESHOLD = 0.5
_LOBE_MASK_ERODE_PX = 4


def demodulate_lobe(
    image: np.ndarray,
    grid: Grid,
    direction: str,
    f0: float,
    phase_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    """解调单帧干涉图某方向的 +f0 载频瓣，返回 (缠绕相位, 幅值)。

    ``direction="x"`` 取 (+f0, 0) 瓣，``"y"`` 取 (0, +f0) 瓣。
    ``phase_offset`` 为光栅模型给出的常数相位（理想棋盘 x 方向为 pi），
    在取辐角前扣除，避免相位正好压在 ±pi 分支切线上。
    """
    I = np.asarray(image, dtype=float)
    x, y = grid.coords()
    n, L = grid.n, 2.0 * grid.extent
    ramp = x if direction == "x" else y

    # 逐像素去载频 -> 基带含 0 级谱与 +f0 瓣内容 -> Butterworth 低通
    baseband = np.fft.fftshift(np.fft.fft2(I * np.exp(-2j * np.pi * f0 * ramp)))
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    rr = np.hypot(xx - n // 2, yy - n // 2)
    win = 1.0 / (1.0 + (rr / (_LOBE_WINDOW_RADIUS_FACTOR * L)) ** 4)
    c = np.fft.ifft2(np.fft.ifftshift(baseband * win))
    if phase_offset:
        c = c * np.exp(-1j * phase_offset)
    return np.angle(c), np.abs(c)


# --------------------------------------------------------------------------- #
# 3. 掩膜内最小二乘（Poisson）相位解包裹
# --------------------------------------------------------------------------- #
# Ghiglia–Romero 方法：在掩膜内求 psi 使其梯度在最小二乘意义下等于缠绕
# 相位的缠绕梯度，即解离散 Poisson 方程
#
#     sum_{掩膜内4邻域 j} (psi_i - psi_j) = -rho_i
#
# rho 为缠绕梯度场的散度。掩膜单连通，解带一个自由常数，把最靠近中心的
# 像素钉到其缠绕值上。掩膜外为 NaN。
def wrap(x: np.ndarray) -> np.ndarray:
    """缠绕到 (-pi, pi]。"""
    return np.angle(np.exp(1j * np.asarray(x, dtype=float)))


def _masked_divergence(phi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """缠绕梯度场的散度：每条两端都在掩膜内的边贡献 ±g。"""
    rho = np.zeros_like(phi)
    for di, dj in ((0, 1), (1, 0)):
        shifted_phi = np.roll(phi, (-di, -dj), axis=(0, 1))
        valid = mask & np.roll(mask, (-di, -dj), axis=(0, 1))
        # np.roll 是周期性的：网格最后一行/列的"邻居"是回绕来的，需排除
        if di:
            valid[-1, :] = False
        if dj:
            valid[:, -1] = False
        g = np.where(valid, wrap(shifted_phi - phi), 0.0)
        rho += g
        rho -= np.roll(g, (di, dj), axis=(0, 1))
    return rho


def unwrap_poisson(phi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """掩膜内最小二乘解包裹；掩膜外为 NaN。掩膜须单连通。"""
    phi = np.asarray(phi, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    out = np.full(phi.shape, np.nan)

    rho = _masked_divergence(phi, mask)
    ys, xs = np.nonzero(mask)
    index = -np.ones(phi.shape, dtype=np.int64)
    index[ys, xs] = np.arange(len(ys))

    # 掩膜拉普拉斯：对角 = 掩膜内邻居数，非对角 = -1
    n_pix = len(ys)
    rows, cols, vals = [], [], []
    for k, (i, j) in enumerate(zip(ys, xs)):
        n_nb = 0
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < phi.shape[0] and 0 <= jj < phi.shape[1] and mask[ii, jj]:
                rows.append(k)
                cols.append(int(index[ii, jj]))
                vals.append(-1.0)
                n_nb += 1
        rows.append(k)
        cols.append(k)
        vals.append(float(n_nb))
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_pix, n_pix))
    rhs = -rho[ys, xs]

    # 钉住最靠中心的像素到其缠绕值，固定解的常数项
    k = int(np.argmin(
        (ys - phi.shape[0] / 2.0) ** 2 + (xs - phi.shape[1] / 2.0) ** 2
    ))
    A = A.tolil()
    A[k, :] = 0.0
    A[k, k] = 1.0
    rhs[k] = phi[ys[k], xs[k]]

    out[ys, xs] = spsolve(A.tocsr(), rhs)
    return out


def _unwrap_in_region(wrapped: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """掩膜内 Poisson 解包裹，并把相位锚定到掩膜中心像素的缠绕值。"""
    phase = unwrap_poisson(wrapped, mask)
    ys, xs = np.nonzero(mask)
    k = int(np.argmin(
        (ys - wrapped.shape[0] / 2.0) ** 2 + (xs - wrapped.shape[1] / 2.0) ** 2
    ))
    return phase + (wrapped[ys[k], xs[k]] - phase[ys[k], xs[k]])


# --------------------------------------------------------------------------- #
# 4. 差分 Zernike 最小二乘重构（论文式 2-28 ... 2-31）
# --------------------------------------------------------------------------- #
@dataclass
class ZernikeFit:
    indices: np.ndarray
    coeffs: np.ndarray                     # 拟合系数（waves）
    rms_residual: float = 0.0
    cond: float = np.nan

    def as_dict(self) -> dict[int, float]:
        return {int(j): float(c) for j, c in zip(self.indices, self.coeffs)}


def fit_differential_zernike(
    dW_x: np.ndarray,
    dW_y: np.ndarray,
    mask_x: np.ndarray,
    mask_y: np.ndarray,
    s: float,
    x: np.ndarray,
    y: np.ndarray,
    *,
    indices: Sequence[int] = tuple(range(2, 17)),
    weights_x: np.ndarray | None = None,
    weights_y: np.ndarray | None = None,
) -> ZernikeFit:
    """最小二乘差分 Zernike 重构（式 2-28 ... 2-31）。

    测得的两个方向差分波前
        dW_x = W(x+s, y) - W(x-s, y),   dW_y = W(x, y+s) - W(x, y-s)
    展开为差分 Zernike 基的线性组合 ΔW = ΔZ c，解 c = (ΔZ^T ΔZ)^{-1} ΔZ^T ΔW。
    ``weights`` 为逐像素权重（如调制度），用于加权最小二乘。
    Z1（平移）没有差分信号，不应出现在 ``indices`` 中。
    """
    rows, rhs, wts = [], [], []
    for direction, d, mask, wgt in (
        ("x", dW_x, mask_x, weights_x),
        ("y", dW_y, mask_y, weights_y),
    ):
        mask = np.asarray(mask, dtype=bool) & np.isfinite(d)
        Z = differential_zernike_matrix(indices, x[mask], y[mask], s, direction)
        rows.append(Z)
        rhs.append(np.asarray(d, dtype=float)[mask])
        wts.append(np.sqrt(np.clip(wgt[mask], 0.0, None)) if wgt is not None
                   else np.ones(int(mask.sum())))
    A = np.vstack(rows)
    b = np.concatenate(rhs)
    w = np.concatenate(wts)

    sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    residual = A @ sol - b
    return ZernikeFit(
        indices=np.asarray(indices),
        coeffs=sol,
        rms_residual=float(np.sqrt(np.mean(residual**2))),
        cond=float(np.linalg.cond(A * w[:, None])),
    )


def wavefront_on_grid(
    coeffs: Sequence[float],
    indices: Sequence[int],
    x: np.ndarray,
    y: np.ndarray,
    pupil: np.ndarray | None = None,
) -> np.ndarray:
    W = wavefront(coeffs, indices, x, y)
    return np.where(pupil, W, np.nan) if pupil is not None else W


# --------------------------------------------------------------------------- #
# 5. 端到端路线：解调 -> 解包裹 -> 重构
# --------------------------------------------------------------------------- #
@dataclass
class DiffPhase:
    """两个剪切方向的解调差分数据。

    ``dW`` 为差分波前（waves）；``mask`` 为剪切区域；``confidence`` 为
    路线相关的信号强度（相移调制度 / 载频瓣幅值），供加权重构。
    ``phase``/``wrapped_phase`` 为画图用的中间量（弧度）。
    """

    dW: dict[str, np.ndarray]
    mask: dict[str, np.ndarray]
    confidence: dict[str, np.ndarray]
    phase: dict[str, np.ndarray] = field(default_factory=dict)
    wrapped_phase: dict[str, np.ndarray] = field(default_factory=dict)


def _demodulate_phase_shift(
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
        wrapped = wrap(res[direction][0] - offset)
        ph = _unwrap_in_region(wrapped, mask)
        dW[direction] = ph / np.pi          # 相位 = pi * 双边差分（waves）
        phase[direction] = ph
        wrapped_phase[direction] = wrapped
        mod[direction] = res[direction][1]
    return DiffPhase(
        dW=dW,
        mask={"x": masks["x"], "y": masks["y"]},
        confidence=mod,
        phase=phase,
        wrapped_phase=wrapped_phase,
    )


def _demodulate_fourier(
    fm: ForwardModel,
    image: np.ndarray,
    *,
    direction: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """单帧载频解调一个方向，返回 (dW, mask, wrapped_phase, amplitude)。

    mask = 载频瓣幅值超过峰值阈值且落在物理剪切支撑（0 级与 ±1 级光瞳
    的交）内；边缘腐蚀若干像素，去掉滤波核混入暗区造成的边界畸变。
    """
    offset = fm.demodulation_offset(direction)
    wrapped, amplitude = demodulate_lobe(
        image, fm.grid, direction, fm.config.carrier_f0, phase_offset=offset
    )
    a, b = (1.0, 0.0) if direction == "x" else (0.0, 1.0)
    support = fm.order_support(0.0, 0.0) & fm.order_support(a, b) & fm.order_support(-a, -b)
    mask = (amplitude > _LOBE_MASK_THRESHOLD * amplitude[support].max()) & support
    mask = ndimage.binary_erosion(mask, iterations=_LOBE_MASK_ERODE_PX)
    phase = _unwrap_in_region(wrapped, mask)
    return phase / np.pi, mask, wrapped, amplitude


def _reconstruct(
    fm: ForwardModel,
    diff: DiffPhase,
    indices: Sequence[int],
) -> ZernikeFit:
    """对解调差分数据做调制度加权的差分 Zernike 最小二乘。"""
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
    frames_x: np.ndarray,
    frames_y: np.ndarray,
    *,
    indices: Sequence[int] = tuple(range(2, 17)),
) -> tuple[ZernikeFit, DiffPhase]:
    """路线 A：相移序列 -> 解调 -> 解包裹 -> 差分 Zernike 重构。"""
    diff = _demodulate_phase_shift(fm, frames_x, frames_y)
    return _reconstruct(fm, diff, indices), diff


def fourier_to_wavefront(
    fm: ForwardModel,
    image: np.ndarray,
    *,
    indices: Sequence[int] = tuple(range(2, 17)),
) -> tuple[ZernikeFit, DiffPhase]:
    """路线 B：单帧载频图 -> 瓣解调 -> 解包裹 -> 差分 Zernike 重构。"""
    dWx, mask_x, wrapped_x, amp_x = _demodulate_fourier(fm, image, direction="x")
    dWy, mask_y, wrapped_y, amp_y = _demodulate_fourier(fm, image, direction="y")
    diff = DiffPhase(
        dW={"x": dWx, "y": dWy},
        mask={"x": mask_x, "y": mask_y},
        confidence={"x": amp_x, "y": amp_y},
        phase={"x": dWx * np.pi, "y": dWy * np.pi},
        wrapped_phase={"x": wrapped_x, "y": wrapped_y},
    )
    return _reconstruct(fm, diff, indices), diff
