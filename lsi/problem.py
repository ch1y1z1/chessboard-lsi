"""单帧无调制反问题。观测与前向模型独立于求解器，不接收真值。"""
from __future__ import annotations

from time import perf_counter
from typing import Sequence

import numpy as np

from .lm import _frame_and_jacobian
from .model import ForwardModel, zernike_matrix

try:  # T3 原生伴随梯度接口；未整合时回退到显式 J 路径（如实标记）
    from .gradient import frame_intensity_jtr as _frame_intensity_jtr
except ImportError:
    _frame_intensity_jtr = None


class IntensityProblem:
    """F(q)=1/2 mean((I(q)-image)^2), q_j=std_pupil(Z_j)*c_j。

    返回已除以 sqrt(M) 的残差和雅可比。mask 只允许固定的几何/仪器掩膜。
    scaling='none' 可用于原始系数坐标消融。全部计算采用 float64。

    ``value_gradient(q)`` 返回 (loss, g_q)：对所有有效像素/帧以 M 归一化，
    按 q_j=scale_j*c_j 链式变换。梯度后端见 ``gradient_backend``；
    ``supports_native_gradient`` 为 False 时走显式 J 兼容路径，调用方须把
    一次 value_gradient 记作一次前向 + 一次完整雅可比，不能冒充 native。
    """

    def __init__(
        self, forward: ForwardModel, indices: Sequence[int], image: np.ndarray,
        *, mask: np.ndarray | None = None, scaling: str = "rms",
    ):
        self._init_geometry(forward, indices, mask, scaling)
        image = np.asarray(image, dtype=float)
        if image.shape != forward.shape:
            raise ValueError("image 必须是一帧，形状与探测器一致")
        self.observed = image.ravel()[self.rows].copy()
        if not np.isfinite(self.observed).all():
            raise ValueError("有效像素必须有限；坏点请用固定 mask 排除")
        self.normalizer = np.sqrt(self.rows.size)

    def _init_geometry(self, forward, indices, mask, scaling):
        started = perf_counter()
        self.forward = forward
        raw_indices = np.asarray(indices)
        if (raw_indices.ndim != 1 or not raw_indices.size
                or not np.issubdtype(raw_indices.dtype, np.integer)):
            raise ValueError("indices 必须是非空整数序列")
        self.indices = raw_indices.astype(int, copy=True)
        if 1 in self.indices or len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("排除不可观测的 Z1，且 indices 不得重复")
        if mask is None:
            mask = np.ones(forward.shape, dtype=bool)
        mask = np.asarray(mask)
        if mask.shape != forward.shape or mask.dtype != bool or not mask.any():
            raise ValueError("mask 必须是非空且形状匹配的布尔掩膜")
        self.rows = np.flatnonzero(mask)
        self.cache = forward.zernike_samples(self.indices, self.rows)
        x, y = forward.grid.coords()
        pupil = forward.grid.pupil()
        self.pupil_basis = zernike_matrix(self.indices, x[pupil], y[pupil])
        if scaling not in ("rms", "none"):
            raise ValueError("scaling 必须是 rms 或 none")
        self.scale = (self.pupil_basis.std(axis=0) if scaling == "rms"
                      else np.ones(len(self.indices)))
        if np.any(self.scale <= 0):
            raise ValueError("网格不足以解析所选模式")
        self.supports_native_gradient = _frame_intensity_jtr is not None
        self.gradient_backend = (
            {"name": "native", "function": "lsi.gradient.frame_intensity_jtr",
             "explicit_jacobian": False}
            if self.supports_native_gradient else
            {"name": "explicit_jacobian_fallback",
             "function": "lsi.lm._frame_and_jacobian + J^T r",
             "explicit_jacobian": True}
        )
        self.precompute_seconds = perf_counter() - started

    def _frame_jtr(self, coeffs, target, modulation):
        """一帧的 (I, (dI/dc)^T (I - target))，原始未归一化 coeff 坐标。

        优先走 T3 的 ``lsi.gradient.frame_intensity_jtr``；未整合时用显式
        雅可比回退——此时一次调用内含一次完整 J 计算，成本归口见
        ``gradient_backend["explicit_jacobian"]``。
        """
        if _frame_intensity_jtr is not None:
            return _frame_intensity_jtr(self.cache, coeffs, target, modulation)
        intensity, jac = _frame_and_jacobian(
            self.cache, coeffs, modulation, compute_jacobian=True)
        return intensity, jac.T @ (intensity - target)

    def value_gradient(self, q):
        """返回 (loss, g_q)。loss = 1/2 * mean((I-I_obs)^2)，g_q = ∂loss/∂q。

        M = 有效像素总数（所有帧合计），即 normalizer**2；
        g_q = (dI/dc)^T (I-I_obs) / (M * scale)。
        """
        c = self.to_coefficients(q)
        intensity, jtr = self._frame_jtr(c, self.observed, None)
        m = float(self.rows.size)
        return (float(np.sum((intensity - self.observed) ** 2) / (2.0 * m)),
                jtr / (m * self.scale))

    def _vector(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != self.indices.shape or not np.isfinite(values).all():
            raise ValueError("系数必须有限且与 indices 形状一致")
        return values

    def to_parameters(self, coeffs):
        return self._vector(coeffs) * self.scale

    def to_coefficients(self, q):
        return self._vector(q) / self.scale

    def evaluate(self, q, *, jacobian=True):
        """返回归一化残差及可选雅可比；不改变观测，不施加额外相位。"""
        intensity, jac = _frame_and_jacobian(
            self.cache, self.to_coefficients(q), None,
            compute_jacobian=jacobian,
        )
        residual = (intensity - self.observed) / self.normalizer
        if jac is not None:
            jac = jac / (self.normalizer * self.scale[None, :])
        return residual, jac


class StackedIntensityProblem(IntensityProblem):
    """多帧联合反问题：每帧携带已知的附加相位（光栅相移 / 空间载频）。

    F(q) = 1/2 mean over all pixels and frames of (I(q)-I_obs)^2；
    所有帧共用同一像素集 rows 与 Zernike 缓存，残差及雅可比按帧堆叠。
    modulations 逐帧给定：(n_orders,) 的每级标量相移，
    或 (n_orders, n, n) / (n_orders, n*n) 的空间相位（按 rows 采样）。
    """

    def __init__(
        self, forward: ForwardModel, indices: Sequence[int],
        frames: Sequence[np.ndarray],
        modulations: Sequence[np.ndarray | None] | None = None,
        *, mask: np.ndarray | None = None, scaling: str = "rms",
    ):
        self._init_geometry(forward, indices, mask, scaling)
        frames = [np.asarray(f, dtype=float) for f in frames]
        n_orders = len(forward.order_list)
        if not frames or any(f.shape != forward.shape for f in frames):
            raise ValueError("frames 必须非空且每帧与探测器形状一致")
        if modulations is None:
            modulations = [None] * len(frames)
        if len(modulations) != len(frames):
            raise ValueError("modulations 必须与 frames 一一对应")
        self.mods = []
        for m in modulations:
            if m is None:
                self.mods.append(None)
                continue
            m = np.asarray(m, dtype=float)
            if m.ndim == 1:
                if m.size != n_orders:
                    raise ValueError("标量调制长度必须等于衍射级数")
                self.mods.append(m)
            elif m.shape[0] == n_orders and m.size == n_orders * self.forward.shape[0] ** 2:
                self.mods.append(m.reshape(n_orders, -1)[:, self.rows])
            else:
                raise ValueError("调制必须是 (n_orders,) 或 (n_orders, n, n)")
        blocks = np.stack([f.ravel()[self.rows] for f in frames])
        if not np.isfinite(blocks).all():
            raise ValueError("有效像素必须有限；坏点请用固定 mask 排除")
        self.blocks = blocks
        self.observed = blocks.ravel()
        self.normalizer = np.sqrt(self.observed.size)

    def evaluate(self, q, *, jacobian=True):
        """返回按帧堆叠的归一化残差及可选雅可比。"""
        c = self.to_coefficients(q)
        residuals, jacobians = [], []
        for i, mod in enumerate(self.mods):
            intensity, jac = _frame_and_jacobian(
                self.cache, c, mod, compute_jacobian=jacobian,
            )
            residuals.append(intensity - self.blocks[i])
            if jacobian:
                jacobians.append(jac)
        residual = np.concatenate(residuals) / self.normalizer
        jac = (np.vstack(jacobians) / (self.normalizer * self.scale[None, :])
               if jacobian else None)
        return residual, jac

    def value_gradient(self, q):
        """返回 (loss, g_q)；逐帧累加 raw jtr 后按总像素数 M 归一化。"""
        c = self.to_coefficients(q)
        m = float(self.observed.size)
        total = 0.0
        jtr = np.zeros_like(c)
        for i, mod in enumerate(self.mods):
            intensity, block_jtr = self._frame_jtr(c, self.blocks[i], mod)
            total += np.sum((intensity - self.blocks[i]) ** 2)
            jtr += block_jtr
        return float(total / (2.0 * m)), jtr / (m * self.scale)
