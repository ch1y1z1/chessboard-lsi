"""单帧无调制反问题。观测与前向模型独立于求解器，不接收真值。"""
from __future__ import annotations

from time import perf_counter
from typing import Sequence

import numpy as np

from .lm import _frame_and_jacobian
from .model import ForwardModel, zernike_matrix


class IntensityProblem:
    """F(q)=1/2 mean((I(q)-image)^2), q_j=std_pupil(Z_j)*c_j。

    返回已除以 sqrt(M) 的残差和雅可比。mask 只允许固定的几何/仪器掩膜。
    scaling='none' 可用于原始系数坐标消融。全部计算采用 float64。
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
        self.precompute_seconds = perf_counter() - started

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
