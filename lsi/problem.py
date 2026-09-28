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
        started = perf_counter()
        self.forward = forward
        raw_indices = np.asarray(indices)
        if (raw_indices.ndim != 1 or not raw_indices.size
                or not np.issubdtype(raw_indices.dtype, np.integer)):
            raise ValueError("indices 必须是非空整数序列")
        self.indices = raw_indices.astype(int, copy=True)
        if 1 in self.indices or len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("排除不可观测的 Z1，且 indices 不得重复")
        image = np.asarray(image, dtype=float)
        if image.shape != forward.shape:
            raise ValueError("image 必须是一帧，形状与探测器一致")
        if mask is None:
            mask = np.ones(forward.shape, dtype=bool)
        mask = np.asarray(mask)
        if mask.shape != forward.shape or mask.dtype != bool or not mask.any():
            raise ValueError("mask 必须是非空且形状匹配的布尔掩膜")
        self.rows = np.flatnonzero(mask)
        self.observed = image.ravel()[self.rows].copy()
        if not np.isfinite(self.observed).all():
            raise ValueError("有效像素必须有限；坏点请用固定 mask 排除")
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
        self.normalizer = np.sqrt(self.rows.size)
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
