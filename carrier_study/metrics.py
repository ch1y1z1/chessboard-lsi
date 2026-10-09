"""Fisher 信息、CRB 与误差度量（全部在 q 坐标）。

白化相对误差 d_rel(dq) = sqrt(dq^T FIM dq / p)：统计误差的期望 d_rel^2 = 1
（有效无偏估计）。优化误差 d_rel <= tau 意味着它给 MSE 带来的增量不超过 tau^2。
"""

from __future__ import annotations

import numpy as np


def fim(model, q: np.ndarray, var_pix) -> np.ndarray:
    """FIM = J^T diag(1/var) J；var_pix 为标量或像素集上的数组。"""
    _, state = model.forward(q)
    J = model.jacobian(state)
    var = np.broadcast_to(np.asarray(var_pix, dtype=float), (model.M,))
    return (J / var[:, None]).T @ J


class Whitener:
    def __init__(self, F: np.ndarray) -> None:
        self.F = F
        self.p = F.shape[0]
        self.L = np.linalg.cholesky(F)          # F = L L^T
        self.crb = np.linalg.inv(F)

    def d_rel(self, dq: np.ndarray) -> np.ndarray:
        """白化相对误差；dq 形状 (p,) 或 (B, p)。"""
        z = np.asarray(dq) @ self.L
        return np.sqrt(np.sum(z * z, axis=-1) / self.p)

    @property
    def crb_wavefront_rms(self) -> float:
        """CRB 对应的期望波前 RMS 误差 sqrt(tr CRB)。"""
        return float(np.sqrt(np.trace(self.crb)))


def wf_err(dq: np.ndarray) -> np.ndarray:
    """拟合模式部分的波前 RMS 误差 ||dq||。"""
    return np.linalg.norm(np.asarray(dq), axis=-1)


def wf_err_notilt(dq: np.ndarray, indices) -> np.ndarray:
    """去掉 Z2/Z3 倾斜后的波前 RMS 误差。"""
    keep = np.array([j not in (2, 3) for j in indices])
    return np.linalg.norm(np.asarray(dq)[..., keep], axis=-1)
