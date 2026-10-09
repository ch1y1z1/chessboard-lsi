"""反演端名义模型：理想 5 光束、点采样、已知载频，参数为 q 坐标。

    E = sum_k A_k exp(i [2 pi Zq_k q + mod_k])     （只在级次 k 的光瞳内）
    I = |E|^2

导数（全部解析）：

    dI/dq        = -4 pi sum_k Zq_k^T Im(conj(E) e_k)
    sum_i u_i d2I_i/dq2
                 = 2 Re(D^H diag(u) D) - 8 pi^2 sum_k Zq_k^T diag(u Re(conj(E) e_k)) Zq_k
    其中 D = dE/dq = sum_k 2 pi i e_k Zq_k。

像素集默认取各级次光瞳的并集；并集之外没有光，不含信息。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from lsi.model import DEFAULT_ORDERS, SystemConfig

from .zern import basis_q

TWO_PI = 2.0 * np.pi


@dataclass
class _Block:
    a: float
    b: float
    amp: complex
    idx: np.ndarray        # 该级次光瞳内的像素（像素集内的下标）
    Zt: np.ndarray         # (M_k, p) q 坐标基
    mod: np.ndarray        # (M_k,) 载频相位


class NominalModel:
    def __init__(
        self,
        cfg: SystemConfig,
        indices: Sequence[int],
        *,
        orders: dict | None = None,
        f0: float | None = None,
        pixels: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg
        self.indices = tuple(int(j) for j in indices)
        self.p = len(self.indices)
        orders = DEFAULT_ORDERS if orders is None else orders
        self.f0 = cfg.carrier_f0 if f0 is None else float(f0)
        s = cfg.s
        x, y = cfg.grid.coords()
        xf, yf = x.ravel(), y.ravel()
        if pixels is None:
            union = np.zeros(xf.size, dtype=bool)
            for a, b in orders:
                union |= (xf + a * s) ** 2 + (yf + b * s) ** 2 <= 1.0
            pixels = np.flatnonzero(union)
        self.pix = np.asarray(pixels)
        self.M = self.pix.size
        xp, yp = xf[self.pix], yf[self.pix]
        self.blocks: list[_Block] = []
        for (a, b), amp in orders.items():
            xs, ys = xp + a * s, yp + b * s
            idx = np.flatnonzero(xs * xs + ys * ys <= 1.0)
            Zt = np.ascontiguousarray(basis_q(self.indices, xs[idx], ys[idx]))
            mod = TWO_PI * self.f0 * (a * xp[idx] + b * yp[idx])
            self.blocks.append(_Block(a, b, complex(amp), idx, Zt, mod))

    # ------------------------------------------------------------------ 前向
    def forward(self, q: np.ndarray):
        q = np.asarray(q, dtype=float)
        E = np.zeros(self.M, dtype=complex)
        es = []
        for blk in self.blocks:
            e = blk.amp * np.exp(1j * (TWO_PI * (blk.Zt @ q) + blk.mod))
            E[blk.idx] += e
            es.append(e)
        I = E.real**2 + E.imag**2
        return I, (E, es)

    def intensity(self, q: np.ndarray) -> np.ndarray:
        return self.forward(q)[0]

    # ------------------------------------------------------------------ 导数
    def _im_terms(self, state):
        E, es = state
        return [np.imag(np.conj(E[blk.idx]) * e) for blk, e in zip(self.blocks, es)]

    def jacobian(self, state) -> np.ndarray:
        """dI/dq，形状 (M, p)。"""
        J = np.zeros((self.M, self.p))
        for blk, v in zip(self.blocks, self._im_terms(state)):
            J[blk.idx] += (-2.0 * TWO_PI * v)[:, None] * blk.Zt
        return J

    def jtv(self, state, u: np.ndarray) -> np.ndarray:
        """sum_i u_i dI_i/dq，不构造雅可比。"""
        g = np.zeros(self.p)
        for blk, v in zip(self.blocks, self._im_terms(state)):
            g += blk.Zt.T @ (-2.0 * TWO_PI * v * u[blk.idx])
        return g

    def hessian_terms(self, state, u: np.ndarray) -> np.ndarray:
        """sum_i u_i * d2I_i/dq2，形状 (p, p)。"""
        E, es = state
        D = np.zeros((self.M, self.p), dtype=complex)
        H2 = np.zeros((self.p, self.p))
        for blk, e in zip(self.blocks, es):
            D[blk.idx] += (1j * TWO_PI * e)[:, None] * blk.Zt
            w = u[blk.idx] * np.real(np.conj(E[blk.idx]) * e)
            H2 -= 2.0 * TWO_PI**2 * (blk.Zt.T * w) @ blk.Zt
        H1 = 2.0 * np.real((np.conj(D) * u[:, None]).T @ D)
        return H1 + H2

    # ------------------------------------------------------------------ 工具
    def embed(self, values: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """像素集上的值放回 n x n 图像。"""
        n = self.cfg.grid.n
        img = np.full(n * n, fill, dtype=np.result_type(values, float))
        img[self.pix] = values
        return img.reshape(n, n)

    def take(self, image: np.ndarray) -> np.ndarray:
        return np.asarray(image).ravel()[self.pix]
