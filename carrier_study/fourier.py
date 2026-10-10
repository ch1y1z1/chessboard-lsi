"""傅里叶解调链（论文路线 B）及其精确一阶线性化。

``FourierChain.estimate`` 与 ``lsi.invert.fourier_to_wavefront`` 同算法同参数
（默认 ``FourierParams()`` 即 lsi 的模块常数），只做三处工程改动：
预计算差分 Zernike 基、向量化构造解包裹的掩膜拉普拉斯、可调参数。
等价性由测试锁定。

线性化（掩膜固定、绕开 ±pi 分支）：

    dc    = 低通( dI * exp(-2 pi i f0 u) ) * exp(-i offset)
    dphi  = Im(dc / c0)
    dpsi  : A dpsi = -div(grad dphi)（掩膜内），钉点 dpsi_k = dphi_k
    dW    = dpsi / pi
    damp  = Re(conj(c0) dc) / |c0|                       （权重 = 幅值）
    dq    = P [dW_x; dW_y] + G^-1 A^T diag(res) [damp_x; damp_y]
            P = (A^T W A)^-1 A^T W，G = A^T W A，res = b - A q0（权重项随残差变化）

``L`` 把整幅 n x n 光强扰动映到 dq；预测协方差 Cov = L diag(var) L^T。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import splu

from lsi.invert import _masked_divergence, _shear_region
from lsi.model import ForwardModel, SystemConfig, differential_zernike_matrix

from .zern import rms_factors


@dataclass(frozen=True)
class FourierParams:
    radius_factor: float = 4.0   # 低通半径 = factor * L（频谱像素）
    threshold: float = 0.5       # 幅值掩膜阈值（相对剪切区内最大幅值）
    erode: int = 4               # 掩膜边缘腐蚀像素数

    def label(self) -> str:
        return f"r{self.radius_factor:g}_t{self.threshold:g}_e{self.erode}"


DEFAULT_PARAMS = FourierParams()


def masked_laplacian(mask: np.ndarray):
    """掩膜内 4 邻域拉普拉斯（与 lsi.invert.unwrap_poisson 同构），并钉住最靠中心的像素。

    返回 (A 的 LU 分解, ys, xs, 钉点下标 k)。
    """
    ys, xs = np.nonzero(mask)
    N = ys.size
    index = -np.ones(mask.shape, dtype=np.int64)
    index[ys, xs] = np.arange(N)
    k = int(np.argmin((ys - mask.shape[0] / 2.0) ** 2 + (xs - mask.shape[1] / 2.0) ** 2))
    rows, cols, vals = [], [], []
    deg = np.zeros(N)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ii, jj = ys + di, xs + dj
        ok = (ii >= 0) & (ii < mask.shape[0]) & (jj >= 0) & (jj < mask.shape[1])
        nb = np.full(N, -1, dtype=np.int64)
        nb[ok] = index[ii[ok], jj[ok]]
        ok &= nb >= 0
        deg += ok
        r = np.flatnonzero(ok)
        keep = r != k
        rows.append(r[keep])
        cols.append(nb[ok][keep])
        vals.append(np.full(int(keep.sum()), -1.0))
    diag = deg.copy()
    diag[k] = 1.0
    rows.append(np.arange(N))
    cols.append(np.arange(N))
    vals.append(diag)
    A = sparse.csc_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)
    )
    return splu(A), ys, xs, k


def unwrap_fast(phi: np.ndarray, mask: np.ndarray, lap=None) -> np.ndarray:
    """掩膜内最小二乘解包裹，结果与 lsi.invert.unwrap_poisson 一致。"""
    lu, ys, xs, k = masked_laplacian(mask) if lap is None else lap
    rho = _masked_divergence(phi, mask)
    rhs = -rho[ys, xs]
    rhs[k] = phi[ys[k], xs[k]]
    out = np.full(phi.shape, np.nan)
    out[ys, xs] = lu.solve(rhs)
    return out


@lru_cache(maxsize=4)
def _diff_basis_q(cfg: SystemConfig, indices: tuple[int, ...]) -> dict[str, np.ndarray]:
    """差分基直接放到 q 坐标：dW = dZ_c c = (dZ_c / rms) q。同一系统的各参数组共享。"""
    x, y = cfg.grid.coords()
    rms = rms_factors(indices)
    return {d: differential_zernike_matrix(indices, x, y, cfg.s, d) / rms for d in ("x", "y")}


class FourierChain:
    def __init__(
        self,
        cfg: SystemConfig,
        indices: Sequence[int],
        params: FourierParams = DEFAULT_PARAMS,
        orders: dict | None = None,
    ) -> None:
        self.cfg = cfg
        self.indices = tuple(indices)
        self.params = params
        self.fm = ForwardModel(cfg, orders)
        g = cfg.grid
        n, L = g.n, 2.0 * g.extent
        x, y = g.coords()
        f0 = cfg.carrier_f0
        self.demod = {"x": np.exp(-2j * np.pi * f0 * x), "y": np.exp(-2j * np.pi * f0 * y)}
        yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        rr = np.hypot(xx - n // 2, yy - n // 2)
        win = 1.0 / (1.0 + (rr / (params.radius_factor * L)) ** 4)
        self.win_u = np.fft.ifftshift(win)
        self.support = {d: _shear_region(self.fm, d) for d in ("x", "y")}
        self.rot = {d: np.exp(-1j * self.fm.demodulation_offset(d)) for d in ("x", "y")}
        self.dZ = _diff_basis_q(cfg, self.indices)

    # ---------------------------------------------------------------- 解调
    def lobe(self, image: np.ndarray, d: str) -> np.ndarray:
        c = np.fft.ifft2(np.fft.fft2(image * self.demod[d]) * self.win_u)
        return c * self.rot[d]

    def _mask(self, amp: np.ndarray, d: str) -> np.ndarray:
        sup = self.support[d]
        mask = (amp > self.params.threshold * amp[sup].max()) & sup
        if self.params.erode > 0:
            mask = ndimage.binary_erosion(mask, iterations=self.params.erode)
        # 只保留最大的 4 连通块：多个连通块会让解包裹的拉普拉斯奇异
        # （lsi 原实现假定掩膜单连通，遇到断开的掩膜只给出警告和错误结果）
        lab, n = ndimage.label(mask)
        if n > 1:
            sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
            mask = lab == (1 + int(np.argmax(sizes)))
        return mask

    def estimate(self, image: np.ndarray, *, keep: bool = False):
        """返回 (q 估计, info)。info 含掩膜、解调复振幅与（keep 时）解包裹分解。"""
        image = np.asarray(image, dtype=float)
        rows, rhs, wts = [], [], []
        info = {}
        for d in ("x", "y"):
            c = self.lobe(image, d)
            amp = np.abs(c)
            mask = self._mask(amp, d)
            lap = masked_laplacian(mask)
            psi = unwrap_fast(np.angle(c), mask, lap)
            dW = psi / np.pi
            rows.append(self.dZ[d][mask])
            rhs.append(dW[mask])
            wts.append(np.sqrt(np.clip(amp[mask], 1e-6, None)))
            info[d] = {"c": c, "mask": mask, "lap": lap if keep else None, "b": dW[mask]}
        A = np.vstack(rows)
        b = np.concatenate(rhs)
        w = np.concatenate(wts)
        q, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
        info["A"], info["w"] = A, w
        return q, info

    # ---------------------------------------------------------------- 线性化
    def linearize(self, image0: np.ndarray) -> "FourierLinearization":
        return FourierLinearization(self, image0)


class FourierLinearization:
    def __init__(self, chain: FourierChain, image0: np.ndarray) -> None:
        self.chain = chain
        self.q0, info = chain.estimate(image0, keep=True)
        self.parts = {}
        for d in ("x", "y"):
            self.parts[d] = info[d]
        A, w = info["A"], info["w"]
        Aw = A * w[:, None]
        self.P = np.linalg.pinv(Aw) * w[None, :]          # (p, N_rows)
        b = np.concatenate([info[d]["b"] for d in ("x", "y")])
        res = b - A @ self.q0
        G = Aw.T @ Aw
        self.Q = np.linalg.solve(G, A.T * res[None, :])    # (p, N_rows)
        self.n_rows = {d: int(info[d]["mask"].sum()) for d in ("x", "y")}

    def apply(self, dI: np.ndarray) -> np.ndarray:
        """dI 形状 (B, n, n) -> dq 形状 (B, p)。"""
        ch = self.chain
        dI = np.asarray(dI, dtype=float)
        if dI.ndim == 2:
            dI = dI[None]
        dWs, damps = [], []
        for d in ("x", "y"):
            part = self.parts[d]
            c0, mask = part["c"], part["mask"]
            lu, ys, xs, k = part["lap"]
            dc = np.fft.ifft2(np.fft.fft2(dI * ch.demod[d], axes=(1, 2)) * ch.win_u, axes=(1, 2))
            dc = dc * ch.rot[d]
            with np.errstate(divide="ignore", invalid="ignore"):
                dphi = np.imag(dc / c0)
            dphi = np.where(mask[None], dphi, 0.0)
            # 线性化的掩膜散度（_masked_divergence 去掉 wrap）
            rho = np.zeros_like(dphi)
            for di, dj in ((0, 1), (1, 0)):
                valid = mask & np.roll(mask, (-di, -dj), axis=(0, 1))
                if di:
                    valid[-1, :] = False
                if dj:
                    valid[:, -1] = False
                g = np.where(valid[None], np.roll(dphi, (-di, -dj), axis=(1, 2)) - dphi, 0.0)
                rho += g
                rho -= np.roll(g, (di, dj), axis=(1, 2))
            rhs = -rho[:, ys, xs]
            rhs[:, k] = dphi[:, ys[k], xs[k]]
            dpsi = lu.solve(np.ascontiguousarray(rhs.T)).T
            dWs.append(dpsi / np.pi)
            damps.append((np.real(np.conj(c0) * dc) / np.abs(c0))[:, ys, xs])
        return np.concatenate(dWs, axis=1) @ self.P.T + np.concatenate(damps, axis=1) @ self.Q.T

    def matrix(self, batch: int = 64) -> np.ndarray:
        """完整线性映射 L，形状 (p, n*n)。"""
        n = self.chain.cfg.grid.n
        N = n * n
        L = np.zeros((self.P.shape[0], N))
        for start in range(0, N, batch):
            stop = min(N, start + batch)
            basis = np.zeros((stop - start, N))
            basis[np.arange(stop - start), np.arange(start, stop)] = 1.0
            L[:, start:stop] = self.apply(basis.reshape(-1, n, n)).T
        return L
