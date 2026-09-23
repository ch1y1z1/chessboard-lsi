"""掩膜内的最小二乘（Poisson）相位解包裹。

Ghiglia–Romero 方法：在掩膜内求 psi 使其梯度在最小二乘意义下等于缠绕
相位的缠绕梯度，即解离散 Poisson 方程

    sum_{掩膜内4邻域 j} (psi_i - psi_j) = -rho_i

rho 为缠绕梯度场的散度。每个 4-连通分量有一个自由常数，把每个分量
最靠近中心的像素钉到其缠绕值上（与 pipeline 的锚定约定一致）。
掩膜外为 NaN。
"""

from __future__ import annotations

import numpy as np

__all__ = ["wrap", "unwrap_poisson"]


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
    """掩膜内最小二乘解包裹；掩膜外为 NaN。"""
    from scipy import ndimage, sparse
    from scipy.sparse.linalg import spsolve

    phi = np.asarray(phi, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    out = np.full(phi.shape, np.nan)
    if not np.any(mask):
        return out

    rho = _masked_divergence(phi, mask)
    labels, n_comp = ndimage.label(mask)
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

    # 每个连通分量钉住其最靠中心的像素到缠绕值
    cy, cx = phi.shape[0] / 2.0, phi.shape[1] / 2.0
    A = A.tolil()
    for lab in range(1, n_comp + 1):
        members = np.nonzero(labels[ys, xs] == lab)[0]
        k = members[np.argmin((ys[members] - cy) ** 2 + (xs[members] - cx) ** 2)]
        A[k, :] = 0.0
        A[k, k] = 1.0
        rhs[k] = phi[ys[k], xs[k]]
    sol = spsolve(A.tocsr(), rhs)
    out[ys, xs] = sol
    return out
