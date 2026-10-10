"""阶段 1 的理论量：定义域统计、曲率、条件数、梯度尺度、收敛与成本模型。

记号：ΔW_x = W(x+s, y) - W(x-s, y)（waves）。+f0 瓣解调后 c ∝ exp(i pi ΔW_x)，
其局部频率为 ½|∇ΔW_x|（cyc/unit）；Butterworth 低通半径 = radius_factor（cyc/unit）。
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
from scipy import ndimage

from lsi.invert import _shear_region
from lsi.model import DEFAULT_ORDERS, ForwardModel, SystemConfig

from .nominal import NominalModel
from .zern import basis_q

TWO_PI = 2.0 * np.pi


def _W(indices, q, x, y):
    return basis_q(indices, x, y) @ q


def domain_stats(cfg: SystemConfig, indices: Sequence[int], q: np.ndarray) -> dict:
    """真值在本系统下的局部频率统计（决定傅里叶链与采样是否失效）。"""
    g = cfg.grid
    x, y = g.coords()
    s, dx, f0 = cfg.s, g.dx, cfg.carrier_f0
    nyq = 1.0 / (2.0 * dx)
    fm = ForwardModel(cfg)
    out = {}
    lobe_all = []
    for d, (ax, ay) in (("x", (1, 0)), ("y", (0, 1))):
        dW = _W(indices, q, x + ax * s, y + ay * s) - _W(indices, q, x - ax * s, y - ay * s)
        sup = ndimage.binary_erosion(_shear_region(fm, d), iterations=1)
        gy, gx = np.gradient(dW, dx)
        f = 0.5 * np.hypot(gx, gy)[sup]
        lobe_all.append(f)
        out[f"dW_{d}_max"] = float(np.abs(dW[sup]).max())
    lobe = np.concatenate(lobe_all)
    out["lobe_freq_max"] = float(lobe.max())
    out["lobe_freq_p95"] = float(np.percentile(lobe, 95))
    # 5 光束两两拍频的局部频率（含载频）对奈奎斯特
    worst = 0.0
    orders = list(DEFAULT_ORDERS)
    for (a1, b1), (a2, b2) in itertools.combinations(orders, 2):
        inside = ((x + a1 * s) ** 2 + (y + b1 * s) ** 2 <= 1) & ((x + a2 * s) ** 2 + (y + b2 * s) ** 2 <= 1)
        inside = ndimage.binary_erosion(inside, iterations=1)
        ph = _W(indices, q, x + a1 * s, y + b1 * s) - _W(indices, q, x + a2 * s, y + b2 * s)
        gy, gx = np.gradient(ph, dx)
        fx = np.abs(gx + f0 * (a1 - a2))[inside]
        fy = np.abs(gy + f0 * (b1 - b2))[inside]
        worst = max(worst, float(max(fx.max(), fy.max())))
    out["beat_freq_max"] = worst
    out["beat_over_nyquist"] = worst / nyq
    return out


def _accel(model: NominalModel, state, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """沿方向 v 的一阶 Jv 与二阶 d2I/dt2。"""
    E, es = state
    dE = np.zeros(model.M, dtype=complex)
    d2E = np.zeros(model.M, dtype=complex)
    for blk, e in zip(model.blocks, es):
        zv = blk.Zt @ v
        dE[blk.idx] += 1j * TWO_PI * e * zv
        d2E[blk.idx] += -(TWO_PI**2) * e * zv * zv
    Jv = 2.0 * np.real(np.conj(E) * dE)
    a = 2.0 * np.abs(dE) ** 2 + 2.0 * np.real(np.conj(E) * d2E)
    return Jv, a


def curvature(model: NominalModel, q: np.ndarray, n_dirs: int = 200, seed: int = 0) -> dict:
    """Bates–Watts 内在/参数效应曲率（单位 1/光强）与 GN 二次收敛常数 C（单位 1/wave）。

    gamma_N(v) = ||P_perp a(v)|| / ||J v||^2,  gamma_T(v) = ||P a(v)|| / ||J v||^2,
    C(v) = ||(J^T J)^-1 J^T a(v)|| / 2  （|v| = 1：一步 GN 后误差 ≈ C e^2）
    """
    _, st = model.forward(q)
    J = model.jacobian(st)
    G = J.T @ J
    w, V = np.linalg.eigh(G)
    rng = np.random.default_rng(seed)
    dirs = list(V.T) + [u / np.linalg.norm(u) for u in rng.normal(size=(n_dirs, model.p))]
    gN, gT, C = [], [], []
    for v in dirs:
        Jv, a = _accel(model, st, v)
        coef = np.linalg.solve(G, J.T @ a)
        proj = J @ coef
        nrm = float(Jv @ Jv)
        gN.append(np.linalg.norm(a - proj) / nrm)
        gT.append(np.linalg.norm(proj) / nrm)
        C.append(0.5 * np.linalg.norm(coef))
    return {
        "gamma_N_max": float(np.max(gN)), "gamma_T_max": float(np.max(gT)),
        "C_gn_max": float(np.max(C)), "C_gn_median": float(np.median(C)),
        "kappa_JtJ": float(w[-1] / w[0]),
        "sigma_min_J": float(np.sqrt(w[0])), "sigma_max_J": float(np.sqrt(w[-1])),
    }


def gradient_scale(cfg: SystemConfig, indices: Sequence[int]) -> dict:
    """单位 RMS 随机方向 u 的波前梯度尺度（光瞳上面积平均）。

    E_u <(∂x W_u)^2> = (1/p) sum_j <(∂x Z~_j)^2>；返回其平方根 g_x（waves/unit）。
    """
    n = 400
    v = (np.arange(n) - (n - 1) / 2.0) * (2.0 / n)
    x, y = np.meshgrid(v, v)
    pupil = x * x + y * y <= 1.0
    h = 1e-5
    Bxp = basis_q(indices, x[pupil] + h, y[pupil])
    Bxm = basis_q(indices, x[pupil] - h, y[pupil])
    dZx = (Bxp - Bxm) / (2 * h)
    gx2 = np.mean(dZx**2, axis=0)            # 每项 <(∂x Z~_j)^2>
    return {"g_x": float(np.sqrt(gx2.mean())), "g_x_per_mode": gx2.tolist()}


# --------------------------------------------------------------------------- #
# 收敛与成本模型（阶段 1.2）
# --------------------------------------------------------------------------- #
def iters_gd(kappa: float, ratio: float) -> float:
    """最速下降（近最优步长）把误差缩小 ratio 倍所需迭代数。"""
    rate = (kappa - 1.0) / (kappa + 1.0)
    return float(np.log(ratio) / -np.log(rate)) if ratio > 1 else 0.0


def iters_accel(kappa: float, ratio: float) -> float:
    """加速梯度 / CG 型：约 sqrt(kappa) ln(ratio)。"""
    return float(np.sqrt(kappa) * np.log(ratio)) if ratio > 1 else 0.0


def iters_gn(e0: float, target: float, C: float, max_iter: int = 50) -> int:
    """GN 二次收敛 e_{k+1} = C e_k^2 的步数（噪声下的线性项 ρ_GN 远小于 1，忽略）。"""
    e, k = e0, 0
    while e > target and k < max_iter:
        e = C * e * e
        k += 1
    return k
