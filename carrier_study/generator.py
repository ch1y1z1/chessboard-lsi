"""数据生成端：与反演端分离，用来打破同模型自证。

``Scene`` 描述"真实世界"与名义模型的差别；``Scene()`` 即 matched 模式，
逐像素等于 ``NominalModel``。``REALISTIC`` 只加入物理上不可避免的像素积分
（4x4 超采样后盒式平均）。其余失配由 E4 单因素扫描。

棋盘光栅（占空比 d）取 t = (1 + s_x s_y) / 2，s 为 ±1 方波：

    S(0) = 2d - 1,   S(m) = (1 - exp(-2 pi i m d)) / (i pi m)
    T(m, n) = (delta_m0 delta_n0 + S(m) S(n)) / 2

d = 0.5 时退化为 lsi.model.chessboard_orders 的 -2 / (pi^2 m n)。探测器坐标
(a, b) = ((m + n) / 2, (n - m) / 2)；m 或 n 为偶数的级次只在 d != 0.5 时出现，
落在半整数 (a, b) 上。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from lsi.model import SystemConfig

from .zern import basis_q, tail_basis_q

TWO_PI = 2.0 * np.pi
MAIN_ORDERS = ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))


@dataclass(frozen=True)
class Scene:
    supersample: int = 1        # 每像素 S x S 子采样后平均（像素积分）
    duty: float = 0.5           # 棋盘占空比
    higher_orders: float = 0.0  # 理想高阶奇-奇级次的振幅缩放（0 = 已滤除）
    amp_err: float = 0.0        # 各级次振幅相对误差（均匀分布半宽）
    phase_err: float = 0.0      # 各级次常数相位误差（rad，均匀分布半宽）
    shear_err: float = 0.0      # 剪切量相对误差
    carrier_err: float = 0.0    # 载频相对误差
    illum: float = 0.0          # 光瞳照明振幅 1 - illum * rho^2
    tail_frac: float = 0.0      # 截断尾项 RMS / 拟合部分 RMS（由调用方提供尾项）
    seed: int = 0               # 振幅/相位误差的随机流

    def label(self) -> str:
        base = Scene()
        diffs = [f"{k}={getattr(self, k)}" for k in self.__dataclass_fields__
                 if k != "seed" and getattr(self, k) != getattr(base, k)]
        return ",".join(diffs) or "matched"


MATCHED = Scene()
REALISTIC = Scene(supersample=4)


def grating_orders(duty: float = 0.5, max_index: int = 2) -> dict[tuple[float, float], tuple[complex, str]]:
    """{(a, b): (振幅, 类别)}；类别 main / higher / duty。"""

    def S(m: int) -> complex:
        if m == 0:
            return 2.0 * duty - 1.0
        return (1.0 - np.exp(-2j * np.pi * m * duty)) / (1j * np.pi * m)

    out = {}
    for m in range(-2 * max_index, 2 * max_index + 1):
        for n in range(-2 * max_index, 2 * max_index + 1):
            a, b = (m + n) / 2.0, (n - m) / 2.0
            if max(abs(a), abs(b)) > max_index:
                continue
            T = 0.5 * ((1.0 if (m == 0 and n == 0) else 0.0) + S(m) * S(n))
            if abs(T) < 1e-14:
                continue
            if (a, b) in MAIN_ORDERS:
                kind = "main"
            elif m % 2 and n % 2:
                kind = "higher"
            else:
                kind = "duty"
            out[(a, b)] = (complex(T), kind)
    return out


def scene_orders(scene: Scene) -> dict[tuple[float, float], complex]:
    """按场景挑选级次并施加振幅/相位误差。"""
    rng = np.random.default_rng(np.random.SeedSequence([7, scene.seed]))
    orders = {}
    for key, (amp, kind) in sorted(grating_orders(scene.duty).items()):
        if kind == "higher":
            if scene.higher_orders == 0.0:
                continue
            amp = amp * scene.higher_orders
        orders[key] = amp
    for key in orders:
        u_a, u_p = rng.uniform(-1.0, 1.0, size=2)
        orders[key] = orders[key] * (1.0 + scene.amp_err * u_a) * np.exp(1j * scene.phase_err * u_p)
    return orders


class Generator:
    """给定真值 q（与可选尾项）生成 n x n 干净光强。"""

    def __init__(self, cfg: SystemConfig, indices: Sequence[int], scene: Scene = MATCHED) -> None:
        self.cfg = cfg
        self.indices = tuple(indices)
        self.scene = scene
        g = cfg.grid
        S = int(scene.supersample)
        x, y = g.coords()
        off = ((np.arange(S) + 0.5) / S - 0.5) * g.dx
        # (n, n, S, S) 子采样坐标；S = 1 时恰为像素中心
        shape4 = (g.n, g.n, S, S)
        self.X = np.broadcast_to(x[:, :, None, None] + off[None, None, None, :], shape4).reshape(-1)
        self.Y = np.broadcast_to(y[:, :, None, None] + off[None, None, :, None], shape4).reshape(-1)
        self.S = S
        self.s = cfg.s * (1.0 + scene.shear_err)
        self.f0 = cfg.carrier_f0 * (1.0 + scene.carrier_err)
        self.orders = scene_orders(scene)

    def clean(self, q: np.ndarray, tail_q: np.ndarray | None = None) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        E = np.zeros(self.X.size, dtype=complex)
        for (a, b), amp in self.orders.items():
            xs, ys = self.X + a * self.s, self.Y + b * self.s
            r2 = xs * xs + ys * ys
            inside = r2 <= 1.0
            xi, yi = xs[inside], ys[inside]
            W = basis_q(self.indices, xi, yi) @ q
            if tail_q is not None and np.any(tail_q):
                W = W + tail_basis_q(xi, yi) @ tail_q
            env = 1.0 - self.scene.illum * r2[inside]
            phase = TWO_PI * W + TWO_PI * self.f0 * (a * self.X[inside] + b * self.Y[inside])
            E[inside] += amp * env * np.exp(1j * phase)
        I = (E.real**2 + E.imag**2)
        n = self.cfg.grid.n
        return I.reshape(n, n, self.S, self.S).mean(axis=(2, 3))


# --------------------------------------------------------------------------- #
# 噪声
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Noise:
    kind: str          # "none" | "gauss" | "poisson"
    level: float = 0.0  # gauss: SNR (dB, 峰值/σ)；poisson: 峰值光电子数
    read_e: float = 5.0  # poisson：读出噪声（e-）

    @property
    def label(self) -> str:
        if self.kind == "gauss":
            return f"G{self.level:g}"
        if self.kind == "poisson":
            return f"P{self.level:.0e}".replace("+0", "").replace("+", "")
        return "clean"


def add_noise(I: np.ndarray, noise: Noise, rng: np.random.Generator):
    """返回 (观测, 真实逐像素方差, 由观测估计的逐像素方差)。"""
    I = np.asarray(I, dtype=float)
    peak = float(I.max())
    if noise.kind == "none":
        z = np.zeros_like(I)
        return I.copy(), z, z
    if noise.kind == "gauss":
        sigma = peak / 10.0 ** (noise.level / 20.0)
        var = np.full_like(I, sigma**2)
        return I + rng.normal(0.0, sigma, size=I.shape), var, var
    if noise.kind == "poisson":
        scale = noise.level / peak                     # 光电子 / 光强单位
        counts = rng.poisson(I * scale) + rng.normal(0.0, noise.read_e, size=I.shape)
        y = counts / scale
        var_true = (I * scale + noise.read_e**2) / scale**2
        var_est = (np.clip(counts, 0.0, None) + noise.read_e**2) / scale**2
        return y, var_true, var_est
    raise ValueError(noise.kind)


def with_seed(scene: Scene, seed: int) -> Scene:
    return replace(scene, seed=seed)
