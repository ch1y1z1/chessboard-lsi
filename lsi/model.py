"""前向模型：45° 旋转棋盘光栅剪切干涉仪（论文 2.3 / 2.4）。

按计算链条自上而下：

    1. Grid          归一化光瞳坐标下的采样网格
    2. SystemConfig  lambda / NA / 光栅周期 -> 剪切量 s、载频 f0
    3. 棋盘光栅级次   衍射级振幅 A_ab（表 2-3）
    4. Zernike 基     Fringe/Wyant 序与双边差分基（式 2-25~2-27）
    5. ForwardModel  级次叠加 E = sum A_ab exp(i[...])，I = |E|^2

前向模型
--------
光栅把入射场分成探测器坐标下的衍射级 (a, b)。设级次振幅为 A_ab，出瞳 P
为单位圆盘，则探测器上的复振幅为

    E(x, y) = sum_{a,b} A_ab * P(x + a s, y + b s)
                       * exp{ i [ 2 pi W(x + a s, y + b s) + delta_ab
                                  + 2 pi f0 (a x + b y) ] }
    I(x, y) = |E(x, y)|^2

其中 W 为被测波前（单位：波长），s 为剪切量，delta_ab 为光栅相移项，
f0 为傅里叶模式的空间载频。

相移律（论文式 2-10）：光栅沿探测器 x 方向平移 dx 时，级 (a, b) 的相位移动

    delta_ab = 2 pi (a t_x + b t_y),   t_x = sqrt(2) dx / p

即相移量与衍射级次成正比：光栅移动 p/sqrt(2)（t_x = 1）使 (+1,+1) 级
恰好走 2 pi，而 (+1,-1) 级不动 —— 正是论文 2.3.1 节的结论。

每个级次自带一个平移后的光瞳，图 2-8/2-9 的 2/3/4/5 光束重叠区自动出现，
无需手工区域簿记；与论文显式五光束公式 (2-14) 的逐点对照见
``test_lsi.py``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# 1. 采样网格与系统参数（论文第三章仿真参数）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Grid:
    """归一化光瞳坐标下的方形采样网格（覆盖 [-extent, extent]^2）。"""

    n: int = 256
    extent: float = 1.10

    @property
    def dx(self) -> float:
        return 2.0 * self.extent / self.n

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        v = (np.arange(self.n) - (self.n - 1) / 2.0) * self.dx
        return np.meshgrid(v, v, indexing="xy")

    def pupil(self) -> np.ndarray:
        """单位圆（出瞳）掩膜。"""
        x, y = self.coords()
        return x * x + y * y <= 1.0


@dataclass(frozen=True)
class SystemConfig:
    """物理参数：波长 / NA / 光栅周期 -> 剪切量 s 与载频 f0。

        相移模式   : NA = 0.34, p = 18 um, lambda = 632.8 nm  ->  s = 0.0731
        傅里叶模式 : NA = 0.34, p = 30 um                    ->  s = 0.0439
    """

    wavelength_nm: float = 632.8
    na: float = 0.34
    period_um: float = 18.0
    grid: Grid = field(default_factory=Grid)

    @property
    def s(self) -> float:
        """归一化剪切量（衍射级位移占光瞳半径的比例）。

        s = sqrt(2) * lambda / (2 NA p) —— sqrt(2) 来自棋盘光栅的 45° 旋转；
        论文表 2-5 直接以此作为单位圆坐标下的位移量。
        """
        lam_um = self.wavelength_nm * 1e-3
        return float(np.sqrt(2.0) * lam_um / (2.0 * self.period_um * self.na))

    @property
    def carrier_f0(self) -> float:
        """空间载频 f0 = m / (2s)（m = 1），单位：周期/归一化坐标。

        论文式 (2-48) 为 f0 = 2m/s；完整载频超出默认网格的奈奎斯特频率，
        默认值是采样驱动的折中，解调数学不变。
        """
        return 1.0 / (2.0 * self.s)

    def describe(self) -> str:
        return (
            f"lambda={self.wavelength_nm:g} nm  NA={self.na:g}  "
            f"p={self.period_um:g} um  s={self.s:.5f}  "
            f"f0={self.carrier_f0:.3f} cyc/unit  "
            f"grid={self.grid.n}x{self.grid.n}"
        )


# --------------------------------------------------------------------------- #
# 2. 棋盘光栅衍射级次（论文 2.1，表 2-3）
# --------------------------------------------------------------------------- #
# 探测器坐标 (a, b) 与光栅坐标 (m, n) 的关系（45° 旋转）：
#
#     a = (m + n) / 2,   b = (n - m) / 2       <=>   m = a - b, n = a + b
#
# 理想 50% 占空比：只有 m、n 同为奇数的级次非零，
#
#     A_00 = 1/2,   A_mn = -2 / (pi^2 m n)   (m, n 均为奇数)
#
# 即 |A_±1| = 2/pi^2 = 0.2026（效率 4.11%），Parseval: sum |A|^2 = 1/2。
def chessboard_orders(max_index: int = 3) -> dict[tuple[float, float], complex]:
    """探测器坐标下的级次振幅表 ``{(a, b): A_ab}``（含 (0, 0) 级）。

    保留 max(|a|, |b|) <= max_index 的级次。
    """
    orders: dict[tuple[float, float], complex] = {}
    for m in range(-2 * max_index, 2 * max_index + 1):
        for n in range(-2 * max_index, 2 * max_index + 1):
            if m % 2 == 0 or n % 2 == 0:
                continue
            a, b = (m + n) / 2.0, (n - m) / 2.0
            if max(abs(a), abs(b)) <= max_index:
                orders[(a, b)] = -2.0 / (np.pi**2 * m * n)
    orders[(0.0, 0.0)] = 0.5
    return dict(sorted(orders.items()))


def diffraction_efficiency(orders: dict) -> dict[str, float | dict]:
    """各衍射级的效率 |A|^2 及常用汇总。"""
    eff = {k: float(abs(v) ** 2) for k, v in orders.items()}
    first = sum(
        eff.get(k, 0.0) for k in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    )
    return {"dc": eff[(0.0, 0.0)], "first_order_total": first, "all": eff}


#: 理想 50% 占空比棋盘光栅的 5 光束组（0 级 + 四个一级），振幅取自表 2-3。
_A1 = 2.0 / np.pi**2
DEFAULT_ORDERS: dict[tuple[float, float], complex] = {
    (0.0, 0.0): 0.5,
    (1.0, 0.0): -_A1,
    (-1.0, 0.0): -_A1,
    (0.0, 1.0): +_A1,
    (0.0, -1.0): +_A1,
}


# --------------------------------------------------------------------------- #
# 3. Zernike 多项式（论文采用的 Fringe/Wyant 排序）
# --------------------------------------------------------------------------- #
# 论文表 2-5 列出前 16 项差分 Zernike 多项式 dZ(x,y) = Z(x+s,y) - Z(x-s,y)，
# 反推可知其基底是经典（未归一化）实 Zernike 集：
#
#     Z_j(rho, theta) = R_n^|m|(rho) * { cos(m theta)  (偶 j)
#                                       sin(m theta)  (奇 j) }
#
# 径向多项式（论文式 2-25）：
#
#     R_n^m(rho) = sum_s (-1)^s (n-s)! / [s! ((n+m)/2-s)! ((n-m)/2-s)!] rho^(n-2s)
#
# 数值实现用分解 Z = Q(rho^2) * Re/Im[(x + i y)^m]，在原点稳定且不需要 atan2。

#: (n, |m|, kind, 像差名称)，下标从 1 开始（Fringe/Wyant 序，到 Z36）。
FRINGE_MODES: tuple[tuple[int, int, str, str], ...] = (
    (0, 0, "radial", "piston"),
    (1, 1, "cos", "x tilt"), (1, 1, "sin", "y tilt"),
    (2, 0, "radial", "defocus"),
    (2, 2, "cos", "primary astigmatism"), (2, 2, "sin", "primary astigmatism"),
    (3, 1, "cos", "primary coma"), (3, 1, "sin", "primary coma"),
    (4, 0, "radial", "primary spherical"),
    (3, 3, "cos", "primary trefoil"), (3, 3, "sin", "primary trefoil"),
    (4, 2, "cos", "secondary astigmatism"), (4, 2, "sin", "secondary astigmatism"),
    (5, 1, "cos", "secondary coma"), (5, 1, "sin", "secondary coma"),
    (6, 0, "radial", "secondary spherical"),
    (4, 4, "cos", "primary quadrafoil"), (4, 4, "sin", "primary quadrafoil"),
    (5, 3, "cos", "secondary trefoil"), (5, 3, "sin", "secondary trefoil"),
    (6, 2, "cos", "tertiary astigmatism"), (6, 2, "sin", "tertiary astigmatism"),
    (7, 1, "cos", "tertiary coma"), (7, 1, "sin", "tertiary coma"),
    (8, 0, "radial", "tertiary spherical"),
    (5, 5, "cos", "primary pentafoil"), (5, 5, "sin", "primary pentafoil"),
    (6, 4, "cos", "secondary quadrafoil"), (6, 4, "sin", "secondary quadrafoil"),
    (7, 3, "cos", "tertiary trefoil"), (7, 3, "sin", "tertiary trefoil"),
    (8, 2, "cos", "quaternary astigmatism"), (8, 2, "sin", "quaternary astigmatism"),
    (9, 1, "cos", "quaternary coma"), (9, 1, "sin", "quaternary coma"),
    (10, 0, "radial", "quaternary spherical"),
)


def zernike(j: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """单位圆上的实 Zernike 多项式 Z_j(x, y)（Fringe/Wyant 序，1 起）。"""
    n, m, kind, _ = FRINGE_MODES[j - 1]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho2 = x * x + y * y

    # R_n^m = Q(rho^2) * rho^m；先累加 Q
    kmax = (n - m) // 2
    q = np.zeros_like(rho2)
    for s in range(kmax + 1):
        coeff = (
            (-1.0) ** s * factorial(n - s)
            / (factorial(s) * factorial((n + m) // 2 - s) * factorial((n - m) // 2 - s))
        )
        q = q + coeff * rho2 ** (kmax - s)
    if m == 0:
        return q
    # (x + i y)^m = rho^m (cos mθ + i sin mθ)
    trig = np.real((x + 1j * y) ** m) if kind == "cos" else np.imag((x + 1j * y) ** m)
    return q * trig


def zernike_matrix(
    indices: Sequence[int], x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """(n_pixels, n_terms) 的 Zernike 基矩阵。"""
    return np.stack([zernike(j, x, y) for j in indices], axis=-1)


def differential_zernike_matrix(
    indices: Sequence[int], x: np.ndarray, y: np.ndarray, s: float, direction: str
) -> np.ndarray:
    """双边差分 Zernike 基矩阵（论文式 2-26/2-27）。

        ΔZx_j = Z_j(x+s, y) - Z_j(x-s, y)
        ΔZy_j = Z_j(x, y+s) - Z_j(x, y-s)
    """
    if direction == "x":
        rows = [zernike(j, x + s, y) - zernike(j, x - s, y) for j in indices]
    else:
        rows = [zernike(j, x, y + s) - zernike(j, x, y - s) for j in indices]
    return np.stack(rows, axis=-1)


def wavefront(
    coeffs: Sequence[float], indices: Sequence[int], x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """W(x, y) = sum_j c_j Z_j（系数单位：波长）。"""
    W = np.zeros_like(np.asarray(x, dtype=float))
    for c, j in zip(coeffs, indices):
        if c != 0.0:
            W = W + c * zernike(j, x, y)
    return W


def zernike_wavefront(coeffs: Sequence[float], indices: Sequence[int]):
    """``W(x, y) = sum_j c_j Z_j`` 的可调用波前（系数单位：波长，Fringe 序）。"""
    return lambda x, y: wavefront(coeffs, indices, x, y)


# --------------------------------------------------------------------------- #
# 4. 级次叠加前向模型
# --------------------------------------------------------------------------- #
class ForwardModel:
    """棋盘光栅剪切干涉的级次叠加前向模型（见模块 docstring 的公式）。"""

    def __init__(
        self,
        config: SystemConfig | None = None,
        orders: dict[tuple[float, float], complex] | None = None,
    ) -> None:
        self.config = config or SystemConfig()
        self.grid: Grid = self.config.grid
        orders = DEFAULT_ORDERS if orders is None else orders
        self.order_list: list[tuple[float, float]] = list(orders)
        self.amplitudes = np.asarray([orders[o] for o in self.order_list], dtype=complex)
        self._x, self._y = self.grid.coords()

    # ------------------------------------------------------------- 几何
    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    @property
    def s(self) -> float:
        return self.config.s

    def order_geometry(self):
        """逐级的 (a, b)、移位坐标 xs=x+a*s / ys=y+b*s 及光瞳掩膜。"""
        out = []
        for (a, b) in self.order_list:
            xs, ys = self._x + a * self.s, self._y + b * self.s
            out.append((a, b, xs, ys, xs * xs + ys * ys <= 1.0))
        return out

    def order_support(self, a: float, b: float) -> np.ndarray:
        """级 (a, b) 的光瞳掩膜。"""
        for aa, bb, _, _, inside in self.order_geometry():
            if (aa, bb) == (a, b):
                return inside
        return np.zeros(self.shape, dtype=bool)

    # ------------------------------------------------------------- 相位律
    def phase_shift_deltas(self, t_x: float = 0.0, t_y: float = 0.0) -> np.ndarray:
        """光栅平移 (t_x, t_y) 时各级相移（弧度），与 order_list 对齐。

        delta_ab = 2 pi (a t_x + b t_y)   —— 论文式 (2-10)。
        """
        ab = np.asarray(self.order_list)
        return 2.0 * np.pi * (ab[:, 0] * t_x + ab[:, 1] * t_y)

    def carrier_phases(self, f0: float) -> np.ndarray:
        """各级载频相位 2 pi f0 (a x + b y)，形状 (n_orders, n, n)。

        采样约束按光强而非场强给：I = |E|^2 的拍频是级次之差，
        (a_i - a_j) f0，故要求级次坐标 span 乘 f0 低于奈奎斯特
        （默认 ±1 级即 2 f0）。
        """
        ab = np.asarray(self.order_list)
        return (
            2.0 * np.pi * f0
            * (ab[:, 0, None, None] * self._x + ab[:, 1, None, None] * self._y)
        )

    # ------------------------------------------------------------- 光强
    def intensity(
        self,
        wf,
        deltas: np.ndarray | None = None,
        carriers: np.ndarray | None = None,
    ) -> np.ndarray:
        """探测器光强 I = |E|^2。``wf`` 为可调用波前 W(x, y)。"""
        E = np.zeros(self.shape, dtype=complex)
        for k, (a, b, xs, ys, inside) in enumerate(self.order_geometry()):
            phase = 2.0 * np.pi * wf(xs, ys)
            if deltas is not None:
                phase = phase + deltas[k]
            if carriers is not None:
                phase = phase + carriers[k]
            E = E + np.where(inside, self.amplitudes[k] * np.exp(1j * phase), 0.0)
        return np.abs(E) ** 2

    # ------------------------------------------------------------- 采集序列
    def phase_shift_frames(self, wf, direction: str, n_steps: int) -> np.ndarray:
        """(n_steps, n, n) 相移干涉图序列。

        第 i 步 t = i / n_steps，即一级相移步进 2 pi / n_steps —— 论文
        式 (2-20) 的 N 步最小二乘相移所要求的均匀步进。
        """
        frames = []
        for i in range(n_steps):
            t = i / float(n_steps)
            deltas = (
                self.phase_shift_deltas(t, 0.0)
                if direction == "x"
                else self.phase_shift_deltas(0.0, t)
            )
            frames.append(self.intensity(wf, deltas=deltas))
        return np.stack(frames, axis=0)

    def carrier_frame(self, wf, f0: float | None = None) -> np.ndarray:
        """单帧载频（傅里叶变换模式）干涉图，论文 2.4.1。"""
        f0 = self.config.carrier_f0 if f0 is None else float(f0)
        return self.intensity(wf, carriers=self.carrier_phases(f0))

    # ------------------------------------------------------------- 解调先验
    def demodulation_offset(self, direction: str) -> float:
        """解调出的频率-1 干涉相位携带的常数项（弧度）。

        两个对称 beat 系数 C_+ = A_+ conj(A_0)、C_- = A_0 conj(A_-)；
        模相等时合成

            D = 2 |C| exp(i pi [W(x+s) - W(x-s)] + i beta) cos(Gamma)

        常数项 beta 是两个 beat 辐角的均值 (arg C_+ + arg C_-) / 2。对理想棋盘
        (A_00, A_10, A_01) = (+1/2, -2/pi^2, +2/pi^2)，x 对为 pi、y 对为 0：
        两个剪切对正好相差半条纹。
        """
        amps = dict(zip(self.order_list, self.amplitudes))
        a0 = amps.get((0.0, 0.0), 0.0)
        ap, am = (
            (amps.get((1.0, 0.0), 0.0), amps.get((-1.0, 0.0), 0.0))
            if direction == "x"
            else (amps.get((0.0, 1.0), 0.0), amps.get((0.0, -1.0), 0.0))
        )
        c_p, c_m = ap * np.conj(a0), a0 * np.conj(am)
        return float(np.angle(c_p) + 0.5 * np.angle(c_m / c_p))

    # ------------------------------------------------------------- LM 用
    def zernike_samples(
        self, indices: Sequence[int], rows: np.ndarray
    ) -> list[tuple[int, complex, np.ndarray, np.ndarray]]:
        """每个衍射级在采样像素 rows 上的 (级位置 k, 振幅, 掩膜, Z 基)。

        Z 形状为 (n_terms, n_rows)，供 ``lm.py`` 的解析雅可比使用。
        """
        out = []
        for k, (a, b, xs, ys, inside) in enumerate(self.order_geometry()):
            xs_f, ys_f = xs.ravel()[rows], ys.ravel()[rows]
            Z = zernike_matrix(indices, xs_f, ys_f).T  # (n_terms, n_rows)
            out.append((k, self.amplitudes[k], inside.ravel()[rows], Z))
        return out
