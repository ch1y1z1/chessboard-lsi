"""前向模型：45° 旋转棋盘光栅剪切干涉仪（论文 2.3 / 2.4）。

模型
----
光栅把入射场分成探测器坐标下的衍射级 (a, b)。设级次振幅为 A_ab（见
``grating.py``），出瞳 P 为单位圆盘，则探测器上的复振幅为

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
无需手工区域簿记；与论文显式区域公式 (2-12)...(2-16) 的逐点对照见
``test_lsi.py``。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import Grid, SystemConfig
from .zernike import wavefront, zernike_matrix

#: 理想 50% 占空比棋盘光栅的 5 光束组（0 级 + 四个一级），振幅取自表 2-3。
_A1 = 2.0 / np.pi**2
DEFAULT_ORDERS: dict[tuple[float, float], complex] = {
    (0.0, 0.0): 0.5,
    (1.0, 0.0): -_A1,
    (-1.0, 0.0): -_A1,
    (0.0, 1.0): +_A1,
    (0.0, -1.0): +_A1,
}


def zernike_wavefront(coeffs: Sequence[float], indices: Sequence[int]):
    """``W(x, y) = sum_j c_j Z_j`` 的可调用波前（系数单位：波长，Fringe 序）。"""
    return lambda x, y: wavefront(coeffs, indices, x, y)


class ForwardModel:
    """棋盘光栅剪切干涉的级次叠加前向模型。"""

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

    # ------------------------------------------------------------- 场与光强
    def field(
        self,
        wf,
        deltas: np.ndarray | None = None,
        carriers: np.ndarray | None = None,
    ) -> np.ndarray:
        """探测器复振幅：I = |E|^2 中的 E。``wf`` 为可调用波前 W(x, y)。"""
        E = np.zeros(self.shape, dtype=complex)
        for k, (a, b, xs, ys, inside) in enumerate(self.order_geometry()):
            phase = 2.0 * np.pi * wf(xs, ys)
            if deltas is not None:
                phase = phase + deltas[k]
            if carriers is not None:
                phase = phase + carriers[k]
            E = E + np.where(inside, self.amplitudes[k] * np.exp(1j * phase), 0.0)
        return E

    def intensity(self, wf, deltas=None, carriers=None) -> np.ndarray:
        return np.abs(self.field(wf, deltas=deltas, carriers=carriers)) ** 2

    # ------------------------------------------------------------- 采集序列
    def phase_shift_frames(
        self, wf, direction: str = "x", n_steps: int | None = None
    ) -> np.ndarray:
        """(n_steps, n, n) 相移干涉图序列。

        第 i 步 t = i / n_steps，即一级相移步进 2 pi / n_steps —— 论文
        式 (2-20) 的 N 步最小二乘相移所要求的均匀步进。
        """
        n_steps = self.config.phase_steps if n_steps is None else int(n_steps)
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


def add_noise(frames: np.ndarray, snr_db: float, seed: int | None = 0) -> np.ndarray:
    """加高斯噪声；``snr_db`` 为峰值光强对噪声标准差之比 20 log10(max I / sigma)。"""
    frames = np.asarray(frames, dtype=float)
    rng = np.random.default_rng(seed)
    sigma = frames.max() / (10.0 ** (snr_db / 20.0))
    return frames + rng.normal(0.0, sigma, size=frames.shape)
