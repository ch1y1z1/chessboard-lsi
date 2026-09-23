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

为什么 5 光束叠加就是模型本身
--------------------------
每个级次自带一个平移后的光瞳，图 2-8/2-9 的 2/3/4/5 光束重叠区自动出现，
无需手工区域簿记。``paper_region_intensity`` 逐字转写了论文的显式
4/5 光束公式 (2-12)...(2-16)，用于对照验证（scripts/01）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .config import Grid, SystemConfig
from .zernike import zernike, zernike_matrix

__all__ = [
    "ZernikeWavefront",
    "ForwardModel",
    "paper_region_intensity",
    "add_noise",
    "DEFAULT_ORDERS",
    "DEFAULT_ORDERS_9",
]

#: 理想 50% 占空比棋盘光栅的 5 光束组（0 级 + 四个一级），振幅取自表 2-3。
_A1 = 2.0 / np.pi**2
DEFAULT_ORDERS: dict[tuple[float, float], complex] = {
    (0.0, 0.0): 0.5,
    (1.0, 0.0): -_A1,
    (-1.0, 0.0): -_A1,
    (0.0, 1.0): +_A1,
    (0.0, -1.0): +_A1,
}
#: 论文 3.1.2 节保留的四个物理三级衍射（探测器坐标下在坐标轴上）。
DEFAULT_ORDERS_9: dict[tuple[float, float], complex] = {
    **DEFAULT_ORDERS,
    (3.0, 0.0): -2.0 / (9.0 * np.pi**2),
    (-3.0, 0.0): -2.0 / (9.0 * np.pi**2),
    (0.0, 3.0): +2.0 / (9.0 * np.pi**2),
    (0.0, -3.0): +2.0 / (9.0 * np.pi**2),
}


@dataclass
class ZernikeWavefront:
    """``W = sum_j c_j Z_j``，论文 Zernike 约定（系数单位：波长）。

    ``coeffs[k]`` 是 1 起 Fringe 序号 ``indices[k]`` 的系数；Z1（平移）
    通常保持为 0。
    """

    coeffs: np.ndarray
    indices: np.ndarray

    def __post_init__(self) -> None:
        self.coeffs = np.atleast_1d(np.asarray(self.coeffs, dtype=float))
        indices = np.atleast_1d(np.asarray(self.indices))
        if indices.dtype == bool:
            raise ValueError("indices 必须是正整数，不接受布尔值")
        indices = np.asarray(indices, dtype=float)
        if self.coeffs.shape != indices.shape:
            raise ValueError("coeffs 与 indices 长度必须一致")
        if not np.all(indices == np.round(indices)) or (indices < 1).any():
            raise ValueError("indices 必须是正整数")
        self.indices = indices.astype(int)

    def w(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.zeros_like(np.asarray(x, dtype=float))
        for c, j in zip(self.coeffs, self.indices):
            if c:
                out = out + c * zernike(int(j), x, y)
        return out

    @property
    def n_terms(self) -> int:
        return len(self.coeffs)


def _eval_w(wf, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """接受带 ``.w(x, y)`` 的对象或普通可调用波前。"""
    return wf.w(x, y) if hasattr(wf, "w") else wf(x, y)


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

    def aperture(self) -> np.ndarray:
        """零级光瞳（单位圆盘）。"""
        return self.grid.pupil()

    def order_geometry(self):
        """逐级的 (a, b)、移位坐标 xs=x+a*s / ys=y+b*s 及光瞳掩膜。"""
        out = []
        for (a, b) in self.order_list:
            xs, ys = self._x + a * self.s, self._y + b * self.s
            out.append((a, b, xs, ys, xs * xs + ys * ys <= 1.0))
        return out

    def order_support(self, a: float, b: float) -> np.ndarray:
        """级 (a, b) 的光瞳掩膜；该级不存在时返回全 False。"""
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
        """各级载频相位 2 pi f0 (a x + b y)，形状 (n_orders, n, n)。"""
        max_ab = max(max(abs(a), abs(b)) for a, b in self.order_list)
        if max_ab * abs(f0) >= self.grid.nyquist:
            raise ValueError(
                f"载频 |f0| = {abs(f0):.3f} x 最高级次 {max_ab} 超过网格奈奎斯特 "
                f"{self.grid.nyquist:.3f} cyc/unit，会混叠"
            )
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
        """探测器复振幅：I = |E|^2 中的 E。"""
        E = np.zeros(self.shape, dtype=complex)
        for k, (a, b, xs, ys, inside) in enumerate(self.order_geometry()):
            amp = self.amplitudes[k]
            if amp == 0.0:
                continue
            phase = 2.0 * np.pi * _eval_w(wf, xs, ys)
            if deltas is not None:
                phase = phase + deltas[k]
            if carriers is not None:
                phase = phase + carriers[k]
            E = E + np.where(inside, amp * np.exp(1j * phase), 0.0)
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
        if direction not in ("x", "y"):
            raise ValueError("direction 必须是 'x' 或 'y'")
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
    def demodulation_offset(self, direction: str = "x") -> float:
        """解调出的频率-1 干涉相位携带的常数项（弧度）。

        两个对称 beat 系数 C_+ = A_+ conj(A_0)、C_- = A_0 conj(A_-)；
        模相等时（``_require_symmetric_pair`` 的前提）合成

            D = 2 |C| exp(i pi [W(x+s) - W(x-s)] + i beta) cos(Gamma)

        常数项 beta 是两个 beat 辐角的均值 (arg C_+ + arg C_-) / 2。对理想棋盘
        (A_00, A_10, A_01) = (+1/2, -2/pi^2, +2/pi^2)，x 对为 pi、y 对为 0：
        两个剪切对正好相差半条纹。占空比误差使该常数随光栅相位漂移
        （arg A_10 = pi - 2 pi (d - 1/2)）—— 论文 4.1.1 节。
        """
        amps = dict(zip(self.order_list, self.amplitudes))
        a0 = amps.get((0.0, 0.0), 0.0)
        ap, am = (
            (amps.get((1.0, 0.0), 0.0), amps.get((-1.0, 0.0), 0.0))
            if direction == "x"
            else (amps.get((0.0, 1.0), 0.0), amps.get((0.0, -1.0), 0.0))
        )
        if a0 == 0 or ap == 0 or am == 0:
            return 0.0
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
            amp = self.amplitudes[k]
            if amp == 0.0:
                continue
            xs_f, ys_f = xs.ravel()[rows], ys.ravel()[rows]
            Z = zernike_matrix(indices, xs_f, ys_f).T  # (n_terms, n_rows)
            out.append((k, amp, inside.ravel()[rows], Z))
        return out


# --------------------------------------------------------------------------- #
# 论文显式区域公式 (2-12) ... (2-16)，逐字转写，用于对照验证
# --------------------------------------------------------------------------- #
def paper_region_intensity(
    region: str,
    W: Callable,
    x: np.ndarray,
    y: np.ndarray,
    shear: float,
    delta: float,
    A0: float = 0.5,
    A1: float | None = None,
) -> np.ndarray:
    """论文式 (2-12)...(2-16) 的逐字转写。

    ``region`` 取值：

    ``"x1"``  (2-12): 0 + (+1,+1) + (-1,+1) + (-1,-1)
    ``"x2"``  (2-13): 0 + (+1,+1) + (-1,-1) + (+1,-1)
    ``"x5"``  (2-14): 0 + 全部四个一级（五光束）
    ``"y1"``  (2-15): 0 + (+1,+1) + (-1,+1) + (+1,-1)
    ``"y2"``  (2-16): 0 + (+1,+1) + (-1,-1) + (-1,+1)
    ``"y5"``  ``"x5"`` 的 y 方向对偶（五光束区，论文未单列公式）
    """
    s = shear
    A1 = 2.0 / np.pi**2 if A1 is None else A1
    tp = 2.0 * np.pi
    w0 = W(x, y)
    wxp, wxm = W(x + s, y), W(x - s, y)
    wyp, wym = W(x, y + s), W(x, y - s)
    hxs, hxd = (wxp + wxm) / 2.0, (wxp - wxm) / 2.0   # x 方向半和/半差
    hys, hyd = (wyp + wym) / 2.0, (wyp - wym) / 2.0

    if region in ("x1", "x2"):
        other = wyp if region == "x1" else wym
        I = A0**2 + 3.0 * A1**2
        I += 2.0 * A0 * A1 * np.cos(tp * (other - w0))
        I += 4.0 * A0 * A1 * np.cos(tp * (hxs - w0)) * np.cos(tp * hxd + delta)
        I += 4.0 * A1**2 * np.cos(tp * (hxs - other)) * np.cos(tp * hxd + delta)
        I += 2.0 * A1**2 * np.cos(tp * 2.0 * hxd + 2.0 * delta)
        return I

    if region in ("y1", "y2"):
        other = wxp if region == "y1" else wxm
        I = A0**2 + 3.0 * A1**2
        I += 2.0 * A0 * A1 * np.cos(tp * (other - w0))
        I += 4.0 * A0 * A1 * np.cos(tp * (hys - w0)) * np.cos(tp * hyd + delta)
        I += 4.0 * A1**2 * np.cos(tp * (hys - other)) * np.cos(tp * hyd + delta)
        I += 2.0 * A1**2 * np.cos(tp * 2.0 * hyd + 2.0 * delta)
        return I

    if region in ("x5", "y5"):
        hs, hd, wp, wm = (hxs, hxd, wyp, wym) if region == "x5" else (hys, hyd, wxp, wxm)
        I = A0**2 + 4.0 * A1**2
        I += 2.0 * A0 * A1 * (np.cos(tp * (wp - w0)) + np.cos(tp * (wm - w0)))
        I += 2.0 * A1**2 * np.cos(tp * (wp - wm))
        for coeff, shift in ((4.0 * A0 * A1, w0), (4.0 * A1**2, wp), (4.0 * A1**2, wm)):
            I += coeff * np.cos(tp * (hs - shift)) * np.cos(tp * hd + delta)
        I += 2.0 * A1**2 * np.cos(tp * 2.0 * hd + 2.0 * delta)
        return I

    raise ValueError(f"未知区域 {region!r}")


# --------------------------------------------------------------------------- #
def add_noise(frames: np.ndarray, snr_db: float, seed: int | None = 0) -> np.ndarray:
    """加高斯噪声；``snr_db`` 为峰值光强对噪声标准差之比 20 log10(max I / sigma)。"""
    frames = np.asarray(frames, dtype=float)
    rng = np.random.default_rng(seed)
    sigma = frames.max() / (10.0 ** (snr_db / 20.0))
    return frames + rng.normal(0.0, sigma, size=frames.shape)
