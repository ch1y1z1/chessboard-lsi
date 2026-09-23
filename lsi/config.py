"""系统配置：45° 旋转棋盘光栅横向剪切干涉仪（论文第三章仿真参数）。

坐标约定：出瞳归一化为单位圆盘，采样网格覆盖 [-extent, extent]^2；
波前以波长为单位（waves）。

    相移模式   : NA = 0.34, p = 18 um, lambda = 632.8 nm  ->  s = 0.0731
    傅里叶模式 : NA = 0.34, p = 30 um                    ->  s = 0.0439
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Grid", "SystemConfig", "PRESET_PHASE_SHIFT", "PRESET_FOURIER"]


@dataclass(frozen=True)
class Grid:
    """归一化光瞳坐标下的方形采样网格。"""

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
    """物理参数。``shear_ratio`` 为 None 时由波长/NA/光栅周期推导。"""

    wavelength_nm: float = 632.8
    na: float = 0.34
    period_um: float = 18.0
    shear_ratio: float | None = None
    phase_steps: int = 8        # 每个方向的相移步数（论文用 8）
    talbot_number: int = 1      # 傅里叶模式的泰伯级次 m
    grid: Grid = field(default_factory=Grid)

    @property
    def s(self) -> float:
        """归一化剪切量（衍射级位移占光瞳半径的比例）。

        s = sqrt(2) * lambda / (2 NA p) —— sqrt(2) 来自棋盘光栅的 45° 旋转；
        论文表 2-5 直接以此作为单位圆坐标下的位移量。
        """
        if self.shear_ratio is not None:
            return float(self.shear_ratio)
        lam_um = self.wavelength_nm * 1e-3
        return float(np.sqrt(2.0) * lam_um / (2.0 * self.period_um * self.na))

    @property
    def carrier_f0(self) -> float:
        """空间载频 f0 = m / (2s)，单位：周期/归一化坐标。

        这是论文式 (2-48) f0 = 2m/s 的四分之一：完整载频超出默认网格的
        奈奎斯特频率，默认值是采样驱动的折中，解调数学不变。
        """
        return self.talbot_number / (2.0 * self.s)

    @property
    def carrier_f0_paper(self) -> float:
        """论文式 (2-48) 的载频 f0 = 2m/s（需要更细的网格）。"""
        return 2.0 * self.talbot_number / self.s

    def describe(self) -> str:
        return (
            f"lambda={self.wavelength_nm:g} nm  NA={self.na:g}  "
            f"p={self.period_um:g} um  s={self.s:.5f}  "
            f"f0={self.carrier_f0:.3f} cyc/unit  "
            f"grid={self.grid.n}x{self.grid.n}"
        )


#: 论文第三章模拟的两套配置。
PRESET_PHASE_SHIFT = SystemConfig()
PRESET_FOURIER = SystemConfig(period_um=30.0)
