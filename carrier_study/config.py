"""研究常量（见 docs/carrier-research-plan.md 第 2 节）。"""

from __future__ import annotations

import numpy as np

from lsi.model import Grid, SystemConfig

WAVELENGTH_NM = 632.8
NA = 0.34
PERIOD_UM = 30.0            # 主配置：s = 0.0439，f0 = 11.40 cyc/unit
N_GRID = 128
EXTENT = 1.10

INDICES_MAIN = tuple(range(2, 37))     # Z2..Z36，p = 35
INDICES_CTRL = tuple(range(2, 14))     # Z2..Z13，p = 12（对照，衔接旧结果）

#: E2 剪切量扫描用的三档光栅周期（均满足 2 f0 < 奈奎斯特）
PERIODS_SHEAR_SCAN = (20.0, 25.0, 30.0)

#: 种子流：dev / test 真值、噪声、初值、方向互不重叠
SEED_DEV = 101_000
SEED_TEST = 202_000
SEED_NOISE = 303_000
SEED_INIT = 404_000
SEED_MISMATCH = 505_000


def system(period_um: float = PERIOD_UM, n: int = N_GRID) -> SystemConfig:
    return SystemConfig(
        wavelength_nm=WAVELENGTH_NM, na=NA, period_um=period_um,
        grid=Grid(n=n, extent=EXTENT),
    )


def nyquist(n: int = N_GRID) -> float:
    """网格奈奎斯特频率（cyc/unit）。"""
    return 1.0 / (2.0 * Grid(n=n, extent=EXTENT).dx)


def indices_for(p: int) -> tuple[int, ...]:
    if p == len(INDICES_MAIN):
        return INDICES_MAIN
    if p == len(INDICES_CTRL):
        return INDICES_CTRL
    raise ValueError(f"未定义 p={p} 的拟合集")


def rng(*key: int) -> np.random.Generator:
    """由整数键派生独立随机流。"""
    return np.random.default_rng(np.random.SeedSequence(list(key)))
