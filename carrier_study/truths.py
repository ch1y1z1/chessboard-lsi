"""真值池：四种谱形 x RMS 档位 x 重复，dev 与 test 用不同种子流。

    pow1 / pow2   系数标准差 ∝ (径向阶 + 1)^(-alpha)，alpha = 1 / 2
    flat          全部拟合项同标准差
    sparse        在 Z4..Z13 中随机取 3 项
全部按 ||q|| 缩放到目标波前 RMS（waves）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import config
from .zern import TAIL_MODES, radial_orders

SHAPES = ("pow1", "pow2", "flat", "sparse")
RMS_TEST = (0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 3.0)
RMS_DEV = (0.03, 0.1, 0.3, 1.0, 2.0)
REPS = {"dev": 2, "test": 4}


@dataclass(frozen=True)
class Truth:
    tid: str
    split: str
    shape: str
    rms: float
    rep: int
    q: np.ndarray = field(repr=False, compare=False)


def sample_q(shape: str, rms: float, indices: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    indices = tuple(indices)
    n = radial_orders(indices).astype(float)
    if shape in ("pow1", "pow2"):
        alpha = 1.0 if shape == "pow1" else 2.0
        z = rng.normal(size=len(indices)) * (n + 1.0) ** (-alpha)
    elif shape == "flat":
        z = rng.normal(size=len(indices))
    elif shape == "sparse":
        cand = [k for k, j in enumerate(indices) if 4 <= j <= 13]
        pick = rng.choice(cand, size=min(3, len(cand)), replace=False)
        z = np.zeros(len(indices))
        z[pick] = rng.normal(size=pick.size)
    else:
        raise ValueError(shape)
    return z / np.linalg.norm(z) * rms


def truth_set(split: str, indices: Sequence[int], *, rms_levels=None, shapes=SHAPES, reps=None) -> list[Truth]:
    base = {"dev": config.SEED_DEV, "test": config.SEED_TEST}[split]
    rms_levels = (RMS_DEV if split == "dev" else RMS_TEST) if rms_levels is None else rms_levels
    reps = REPS[split] if reps is None else reps
    p = len(indices)
    out = []
    for si, shape in enumerate(shapes):
        for ri, rms in enumerate(rms_levels):
            for rep in range(reps):
                rng = config.rng(base, p, si, int(round(rms * 1000)), rep)
                q = sample_q(shape, rms, indices, rng)
                tid = f"{split}-p{p}-{shape}-{rms:g}-{rep}"
                out.append(Truth(tid, split, shape, float(rms), rep, q))
    return out


def tail_q(rms_tail: float, rng: np.random.Generator) -> np.ndarray:
    """截断尾项（Fringe 第 6、7 组），平坦谱，RMS = rms_tail。"""
    z = rng.normal(size=len(TAIL_MODES))
    return z / np.linalg.norm(z) * rms_tail
