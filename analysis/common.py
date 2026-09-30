"""analysis 共享工具：帧设计构造、采样、初值生成、收敛点分类。

与 experiment.py 第 5 节同一套约定：

* 真值 TRUTH_IDX/TRUTH_C（waves），拟合 INDICES = Z2..Z13（Z1 不可观）
* 网格 n=128，extent=1.10；相移设计用 p=18 um 配置，载频设计用 p=30 um
* 采样像素沿用 ``fit_wavefront_from_frames`` 的黄金比例低差异序列，
  保证对称性度量、Hessian 与 LM 看到的是同一像素支撑

帧设计表 ``build_designs`` 返回 (label, forward, frames, modulations,
use_carrier_fit)，其中 ``modulations`` 与 ``fit_wavefront_from_frames``
的约定一致：每帧一项，(n_orders,) 为相移、(n_orders,n,n) 为载频。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np

from lsi.lm import _frame_and_jacobian
from lsi.model import ForwardModel, Grid, SystemConfig, zernike_wavefront

# --------------------------------------------------------------------------- #
# 实验常量（与 experiment.py 第 5 节一致）
# --------------------------------------------------------------------------- #
INDICES = np.arange(2, 14)                     # Z2..Z13
TRUTH_IDX = np.array([4, 5, 6, 7, 8])
TRUTH_C = np.array([0.31, -0.12, 0.07, 0.42, 0.05])
GRID = Grid(n=128, extent=1.10)
CFG_SHIFT = SystemConfig(grid=GRID)                    # p=18 um, s=0.0731
CFG_CARRIER = SystemConfig(grid=GRID, period_um=30.0)  # p=30 um, s=0.0439
SAMPLES = 4096
SEED = 42
N_DIRECTIONS = 10
INITIAL_RMS = (0.03, 0.1, 0.3)


def truth_vector(indices: Sequence[int] = INDICES) -> np.ndarray:
    """INDICES 顺序下的真值系数向量（缺省模式为 0）。"""
    m = dict(zip(TRUTH_IDX.tolist(), TRUTH_C.tolist()))
    return np.array([m.get(int(j), 0.0) for j in indices])


def sample_rows(forward: ForwardModel, samples: int | None) -> np.ndarray:
    """与 ``fit_wavefront_from_frames`` 相同的确定性像素采样。"""
    n_pix = forward.shape[0] * forward.shape[1]
    if samples is None or samples >= n_pix:
        return np.arange(n_pix)
    return np.sort(
        (np.arange(samples) * 0.6180339887498949 % 1.0 * n_pix).astype(int)
    )


def pupil_rms(coeffs: np.ndarray, indices: Sequence[int], grid: Grid) -> float:
    """系数向量对应波前在光瞳内的去平移 RMS（waves）。"""
    x, y = grid.coords()
    pupil = grid.pupil()
    W = zernike_wavefront(coeffs, indices)(x, y)[pupil]
    return float(np.sqrt(np.mean((W - W.mean()) ** 2)))


def initial_guesses(
    grid: Grid = GRID,
    indices: Sequence[int] = INDICES,
    n_directions: int = N_DIRECTIONS,
    rms_levels: Sequence[float] = INITIAL_RMS,
    seed: int = SEED,
    include_zero: bool = True,
) -> list[tuple[float, int, np.ndarray]]:
    """固定随机方向 × 波前 RMS 的初值表（experiment.py 第 5 节协议）。

    返回 ``(initial_rms, direction_id, c0)``；``direction_id=-1`` 为零初值。
    """
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_directions, len(indices)))
    out: list[tuple[float, int, np.ndarray]] = []
    if include_zero:
        out.append((0.0, -1, np.zeros(len(indices))))
    for target in rms_levels:
        for k, d in enumerate(directions):
            out.append((target, k, d * target / pupil_rms(d, indices, grid)))
    return out


# --------------------------------------------------------------------------- #
# 帧设计
# --------------------------------------------------------------------------- #
def build_designs(
    truth=None,
) -> "OrderedDict[str, dict]":
    """帧设计网格：label -> {forward, frames, modulations, carrier}。

    调制序列（无调制帧为 None）：

    * raw1      单帧无调制
    * dx_1/8    单帧 x 相移 t=1/8；dx_1/4 同理；dy_1/8 为 y 相移
    * dx+dy     dx(1/8) + dy(1/8) 两帧
    * dx4       t=i/4, i=0..3；dx8 为 t=i/8, i=0..7；dx8dy8 为两方向各 8 帧
    * carrier   单帧载频（p=30 um 配置，f0=1/(2s)）
    """
    fm = ForwardModel(CFG_SHIFT)
    fm_c = ForwardModel(CFG_CARRIER)
    if truth is None:
        truth = zernike_wavefront(TRUTH_C, TRUTH_IDX)

    dx = lambda t: fm.phase_shift_deltas(t, 0.0)
    dy = lambda t: fm.phase_shift_deltas(0.0, t)

    def shift_frames(deltas_list):
        frames = [fm.intensity(truth, deltas=d) for d in deltas_list]
        return frames, list(deltas_list)

    designs: "OrderedDict[str, dict]" = OrderedDict()

    def add(label, forward, frames, mods, carrier=False):
        designs[label] = dict(
            forward=forward, frames=frames, modulations=mods, carrier=carrier
        )

    add("raw1", fm, [fm.intensity(truth)], [None])
    add("dx_1/8", fm, *shift_frames([dx(1 / 8)]))
    add("dx_1/4", fm, *shift_frames([dx(1 / 4)]))
    add("dy_1/8", fm, *shift_frames([dy(1 / 8)]))
    add("dx+dy_1/8", fm, *shift_frames([dx(1 / 8), dy(1 / 8)]))
    add("dx4", fm, *shift_frames([dx(i / 4) for i in range(4)]))
    add("dx8", fm, *shift_frames([dx(i / 8) for i in range(8)]))
    add(
        "dx8+dy8",
        fm,
        *shift_frames(
            [dx(i / 8) for i in range(8)] + [dy(i / 8) for i in range(8)]
        ),
    )
    add(
        "carrier_p30",
        fm_c,
        [fm_c.carrier_frame(truth)],
        [fm_c.carrier_phases(fm_c.config.carrier_f0)],
        carrier=True,
    )
    return designs


def frame_cache_and_meas(design: dict, indices: Sequence[int], samples: int):
    """设计在采样像素上的 (rows, cache, meas)；与 LM 内部一致。

    ``design["modulations"]`` 每项为 (n_orders,) 相移标量或
    (n_orders, n, n) 载频；后者在此展平并按采样行切片为
    (n_orders, n_rows)。
    """
    fm = design["forward"]
    rows = sample_rows(fm, samples)
    cache = fm.zernike_samples(indices, rows)
    meas = np.concatenate([fr.ravel()[rows] for fr in design["frames"]])
    mods = []
    for m in design["modulations"]:
        if m is None:
            mods.append(None)
        else:
            m = np.asarray(m, dtype=float)
            mods.append(m if m.ndim == 1 else m.reshape(len(cache), -1)[:, rows])
    return rows, cache, meas, mods


def design_symmetry_metric(
    design: dict, coeffs: np.ndarray, indices: Sequence[int], samples: int
) -> float:
    """max over 帧与像素 |I(c) - I(-c)|：0 表示 -c 仍是零代价解（± 简并）。"""
    _, cache, _, mods = frame_cache_and_meas(design, indices, samples)
    worst = 0.0
    for m in mods:
        Ip, _ = _frame_and_jacobian(cache, coeffs, m)
        Im, _ = _frame_and_jacobian(cache, -coeffs, m)
        worst = max(worst, float(np.max(np.abs(Ip - Im))))
    return worst


# --------------------------------------------------------------------------- #
# 收敛点分类
# --------------------------------------------------------------------------- #
def classify_convergence(
    coeffs: np.ndarray, converged: bool, truth: np.ndarray, tol: float = 1e-6
) -> str:
    """+W / -W / other / nonconv。"""
    if not converged:
        return "nonconv"
    if np.max(np.abs(coeffs - truth)) < tol:
        return "+W"
    if np.max(np.abs(coeffs + truth)) < tol:
        return "-W"
    return "other"


def cluster_points(
    vectors: Sequence[np.ndarray], tol: float = 1e-4
) -> list[tuple[np.ndarray, list[int]]]:
    """贪心距离聚类：max|c - rep| < tol 归入既有簇，否则开新簇。

    返回 ``[(代表向量=首个成员, 成员下标列表)]``。系数空间的逐系数
    max 距离阈值聚类，避免网格舍入把单元边界两侧的同一点劈成两簇。
    """
    reps: list[np.ndarray] = []
    members: list[list[int]] = []
    for i, v in enumerate(vectors):
        v = np.asarray(v, dtype=float)
        for j, rep in enumerate(reps):
            if np.max(np.abs(v - rep)) < tol:
                members[j].append(i)
                break
        else:
            reps.append(v.copy())
            members.append([i])
    return list(zip(reps, members))
