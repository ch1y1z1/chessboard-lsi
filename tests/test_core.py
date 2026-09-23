"""核心算法的最小回归测试。

覆盖论文计算链条的每一级：

    光栅级次 -> 前向模型 -> 相移/傅里叶解调 -> 差分 Zernike -> LM 反演

运行：  pytest tests/test_core.py
"""

from __future__ import annotations

import numpy as np
import pytest

from lsi.config import Grid, SystemConfig
from lsi.forward import (
    ForwardModel,
    ZernikeWavefront,
    paper_region_intensity,
)
from lsi.ftmode import demodulate_lobe
from lsi.grating import chessboard_orders, diffraction_efficiency
from lsi.lm import LMConfig, fit_wavefront_from_frames
from lsi.phaseshift import lsq_phase_shift, shear_regions
from lsi.pipeline import phase_shift_to_wavefront
from lsi.reconstruct import fit_differential_zernike
from lsi.unwrap import unwrap_poisson, wrap
from lsi.zernike import differential_zernike_matrix, zernike

CFG = SystemConfig(grid=Grid(n=64, extent=1.10))
INDICES = tuple(range(2, 14))


def test_zernike_spot_values():
    """Z4 = 2rho^2 - 1（离焦）; Z7 是 Fringe 序的 (3,1)cos 初级彗差。"""
    x, y = np.array([0.0, 0.5]), np.array([0.0, 0.0])
    np.testing.assert_allclose(zernike(4, x, y), 2 * x**2 - 1)
    np.testing.assert_allclose(zernike(7, x, y), (3 * x**3 - 2 * x))
    # 中心处离焦 = -1；顶点处 Z7 = 1
    assert zernike(4, np.array([0.0]), np.array([0.0]))[0] == pytest.approx(-1.0)
    assert zernike(7, np.array([1.0]), np.array([0.0]))[0] == pytest.approx(1.0)


def test_grating_orders_match_table_2_3():
    orders = chessboard_orders(max_index=9)
    eff = diffraction_efficiency(orders)
    assert eff["dc"] == pytest.approx(0.25, abs=1e-12)
    assert eff["first_order_total"] / 4 == pytest.approx(0.0411, abs=1e-4)
    assert abs(orders[(1.0, 0.0)]) == pytest.approx(2.0 / np.pi**2, abs=1e-12)


def test_forward_matches_paper_region_formulas():
    """统一的级次叠加 == 论文式 (2-12)...(2-16) 的逐点值。"""
    fm = ForwardModel(CFG)
    x, y = CFG.grid.coords()
    s = CFG.s
    wf = ZernikeWavefront([0.4, -0.1, 0.2], [7, 4, 11])
    cases = {
        "x1": [(0, 0), (1, 0), (-1, 0), (0, 1)],
        "x2": [(0, 0), (1, 0), (-1, 0), (0, -1)],
        "x5": [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)],
        "y1": [(0, 0), (0, 1), (0, -1), (1, 0)],
    }
    for region, orders in cases.items():
        amps = {tuple(map(float, o)): (0.5 if o == (0, 0) else 2.0 / np.pi**2)
                for o in orders}
        sub = ForwardModel(CFG, amps)
        for t in (0.0, 0.125, 0.3):
            deltas = (sub.phase_shift_deltas(t, 0.0) if region.startswith("x")
                      else sub.phase_shift_deltas(0.0, t))
            model = sub.intensity(wf, deltas=deltas)
            mask = np.ones(CFG.grid.shape, bool)
            for a, b in orders:
                mask &= (x + a * s) ** 2 + (y + b * s) ** 2 <= 1.0
            paper = paper_region_intensity(region, wf.w, x, y, s, 2 * np.pi * t)
            np.testing.assert_allclose(model[mask], paper[mask], atol=1e-12)


def test_phase_shift_demodulation_recovers_phase():
    """合成 I = B + M cos(delta + psi)，解调应还原 psi、M、B。"""
    rng = np.random.default_rng(0)
    B = 1.0 + 0.1 * rng.random((8, 8))
    M = 0.5 + 0.1 * rng.random((8, 8))
    psi = 0.7 + 0.2 * rng.random((8, 8))
    deltas = 2 * np.pi * np.arange(8) / 8
    frames = (B[None] + M[None] * np.cos(deltas[:, None, None] + psi[None]))
    res = lsq_phase_shift(frames)
    np.testing.assert_allclose(res.phase, np.angle(np.exp(1j * psi)), atol=1e-12)
    np.testing.assert_allclose(res.modulation, M, atol=1e-12)
    np.testing.assert_allclose(res.background, B, atol=1e-12)
    # 非均匀步长同样成立（一般最小二乘，非闭式求和）
    deltas_irr = np.array([0.0, 0.6, 1.7, 2.4, 3.9, 4.4, 5.1, 5.9])
    frames2 = (B[None] + M[None] * np.cos(deltas_irr[:, None, None] + psi[None]))
    res2 = lsq_phase_shift(frames2, deltas_irr)
    np.testing.assert_allclose(res2.phase, np.angle(np.exp(1j * psi)), atol=1e-12)


def test_unwrap_recovers_smooth_phase():
    phi = np.linspace(-3.0, 3.0, 40)[:, None] + np.zeros((1, 40))
    mask = np.ones((40, 40), bool)
    out = unwrap_poisson(wrap(phi), mask)
    diff = out - np.nanmean(out) - (phi - phi.mean())
    assert np.nanmax(np.abs(diff)) < 1e-8


def test_differential_zernike_recovers_coefficients():
    """对解析构造的 dW_x, dW_y 做最小二乘，恢复系数。"""
    x, y = CFG.grid.coords()
    s = CFG.s
    masks = shear_regions(CFG.grid, s)
    truth = {7: 0.5, 4: -0.2, 11: 0.1}
    coeffs = np.array([truth.get(j, 0.0) for j in INDICES])
    dW = {}
    for d in ("x", "y"):
        Z = differential_zernike_matrix(INDICES, x, y, s, d)
        dW[d] = Z @ coeffs
    fit = fit_differential_zernike(
        dW["x"], dW["y"], masks["region_x"], masks["region_y"], s, x, y,
        indices=INDICES,
    )
    np.testing.assert_allclose(fit.coeffs, coeffs, atol=1e-10)


def test_phase_shift_pipeline_recovers_wavefront():
    """端到端：16 帧相移 -> Zernike 系数，误差 ~1e-13。"""
    fm = ForwardModel(CFG)
    truth = ZernikeWavefront([0.31, -0.12, 0.42], [4, 6, 7])
    fit, _ = phase_shift_to_wavefront(
        fm, fm.phase_shift_frames(truth, "x", 8),
        fm.phase_shift_frames(truth, "y", 8), indices=INDICES,
    )
    tab = dict(zip(fit.indices.tolist(), fit.coeffs.tolist()))
    assert tab[7] == pytest.approx(0.42, abs=1e-9)
    assert tab[4] == pytest.approx(0.31, abs=1e-9)
    assert max(abs(v) for j, v in tab.items() if j not in truth.indices) < 1e-9


def test_lm_jacobian_matches_finite_difference():
    """解析雅可比 vs 中心差分（LM 正确性的关键）。"""
    fm = ForwardModel(CFG)
    indices = np.array([4, 7])
    cache = fm.zernike_samples(indices, np.arange(0, 64 * 64, 37))
    c = np.array([0.2, 0.5])
    deltas = fm.phase_shift_deltas(0.25, 0.0)

    from lsi.lm import _frame_and_jacobian

    _, J = _frame_and_jacobian(cache, c, deltas, None)
    eps = 1e-7
    for j in range(len(indices)):
        dp, dm = c.copy(), c.copy()
        dp[j] += eps
        dm[j] -= eps
        Ip, _ = _frame_and_jacobian(cache, dp, deltas, None)
        Im, _ = _frame_and_jacobian(cache, dm, deltas, None)
        np.testing.assert_allclose(J[:, j], (Ip - Im) / (2 * eps), rtol=1e-5, atol=1e-8)


def test_lm_recovers_coefficients_from_frames():
    """LM 直接由光强帧恢复系数（不经解调/解包裹）。"""
    fm = ForwardModel(CFG)
    truth = ZernikeWavefront([0.31, -0.12, 0.42], [4, 6, 7])
    frames = np.concatenate(
        [fm.phase_shift_frames(truth, "x", 8),
         fm.phase_shift_frames(truth, "y", 8)], axis=0)
    deltas = [fm.phase_shift_deltas(k / 8, 0.0) for k in range(8)]
    deltas += [fm.phase_shift_deltas(0.0, k / 8) for k in range(8)]
    res = fit_wavefront_from_frames(fm, INDICES, frames, deltas,
                                   samples=4000, config=LMConfig(max_iter=60))
    tab = dict(zip(INDICES, res.x.tolist()))
    assert tab[7] == pytest.approx(0.42, abs=1e-8)
    assert tab[4] == pytest.approx(0.31, abs=1e-8)
    assert res.rms_residual < 1e-10


def test_fourier_lobe_is_two_sided_difference():
    """+f0 瓣携带双边差分 pi [W(x+s) - W(x-s)]（论文式 2-42）。"""
    cfg = SystemConfig(grid=Grid(n=128, extent=1.10), period_um=30.0)
    fm = ForwardModel(cfg)
    truth = ZernikeWavefront([0.5], [7])
    lobe = demodulate_lobe(fm.carrier_frame(truth), cfg.grid, "x",
                           cfg.carrier_f0,
                           phase_offset=fm.demodulation_offset("x"))
    x, y = cfg.grid.coords()
    two_sided = np.pi * (truth.w(x + cfg.s, y) - truth.w(x - cfg.s, y))
    inner = (np.hypot(x, y) < 0.6) & (np.hypot(x - cfg.s, y) < 1) & (
        np.hypot(x + cfg.s, y) < 1)
    residual = wrap(lobe.phase - two_sided)
    residual -= np.angle(np.mean(np.exp(1j * residual[inner])))
    assert np.sqrt(np.mean(residual[inner] ** 2)) < 0.05
