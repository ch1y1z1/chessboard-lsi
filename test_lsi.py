"""核心算法的端到端回归测试：光栅级次 -> 前向模型 -> 解调 -> 重构 -> LM。

运行：  pytest test_lsi.py
"""

from __future__ import annotations

import numpy as np
import pytest

from lsi.invert import (
    fourier_to_wavefront,
    lsq_phase_shift,
    phase_shift_to_wavefront,
    unwrap_poisson,
    wrap,
)
from lsi.lm import _frame_and_jacobian, fit_wavefront_from_frames
from lsi.model import (
    DEFAULT_ORDERS,
    FRINGE_MODES,
    ForwardModel,
    Grid,
    SystemConfig,
    chessboard_orders,
    diffraction_efficiency,
    zernike,
    zernike_wavefront,
)

CFG = SystemConfig(grid=Grid(n=64, extent=1.10))
INDICES = np.arange(2, 14)
TRUTH_IDX = np.array([4, 6, 7])
TRUTH_C = np.array([0.31, -0.12, 0.42])
truth = zernike_wavefront(TRUTH_C, TRUTH_IDX)


def paper_x5_intensity(W, x, y, s, delta, A0=0.5, A1=2.0 / np.pi**2):
    """论文式 (2-14) 五光束区光强的逐字转写，作前向模型的对照锚点。"""
    tp = 2.0 * np.pi
    w0 = W(x, y)
    wxp, wxm = W(x + s, y), W(x - s, y)
    wyp, wym = W(x, y + s), W(x, y - s)
    hxs, hxd = (wxp + wxm) / 2.0, (wxp - wxm) / 2.0   # x 方向半和/半差
    I = A0**2 + 4.0 * A1**2
    I += 2.0 * A0 * A1 * (np.cos(tp * (wyp - w0)) + np.cos(tp * (wym - w0)))
    I += 2.0 * A1**2 * np.cos(tp * (wyp - wym))
    for coeff, shift in ((4.0 * A0 * A1, w0), (4.0 * A1**2, wyp), (4.0 * A1**2, wym)):
        I += coeff * np.cos(tp * (hxs - shift)) * np.cos(tp * hxd + delta)
    return I + 2.0 * A1**2 * np.cos(tp * 2.0 * hxd + 2.0 * delta)


def test_zernike_spot_values():
    """Z4 = 2rho^2 - 1（离焦）; Z7 = (3rho^3 - 2rho)cos(theta) 初级彗差。"""
    x, y = np.array([0.0, 0.5]), np.array([0.0, 0.0])
    np.testing.assert_allclose(zernike(4, x, y), 2 * x**2 - 1)
    np.testing.assert_allclose(zernike(7, x, y), (3 * x**3 - 2 * x))
    assert zernike(4, np.array([0.0]), np.array([0.0]))[0] == pytest.approx(-1.0)
    assert zernike(7, np.array([1.0]), np.array([0.0]))[0] == pytest.approx(1.0)


def test_zernike_rejects_out_of_range_index():
    """Fringe 序是 1-based：0 或越界序号必须报错而非静默取到别的模式。"""
    x, y = np.array([0.5]), np.array([0.0])
    with pytest.raises(ValueError):
        zernike(0, x, y)
    with pytest.raises(ValueError):
        zernike(len(FRINGE_MODES) + 1, x, y)


def test_lsq_phase_shift_closed_form():
    """合成帧 I = B + C cosδ_i + S sinδ_i（含非零背景）直接验闭式解。"""
    B, C, S, n = 1.7, -0.4, 0.25, 8
    deltas = 2.0 * np.pi * np.arange(n) / n
    frames = np.broadcast_to(
        B + C * np.cos(deltas)[:, None, None] + S * np.sin(deltas)[:, None, None],
        (n, 5, 5),
    )
    phase, modulation = lsq_phase_shift(frames)
    np.testing.assert_allclose(phase, np.arctan2(-S, C), atol=1e-12)
    np.testing.assert_allclose(modulation, np.hypot(C, S), rtol=1e-12)


def test_piston_mode_is_rejected():
    """Z1 平移对差分与光强都不可观测：两条反演入口都必须拒绝。"""
    fm = ForwardModel(CFG)
    fx = fm.phase_shift_frames(truth, "x", 8)
    fy = fm.phase_shift_frames(truth, "y", 8)
    with pytest.raises(ValueError):
        phase_shift_to_wavefront(fm, fx, fy, indices=[1, 4, 7])
    deltas = [fm.phase_shift_deltas(k / 8, 0.0) for k in range(8)]
    with pytest.raises(ValueError):
        fit_wavefront_from_frames(fm, [1, 4, 7], fx, deltas)


def test_grating_orders_match_table_2_3():
    orders = chessboard_orders(max_index=9)
    eff = diffraction_efficiency(orders)
    assert eff["dc"] == pytest.approx(0.25, abs=1e-12)
    assert eff["first_order_total"] / 4 == pytest.approx(0.0411, abs=1e-4)
    assert abs(orders[(1.0, 0.0)]) == pytest.approx(2.0 / np.pi**2, abs=1e-12)
    # 符号约定：x 对为负、y 对为正（决定解调相位的半条纹偏移）
    for k, v in DEFAULT_ORDERS.items():
        assert orders[k] == pytest.approx(v)


def test_forward_matches_paper_region_formula():
    """级次叠加模型 == 论文显式五光束公式 (2-14) 的逐点值。

    论文公式把四个 ±1 级写成同一正振幅 A1；对应的叠加模型取全正振幅
    （真实棋盘的符号差异只体现在解调常数上，见 demodulation_offset）。
    """
    amps = {o: (0.5 if o == (0.0, 0.0) else 2.0 / np.pi**2) for o in DEFAULT_ORDERS}
    fm = ForwardModel(CFG, amps)
    x, y = CFG.grid.coords()
    s = CFG.s
    mask = np.ones(CFG.grid.shape, bool)
    for a, b in DEFAULT_ORDERS:
        mask &= (x + a * s) ** 2 + (y + b * s) ** 2 <= 1.0
    for t in (0.0, 0.125, 0.3):
        model = fm.intensity(truth, deltas=fm.phase_shift_deltas(t, 0.0))
        paper = paper_x5_intensity(truth, x, y, s, 2 * np.pi * t)
        np.testing.assert_allclose(model[mask], paper[mask], atol=1e-12)


def test_unwrap_recovers_smooth_phase():
    phi = np.linspace(-3.0, 3.0, 40)[:, None] + np.zeros((1, 40))
    out = unwrap_poisson(wrap(phi), np.ones((40, 40), bool))
    diff = out - np.nanmean(out) - (phi - phi.mean())
    assert np.nanmax(np.abs(diff)) < 1e-8


def test_phase_shift_pipeline_recovers_wavefront():
    """端到端：16 帧相移 -> Zernike 系数。"""
    fm = ForwardModel(CFG)
    fit, _ = phase_shift_to_wavefront(
        fm, fm.phase_shift_frames(truth, "x", 8),
        fm.phase_shift_frames(truth, "y", 8), indices=INDICES,
    )
    tab = fit.as_dict()
    for j, c in zip(TRUTH_IDX, TRUTH_C):
        assert tab[j] == pytest.approx(c, abs=1e-9)
    assert max(abs(v) for j, v in tab.items() if j not in TRUTH_IDX) < 1e-9


def test_fourier_pipeline_recovers_wavefront():
    """端到端：单帧载频图 -> 瓣解调 -> 解包裹 -> 差分 Zernike。"""
    cfg = SystemConfig(grid=Grid(n=128, extent=1.10), period_um=30.0)
    fm = ForwardModel(cfg)
    w = zernike_wavefront([0.5], [7])
    fit, _ = fourier_to_wavefront(fm, fm.carrier_frame(w), indices=INDICES)
    tab = fit.as_dict()
    assert tab[7] == pytest.approx(0.5, abs=0.05)
    assert max(abs(v) for j, v in tab.items() if j != 7) < 0.05


def test_lm_jacobian_matches_finite_difference():
    """解析雅可比 vs 中心差分（LM 正确性的关键）。"""
    fm = ForwardModel(CFG)
    cache = fm.zernike_samples([4, 7], np.arange(0, 64 * 64, 37))
    c = np.array([0.2, 0.5])
    deltas = fm.phase_shift_deltas(0.25, 0.0)
    _, J = _frame_and_jacobian(cache, c, deltas, None)
    for j in range(2):
        dp, dm = c.copy(), c.copy()
        dp[j] += 1e-7
        dm[j] -= 1e-7
        Ip, _ = _frame_and_jacobian(cache, dp, deltas, None)
        Im, _ = _frame_and_jacobian(cache, dm, deltas, None)
        np.testing.assert_allclose(J[:, j], (Ip - Im) / 2e-7, rtol=1e-5, atol=1e-8)


def test_lm_recovers_coefficients_from_frames():
    """LM 直接由光强帧恢复系数（不经解调/解包裹）。"""
    fm = ForwardModel(CFG)
    frames = np.concatenate(
        [fm.phase_shift_frames(truth, "x", 8),
         fm.phase_shift_frames(truth, "y", 8)], axis=0)
    deltas = [fm.phase_shift_deltas(k / 8, 0.0) for k in range(8)]
    deltas += [fm.phase_shift_deltas(0.0, k / 8) for k in range(8)]
    res = fit_wavefront_from_frames(fm, INDICES, frames, deltas, samples=4000)
    tab = res.as_dict()
    for j, c in zip(TRUTH_IDX, TRUTH_C):
        assert tab[j] == pytest.approx(c, abs=1e-8)
    assert res.rms_residual < 1e-10
