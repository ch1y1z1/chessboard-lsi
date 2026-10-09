"""J^T r 直算接口（lsi.gradient）与显式雅可比路径的等价性回归。

对照基准是 lm._frame_and_jacobian 的物化 J：两种实现仅在浮点
求和顺序上不同，预期差异 ~1e-12 相对量级（大尺度情形按 eps*scale
解释，不套固定绝对阈值）。方向差分复核 jtr = ∇(½||I-target||²)。
"""
import numpy as np
import pytest

from lsi.gradient import frame_intensity_jtr
from lsi.lm import _frame_and_jacobian
from lsi.model import ForwardModel, Grid, SystemConfig

# 载频（傅里叶）模式几何：period_um=30 -> s=0.0439, f0=11.4
CARRIER = dict(period_um=30.0)
PARAM_SETS = {12: list(range(2, 14)), 20: list(range(2, 22)),
              35: list(range(2, 37))}


def _forward(n, carrier=True):
    return ForwardModel(SystemConfig(
        grid=Grid(n=n), **(CARRIER if carrier else {})))


def _rows(fm, kind):
    """full / 固定坏点剔除 / 确定性子采样。"""
    n_pix = fm.shape[0] * fm.shape[1]
    if kind == "full":
        return np.arange(n_pix)
    if kind == "bad_pixels":
        mask = np.ones(n_pix, dtype=bool)
        mask[np.arange(0, n_pix, 7919)] = False          # 稀疏坏点列
        mask[12345:12360] = False                        # 连续坏块
        return np.flatnonzero(mask)
    if kind == "subsample":
        return np.sort(
            (np.arange(4096) * 0.6180339887498949 % 1.0 * n_pix).astype(int))
    raise ValueError(kind)


def _modulation(fm, rows, kind):
    """None / 每级标量相移 / 按 rows 采样的空间载频相位。"""
    if kind == "none":
        return None
    if kind == "scalar":
        return fm.phase_shift_deltas(0.37, -0.11)
    if kind == "spatial":
        return fm.carrier_phases(fm.config.carrier_f0).reshape(
            len(fm.order_list), -1)[:, rows]
    raise ValueError(kind)


def _coeffs(rng, n_terms, rms):
    c = rng.normal(size=n_terms)
    return c * (rms / np.linalg.norm(c) * np.sqrt(n_terms))


def _assert_jtr_close(jtr, jtr_ref, *, rtol=5e-12, atol_floor=1e-13):
    """绝对+相对混合容差：允许 eps*scale 量级误差，报告实际量级。"""
    scale = max(1.0, float(np.abs(jtr_ref).max()))
    diff = float(np.abs(jtr - jtr_ref).max())
    assert diff <= atol_floor + rtol * scale, (
        f"|Δjtr|max={diff:.3e} 超过 {atol_floor:.1e}+{rtol:.0e}*{scale:.3e}")
    return diff, diff / scale


CASES = [
    # (n, n_params, rows, modulation, coeff_rms, residual_kind)
    (128, 12, "full", "none", 0.1, "model"),
    (128, 12, "subsample", "spatial", 0.1, "model"),
    (128, 12, "bad_pixels", "spatial", 0.1, "large"),
    (128, 20, "bad_pixels", "scalar", 0.1, "model"),
    (128, 20, "subsample", "none", 3.0, "large"),
    (128, 35, "full", "spatial", 0.1, "model"),
    (128, 35, "bad_pixels", "spatial", 3.0, "large"),
    (128, 35, "subsample", "scalar", 1.0, "model"),
    (256, 12, "subsample", "spatial", 0.1, "model"),
    (256, 35, "bad_pixels", "spatial", 0.3, "large"),
    (64, 20, "full", "none", 3.0, "large"),   # 无载频消融同口径
]


@pytest.mark.parametrize(
    "n,n_params,row_kind,mod_kind,coeff_rms,residual_kind", CASES)
def test_matches_explicit_jacobian(n, n_params, row_kind, mod_kind,
                                   coeff_rms, residual_kind):
    fm = _forward(n)
    indices = PARAM_SETS[n_params]
    rows = _rows(fm, row_kind)
    cache = fm.zernike_samples(indices, rows)
    modulation = _modulation(fm, rows, mod_kind)
    rng = np.random.default_rng(20261009 + n + n_params)
    coeffs = _coeffs(rng, len(indices), coeff_rms)

    i_ref, j_ref = _frame_and_jacobian(cache, coeffs, modulation)
    if residual_kind == "model":
        target = i_ref.copy()                   # 小残差：r≈0
        target[::7] += 1e-6                     # 保留微小非零残差
    else:
        other = _coeffs(rng, len(indices), coeff_rms)
        target, _ = _frame_and_jacobian(cache, other, modulation,
                                        compute_jacobian=False)
        target += 0.05                          # 大且有偏的残差

    intensity, jtr = frame_intensity_jtr(cache, coeffs, target, modulation)
    np.testing.assert_allclose(intensity, i_ref, atol=1e-13, rtol=1e-13)
    jtr_ref = j_ref.T @ (i_ref - target)
    _assert_jtr_close(jtr, jtr_ref)


def test_zero_residual_jtr_is_zero():
    fm = _forward(128)
    indices = PARAM_SETS[20]
    rows = _rows(fm, "subsample")
    cache = fm.zernike_samples(indices, rows)
    modulation = _modulation(fm, rows, "spatial")
    rng = np.random.default_rng(7)
    coeffs = _coeffs(rng, len(indices), 0.2)
    target, j_ref = _frame_and_jacobian(cache, coeffs, modulation)
    _, jtr = frame_intensity_jtr(cache, coeffs, target, modulation)
    # r=0 时 jtr 应小到浮点噪声；上界取雅可比尺度而非固定绝对值。
    jac_scale = float(np.abs(j_ref).max())
    assert np.abs(jtr).max() <= 1e-12 * max(1.0, jac_scale)


def test_directional_derivative_finite_difference():
    fm = _forward(128)
    indices = PARAM_SETS[12]
    rows = _rows(fm, "bad_pixels")
    cache = fm.zernike_samples(indices, rows)
    modulation = _modulation(fm, rows, "spatial")
    rng = np.random.default_rng(3)
    coeffs = _coeffs(rng, len(indices), 0.15)
    target, _ = _frame_and_jacobian(
        cache, _coeffs(rng, len(indices), 0.15), modulation,
        compute_jacobian=False)

    def loss(c):
        i, _ = frame_intensity_jtr(cache, c, target, modulation)
        r = i - target
        return 0.5 * float(r @ r)

    _, jtr = frame_intensity_jtr(cache, coeffs, target, modulation)
    for _ in range(4):
        v = rng.normal(size=coeffs.size)
        v /= np.linalg.norm(v)
        h = 1e-7 * (1.0 + np.linalg.norm(coeffs))
        fd = (loss(coeffs + h * v) - loss(coeffs - h * v)) / (2.0 * h)
        directional = float(jtr @ v)
        scale = max(1.0, abs(directional))
        assert abs(fd - directional) <= 1e-6 * scale + 1e-10, (
            f"方向导数不符: fd={fd:.6e} jtr·v={directional:.6e}")


def test_input_validation():
    fm = _forward(64)
    indices = PARAM_SETS[12]
    rows = _rows(fm, "subsample")
    cache = fm.zernike_samples(indices, rows)
    coeffs = np.zeros(len(indices))
    target = np.zeros(rows.size)
    with pytest.raises(ValueError):
        frame_intensity_jtr(cache, coeffs, np.zeros(rows.size + 1))
    with pytest.raises(ValueError):
        frame_intensity_jtr(cache, np.zeros(len(indices) + 1), target)
    with pytest.raises(ValueError):
        frame_intensity_jtr(cache, coeffs, target,
                            np.zeros(len(fm.order_list) + 1))
    with pytest.raises(ValueError):
        frame_intensity_jtr(
            cache, coeffs, target,
            np.zeros((len(fm.order_list), rows.size + 1)))
    with pytest.raises(ValueError):
        frame_intensity_jtr([], coeffs, np.zeros(0))
    bad_cache = [(k, amp, inside.astype(int), Z)
                 for k, amp, inside, Z in cache]
    with pytest.raises(ValueError):
        frame_intensity_jtr(bad_cache, coeffs, target)
    with pytest.raises(ValueError):
        frame_intensity_jtr(cache, coeffs, np.full(rows.size, np.nan))
