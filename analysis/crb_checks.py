"""crb-efficiency.md 若干定量论断的最小验证（对应 review MAJOR-2）。

    uv run python analysis/crb_checks.py

    check1  剪切区受限 FIM：每帧只保留其剪切方向三重交掩膜内的行后
            (JᵀJ)⁻¹ 对角膨胀 —— 证明解调链 ~5x 方差损失并非"少用了像素"。
    check2  逐种子线性恒等式：LM 偏差 ≡ (JᵀJ)⁻¹Jᵀε（同噪声实现）——
            LM 即 BLUE 线性估计器本身，是比 Wishart omnibus 更强的
            贴界证据。
    check3  容差不变性：FTOL/XTOL/GTOL→0 + MAX_ITER 120→400 后，逐种子
            估计与默认容差一致（max|Δc|=2.4e-13）——排除"早停收缩"
            伪影解释。
    check4  σ=0 确定性偏差：干净帧上载频链的系数偏差地板
            （Z2/Z7/Z10 ~1e-3 waves），及其幅值阈值+腐蚀掩膜的实际
            行数 vs 名义剪切区行数。
    check5  Wishart_12(63) 边缘涨落：零假设模拟中 min-eig 跌出 MP
            渐近下沿 0.318 的频率，及实测 0.253 的经验分位 ——
            MP 带是渐近支撑而非硬判据。
"""

from __future__ import annotations

import numpy as np

import lsi.lm as lm_mod
from crb_efficiency import (
    INDICES,
    TRUTH_VEC,
    make_designs,
    shear_region,
)
from lsi.invert import fourier_to_wavefront, phase_shift_to_wavefront
from lsi.lm import _frame_and_jacobian, fit_wavefront_from_frames

A, B = make_designs()          # phase_shift / carrier


def stacked_jacobian(design, mask_rows=None):
    """每帧的 J 逐行堆叠；mask_rows[k] 可限定帧 k 只取指定行。"""
    fm = design.fm
    rows = np.arange(fm.shape[0] * fm.shape[1])
    cache = fm.zernike_samples(INDICES, rows)
    n_orders = len(fm.order_list)
    parts = []
    for k, m in enumerate(design.mods):
        if m is not None and np.asarray(m).ndim > 1:
            m = np.asarray(m).reshape(n_orders, -1)[:, rows]
        _, J = _frame_and_jacobian(cache, TRUTH_VEC, m)
        parts.append(J if mask_rows is None else J[mask_rows[k]])
    return np.vstack(parts)


# --------------------------------------------------------------------------- #
def check_restricted_fim() -> None:
    print("== check1 剪切区受限 FIM（解调链实际使用的像素上的 CRB）==")
    for d in (A, B):
        masks = [shear_region(d.fm, dir_).ravel() for dir_ in ("x", "y")]
        # 每帧只保留对应方向的剪切区行（载频单帧取两方向并集）
        if d.key == "phase_shift":
            per_frame = [masks[0]] * 8 + [masks[1]] * 8
        else:
            per_frame = [masks[0] | masks[1]]
        J_full = stacked_jacobian(d)
        F_full = J_full.T @ J_full
        J_res = stacked_jacobian(d, per_frame)
        F_res = J_res.T @ J_res
        ratio = np.diag(np.linalg.inv(F_res)) / np.diag(np.linalg.inv(F_full))
        print(f"  {d.key}: 系数 CRB 方差膨胀  中位 {np.median(ratio)-1:.3f}"
              f"  范围 [{ratio.min()-1:.3f}, {ratio.max()-1:.3f}]")


# --------------------------------------------------------------------------- #
def check_linear_identity(n_seeds: int = 16, snrs=(60, 20)) -> None:
    print("== check2 逐种子恒等式：ĉ_lm − c*  vs  (JᵀJ)⁻¹Jᵀε ==")
    for d in (A, B):
        J_all = stacked_jacobian(d)
        H = J_all.T @ J_all
        n_frames = d.frames.shape[0]
        for snr in snrs:
            corrs, diffs = [], []
            for seed in range(n_seeds):
                rng = np.random.default_rng([{"phase_shift": 11,
                                              "carrier": 22}[d.key], seed])
                eps = d.sigma(snr) * rng.normal(size=d.frames.shape)
                pred = np.linalg.solve(H, J_all.T @ eps.reshape(n_frames, -1)
                                       .ravel())
                res = (fit_wavefront_from_frames(d.fm, INDICES, d.frames + eps,
                                                d.mods, samples=None)
                       if d.key == "phase_shift" else
                       lm_mod.fit_wavefront_from_carrier_frame(
                           d.fm, INDICES, (d.frames + eps)[0], samples=None))
                dev = res.coeffs - TRUTH_VEC
                corrs.append(np.corrcoef(dev, pred)[0, 1])
                diffs.append(np.sqrt(np.mean((dev - pred) ** 2)))
            print(f"  {d.key} @{snr}dB: 逐种子 corr 中位 "
                  f"{np.median(corrs):.6f}（min {np.min(corrs):.6f}），"
                  f"rms(差) 中位 {np.median(diffs):.2e} waves")


# --------------------------------------------------------------------------- #
def check_tolerance_invariance(n_seeds: int = 8, snr: int = 60) -> None:
    print("== check3 容差收紧（FTOL=XTOL=GTOL=0, MAX_ITER 400）不变性 ==")
    tight = dict(_FTOL=0.0, _XTOL=0.0, _GTOL=0.0, _MAX_ITER=400)
    bak = {k: getattr(lm_mod, k) for k in tight}
    d = A

    def fit(seed):
        rng = np.random.default_rng([11, seed])
        eps = d.sigma(snr) * rng.normal(size=d.frames.shape)
        return fit_wavefront_from_frames(
            d.fm, INDICES, d.frames + eps, d.mods, samples=None).coeffs

    ref = [fit(s) for s in range(n_seeds)]
    try:
        for k, v in tight.items():
            setattr(lm_mod, k, v)
        diffs = [np.abs(a - fit(s)).max() for s, a in enumerate(ref)]
    finally:
        for k, v in bak.items():
            setattr(lm_mod, k, v)
    print(f"  phase_shift @{snr}dB × {n_seeds} seeds: "
          f"max|Δcoeff| = {max(diffs):.3e}（0 = 逐位一致）")


# --------------------------------------------------------------------------- #
def check_bias_floor() -> None:
    print("== check4 σ=0 确定性偏差地板 + 掩膜行数 ==")
    fit, diff = fourier_to_wavefront(B.fm, B.frames[0], indices=INDICES)
    bias = fit.coeffs - TRUTH_VEC
    print("  carrier demod 干净帧偏差:",
          {f"Z{j}": f"{bias[k]:+.2e}" for k, j in enumerate(INDICES)
           if abs(bias[k]) > 1e-4})
    n_nom = int(shear_region(B.fm, "x").sum() + shear_region(B.fm, "y").sum())
    n_eff = int(diff.mask["x"].sum() + diff.mask["y"].sum())
    print(f"  载频重构实际行数 {n_eff} vs 名义 ΔZ 行数 {n_nom}"
          "（x/y 两方向剪切区行数之和）")
    fit_ps, _ = phase_shift_to_wavefront(
        A.fm, A.frames[:8], A.frames[8:], indices=INDICES)
    print(f"  phase_shift demod 干净帧最大偏差 "
          f"{np.abs(fit_ps.coeffs - TRUTH_VEC).max():.2e}")


# --------------------------------------------------------------------------- #
def check_wishart_edge(n_rep: int = 2000, dof: int = 63, p: int = 12) -> None:
    # dof=63 与实测值 0.253 耦合到主实验的 64 seeds / phase_shift LM 行
    print("== check5 Wishart 边缘涨落（H0: 估计器恰达 CRB）==")
    rng = np.random.default_rng(0)
    lo, hi = (1 - np.sqrt(p / dof)) ** 2, (1 + np.sqrt(p / dof)) ** 2
    lam_min = np.empty(n_rep)
    lam_max = np.empty(n_rep)
    for i in range(n_rep):
        X = rng.normal(size=(dof, p))
        ev = np.linalg.eigvalsh(X.T @ X / dof)
        lam_min[i], lam_max[i] = ev[0], ev[-1]
    q = float(np.mean(lam_min <= 0.253))
    print(f"  MP 渐近支撑 [{lo:.3f}, {hi:.3f}]；零假设下 "
          f"P(λ_min<{lo:.3f})={np.mean(lam_min < lo):.3f}，"
          f"P(λ_max>{hi:.3f})={np.mean(lam_max > hi):.3f}")
    print(f"  实测 λ_min=0.253 的经验分位 ≈ {q*100:.1f}%")


if __name__ == "__main__":
    check_restricted_fim()
    check_linear_identity()
    check_tolerance_invariance()
    check_bias_floor()
    check_wishart_edge()
