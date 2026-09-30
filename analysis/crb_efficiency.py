"""三条反演路线的统计效率 vs Cramér–Rao 下界（Fisher 信息分析）。

围绕一个可辩护的命题组织计算：iid 高斯噪声下，对光强残差做最小二乘的
LM 就是 MLE，其渐近协方差应达到 FIM 的逆。论文的两条解调链（相移 /
傅里叶载频 -> 解包裹 -> 差分 Zernike 最小二乘）则是有损的前端。

    1. 在真值 c* 处对全部帧装配堆叠雅可比 J_all（每帧用
       ``lsi.lm._frame_and_jacobian``，行集合 = 全部探测器像素），
       FIM = J_all^T J_all / sigma^2，CRB_i = sigma^2 (J^T J)^{-1}_ii。
       sigma 与 experiment.add_noise 同约定：max(I) / 10^(SNR/20)。
    2. J^T J 谱：奇异值与近零方向；与差分 Zernike 设计矩阵 ΔZ 的
       谱/条件数对照。
    3. Monte Carlo：两种帧设计 x 5 个 SNR x N_SEEDS 种子，同一噪声
       实现（common random numbers）喂给解调链与 LM 两条路线。
    4. 离群会计：MAD 稳健 sigma，|Δc-med| > 3 sigma_robust 剔除后重算；
       另报固定物理阈值的灾难率 max|Δc| > 0.05 wave。
    5. Wishart 对照：经验协方差 C_emp 对 CRB 协方差 C_crb 的广义特征值
       应落在 Marchenko–Pastur 支撑 [(1-r)^2, (1+r)^2]（r=sqrt(p/n)）
       附近——比逐系数 eta 更能区分"真失效"与"抽样涨落"。

用法
----
    uv run python analysis/crb_efficiency.py                # 全量
    uv run python analysis/crb_efficiency.py --seeds 8 --snrs 60 30   # 冒烟
    uv run python analysis/crb_efficiency.py --plots-only   # 复用 npz 只重画图/表
        （前置：需先全量跑过生成 output/mc_estimates.npz，且 --snrs 与之一致）

产物写入 output/（.gitignore 内，属可再生实验产物）。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lsi.invert import (  # noqa: E402
    fourier_to_wavefront,
    phase_shift_to_wavefront,
)
from lsi.lm import (  # noqa: E402
    _frame_and_jacobian,
    fit_wavefront_from_carrier_frame,
    fit_wavefront_from_frames,
)
from lsi.model import (  # noqa: E402
    FRINGE_MODES,
    ForwardModel,
    Grid,
    SystemConfig,
    differential_zernike_matrix,
    zernike_wavefront,
)

OUT = Path("output")

# 与 experiment.py 一致的真值与拟合集
INDICES = np.arange(2, 14)                     # Z2..Z13
TRUTH_IDX = np.array([4, 5, 6, 7, 8])
TRUTH_C = np.array([0.31, -0.12, 0.07, 0.42, 0.05])
TRUTH_VEC = np.array(
    [dict(zip(TRUTH_IDX, TRUTH_C)).get(j, 0.0) for j in INDICES]
)
Z_NAME = {j: FRINGE_MODES[j - 1][3] for j in INDICES}

ROUTES = ("demod", "lm")


# --------------------------------------------------------------------------- #
# 帧设计
# --------------------------------------------------------------------------- #
@dataclass
class Design:
    """一种采集设计：前向模型 + 干净帧 + 逐帧调制 + 两条路线入口。"""

    key: str
    label: str
    fm: ForwardModel
    frames: np.ndarray            # (n_frames, n, n) 无噪声
    mods: list                    # 与 fit_wavefront_from_frames 相同的每帧调制

    def sigma(self, snr_db: float) -> float:
        """iid 高斯噪声标准差：max(I)/10^(SNR/20)，对整个帧组取 max。"""
        return float(self.frames.max() / 10.0 ** (snr_db / 20.0))


def make_designs() -> list[Design]:
    truth = zernike_wavefront(TRUTH_C, TRUTH_IDX)

    # (a) 8+8 相移帧：p = 18 um -> s = 0.0731，n = 128
    cfg_a = SystemConfig(grid=Grid(n=128, extent=1.10))
    fm_a = ForwardModel(cfg_a)
    fx = fm_a.phase_shift_frames(truth, "x", 8)
    fy = fm_a.phase_shift_frames(truth, "y", 8)
    frames_a = np.concatenate([fx, fy])
    mods_a = [fm_a.phase_shift_deltas(k / 8, 0.0) for k in range(8)]
    mods_a += [fm_a.phase_shift_deltas(0.0, k / 8) for k in range(8)]

    # (b) 单帧载频：p = 30 um -> s = 0.0439，n = 256
    cfg_b = SystemConfig(grid=Grid(n=256, extent=1.10), period_um=30.0)
    fm_b = ForwardModel(cfg_b)
    frame_b = fm_b.carrier_frame(truth)[None]          # (1, n, n)
    mods_b = [fm_b.carrier_phases(cfg_b.carrier_f0)]

    return [
        Design(
            key="phase_shift",
            label="8+8 相移帧 (n=128, s=0.0731)",
            fm=fm_a,
            frames=frames_a,
            mods=mods_a,
        ),
        Design(
            key="carrier",
            label="单帧载频 (n=256, p=30um, s=0.0439)",
            fm=fm_b,
            frames=frame_b,
            mods=mods_b,
        ),
    ]


def run_demod(design: Design, noisy: np.ndarray):
    if design.key == "phase_shift":
        fit, _ = phase_shift_to_wavefront(
            design.fm, noisy[:8], noisy[8:], indices=INDICES
        )
    else:
        fit, _ = fourier_to_wavefront(design.fm, noisy[0], indices=INDICES)
    return fit


def run_lm(design: Design, noisy: np.ndarray):
    if design.key == "phase_shift":
        return fit_wavefront_from_frames(
            design.fm, INDICES, noisy, design.mods, samples=None
        )
    return fit_wavefront_from_carrier_frame(
        design.fm, INDICES, noisy[0], samples=None
    )


# --------------------------------------------------------------------------- #
# Fisher 信息装配
# --------------------------------------------------------------------------- #
def assemble_fim(design: Design) -> np.ndarray:
    """全部像素上的堆叠 Fisher 信息矩阵 J^T J（未除 sigma^2）。

    行集合 = 全部 n*n 探测器像素。理由：LM 以 samples=None 在全图上拟合，
    解调链也用全图——CRB 因此对每条路线所见的数据都精确成立，不存在
    "界对着子采样、估计器却用全图"的口径差。黄金比例子采样仅在 n 很大
    时才需要（n=128 全图本来就只有 16384 像素）。
    """
    fm = design.fm
    n_pix = fm.shape[0] * fm.shape[1]
    rows = np.arange(n_pix)
    cache = fm.zernike_samples(INDICES, rows)
    n_orders = len(fm.order_list)
    fim = np.zeros((len(INDICES), len(INDICES)))
    for m in design.mods:
        if m is not None and np.asarray(m).ndim > 1:
            m = np.asarray(m).reshape(n_orders, -1)[:, rows]
        _, J = _frame_and_jacobian(cache, TRUTH_VEC, m)
        fim += J.T @ J
    return fim


def shear_region(fm: ForwardModel, direction: str) -> np.ndarray:
    """剪切干涉区 = 0 级与 ±1 级光瞳三重交（与 invert.py 一致）。"""
    a, b = (1.0, 0.0) if direction == "x" else (0.0, 1.0)
    return (
        fm.order_support(0.0, 0.0)
        & fm.order_support(a, b)
        & fm.order_support(-a, -b)
    )


def diff_zernike_design(fm: ForwardModel) -> np.ndarray:
    """名义堆叠差分 Zernike 设计矩阵：x/y 两方向各自剪切区上的 ΔZ 行
    拼接（未加权；两区重叠的像素在两个方向块中各占一行）。

    注意这只是谱对照用的名义矩阵：相移链实际拟合的确用剪切区掩膜，
    而载频链还叠加幅值阈值+边缘腐蚀（行数更少，见 crb_checks.py 的
    check_bias_floor 打印）。
    """
    x, y = fm.grid.coords()
    parts = [
        differential_zernike_matrix(
            INDICES, x[shear_region(fm, d)], y[shear_region(fm, d)], fm.s, d
        )
        for d in ("x", "y")
    ]
    return np.vstack(parts)


def corr_from_cov(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    return cov / np.outer(d, d)


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
def run_monte_carlo(design: Design, snrs: list[int], n_seeds: int):
    """同一噪声实现（common random numbers）喂给两条路线。

    每个种子先抽一张标准正态 z，逐 SNR 缩放 sigma*z —— 既保证同格内
    两条路线严格配对，也使不同 SNR 之间可配对比較。
    """
    n_t = len(INDICES)
    est = {r: {s: np.full((n_seeds, n_t), np.nan) for s in snrs} for r in ROUTES}
    runs = []
    salt = {"phase_shift": 11, "carrier": 22}[design.key]
    for seed in range(n_seeds):
        rng = np.random.default_rng([salt, seed])
        z = rng.normal(size=design.frames.shape)
        for snr in snrs:
            noisy = design.frames + design.sigma(snr) * z
            for route in ROUTES:
                t0 = time.time()
                ok, converged, coeffs, err = True, None, None, ""
                try:
                    res = run_demod(design, noisy) if route == "demod" else run_lm(design, noisy)
                    coeffs = np.asarray(res.coeffs, dtype=float)
                    if route == "lm":
                        converged = bool(res.converged)
                    if not np.all(np.isfinite(coeffs)):
                        ok = False
                except Exception as exc:  # noqa: BLE001 - 失败本身是被统计量
                    ok, err = False, repr(exc)[:120]
                    coeffs = np.full(n_t, np.nan)
                est[route][snr][seed] = coeffs
                d = coeffs - TRUTH_VEC
                runs.append(
                    dict(
                        design=design.key, route=route, snr=snr, seed=seed,
                        ok=ok, converged=converged, wall_s=round(time.time() - t0, 3),
                        max_abs_err=float(np.nanmax(np.abs(d))) if ok else np.nan,
                        error=err,
                    )
                )
    return est, runs


# --------------------------------------------------------------------------- #
# 统计汇总（含 MAD 离群会计）
# --------------------------------------------------------------------------- #
def summarize(est, crb_var: dict, snrs: list[int], design_key: str):
    """est[route][snr] = (n_seeds, n_terms)。返回长表行。"""
    rows = []
    for route in ROUTES:
        for snr in snrs:
            E = est[route][snr]                       # (n_seeds, n_t)
            D = E - TRUTH_VEC[None, :]
            # 固定物理阈值灾难率；NaN 失败行计入分母但不计分子
            # （本数据 0 失败；若有失败则与 n_fail 口径一致地偏保守）
            cat_rate = float(np.mean(
                np.nanmax(np.abs(D), axis=1) > 0.05))
            for j_idx, j in enumerate(INDICES):
                d = D[:, j_idx]
                finite = np.isfinite(d)
                n, n_fail = int(finite.sum()), int((~finite).sum())
                dv = d[finite]
                base = dict(design=design_key, route=route, snr=snr,
                            z=int(j), n_ok=n, n_fail=n_fail,
                            catastrophe_rate=cat_rate,
                            crb_var=crb_var[snr][j_idx])
                if n < 3:
                    rows.append(base)
                    continue
                bias = float(dv.mean())
                var = float(dv.var(ddof=1))
                med = np.median(dv)
                s_rob = max(1.4826 * np.median(np.abs(dv - med)), 1e-12)
                out = np.abs(dv - med) > 3.0 * s_rob
                keep = dv[~out]
                vb = dict(
                    bias=bias, bias2=bias**2, var=var,
                    mse=float(np.mean(dv**2)),
                    out_rate=float(out.mean()), s_robust=float(s_rob),
                    eta=crb_var[snr][j_idx] / var if var > 0 else np.nan,
                    eta_mse=crb_var[snr][j_idx] / np.mean(dv**2),
                )
                if keep.size >= 3:
                    vb.update(
                        bias_clean=float(keep.mean()),
                        bias2_clean=float(keep.mean()) ** 2,
                        var_clean=float(keep.var(ddof=1)),
                        mse_clean=float(np.mean(keep**2)),
                        eta_clean=crb_var[snr][j_idx] / keep.var(ddof=1),
                        eta_mse_clean=crb_var[snr][j_idx]
                        / np.mean(keep**2),
                    )
                rows.append({**base, **vb})
    return rows


def wishart_check(est, crb_covs: dict, snrs: list[int], design_key: str):
    """C_emp 相对 C_crb 的广义特征值谱 + 迹（每条路线 x SNR 一行）。

    MP 带是渐近支撑而非硬接受域：有限 n 下真有效估计量的边缘特征值
    仍有非平凡概率落出带外（Tracy–Widom 涨落），定量参照见
    analysis/crb_checks.py 的零假设模拟。
    """
    from scipy.linalg import eigh
    rows = []
    p = len(INDICES)
    for route in ROUTES:
        for snr in snrs:
            D = est[route][snr] - TRUTH_VEC[None, :]
            D = D[np.all(np.isfinite(D), axis=1)]
            n = D.shape[0]
            if n < p + 2:
                continue
            r_mp = np.sqrt(p / (n - 1))
            C_emp = np.cov(D.T)
            w = np.sort(eigh(C_emp, crb_covs[snr], eigvals_only=True))
            rows.append(dict(
                design=design_key, route=route, snr=snr,
                n=n, trace_mean=float(w.mean()),
                gen_eig_min=float(w[0]), gen_eig_max=float(w[-1]),
                mp_low=float((1 - r_mp) ** 2), mp_high=float((1 + r_mp) ** 2),
            ))
    return rows


# --------------------------------------------------------------------------- #
# 图
# --------------------------------------------------------------------------- #
def fig_spectrum(spectra: dict, dz_spectra: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (key, sv) in zip(axes, spectra.items()):
        dz = dz_spectra[key]
        k = np.arange(1, len(sv) + 1)
        ax.semilogy(k, sv / sv.max(), "o-", label="J (intensity Jacobian)")
        ax.semilogy(k, dz / dz.max(), "s--", label="dZ (differential basis)")
        ax.set_xlabel("spectral rank (descending)")
        ax.set_title(key)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("singular value / max")
    fig.suptitle("Observability spectrum: intensity FIM vs differential-Zernike")
    fig.tight_layout()
    fig.savefig(OUT / "jtj_vs_diffz_spectrum.png", dpi=150)
    plt.close(fig)


def fig_correlation(corrs: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (key, C) in zip(axes, corrs.items()):
        im = ax.imshow(np.abs(C), vmin=0, vmax=1, cmap="magma",
                       interpolation="nearest")
        ax.set_xticks(range(len(INDICES)), [f"Z{j}" for j in INDICES],
                      rotation=90, fontsize=7)
        ax.set_yticks(range(len(INDICES)), [f"Z{j}" for j in INDICES], fontsize=7)
        ax.set_title(f"{key}: |(JtJ)^-1| correlation")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "crb_correlation.png", dpi=150)
    plt.close(fig)


def fig_eta(summary: pd.DataFrame, snrs: list[int]) -> None:
    """每设计一个面板：各路线的效率 vs SNR（逐系数细线 + 中位粗线）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    colors = {"demod": "tab:red", "lm": "tab:blue"}
    for ax, (key, sub) in zip(axes, summary.groupby("design")):
        for route in ROUTES:
            r = sub[sub.route == route]
            piv = r.pivot_table(index="z", columns="snr", values="eta_clean")
            for z, rowv in piv.iterrows():
                ax.plot(snrs, rowv[snrs], ".-", color=colors[route],
                        alpha=0.25, lw=0.8, ms=3)
            med = piv.median()
            ax.plot(snrs, med[snrs], "o-", color=colors[route], lw=2,
                    label=route)
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xscale("linear"); ax.invert_xaxis()
        ax.set_xlabel("SNR (dB)"); ax.set_title(key)
        ax.grid(alpha=0.3); ax.legend()
        ax.set_ylim(-0.05, 1.9)   # 容纳实测最大 eta ~1.76
    axes[0].set_ylabel("efficiency eta = CRB / Var (outliers removed)")
    fig.suptitle("Statistical efficiency vs SNR (thin lines = each Zernike)")
    fig.tight_layout()
    fig.savefig(OUT / "efficiency_vs_snr.png", dpi=150)
    plt.close(fig)


def fig_var_vs_crb(summary: pd.DataFrame, snrs: list[int]) -> None:
    """Z4 / Z9 两代表系数的估计方差 vs SNR，叠加 CRB 线。"""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True)
    colors = {"demod": "tab:red", "lm": "tab:blue"}
    for col, z in ((0, 4), (1, 9)):
        for row, (key, sub) in enumerate(
            summary.groupby("design")
        ):
            ax = axes[row, col]
            s = sub[sub.z == z]
            crb = s[s.route == "lm"].set_index("snr").loc[snrs, "crb_var"]
            ax.loglog(snrs, crb, "k--", lw=1.5, label="CRB")
            for route in ROUTES:
                r = s[s.route == route].set_index("snr").loc[snrs]
                ax.loglog(snrs, r["var_clean"], "o-", color=colors[route],
                          label=f"{route} var")
            ax.set_title(f"{key}  Z{z} ({Z_NAME[z]})")
            ax.grid(alpha=0.3, which="both")
            if row == 1:
                ax.set_xlabel("SNR (dB)")
            if col == 0:
                ax.set_ylabel("Var (waves^2)")
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "variance_vs_crb.png", dpi=150)
    plt.close(fig)


def fig_outlier_hist(all_est, snrs: list[int]) -> None:
    """Δc 直方图：解包裹灾难在低 SNR 产生的重尾/离群结构。

    每设计一面板：解调链各 SNR 的全部 (seed x coeff) 偏差直方图
    （log-y），并以 LM 同 SNR 的偏差作对照。
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    bins = np.linspace(-0.5, 0.5, 201)
    for ax, (key, est) in zip(axes, all_est.items()):
        for snr in snrs:
            d = (est["demod"][snr] - TRUTH_VEC).ravel()
            d = d[np.isfinite(d)]
            ax.hist(np.clip(d, -0.5, 0.5), bins=bins, histtype="step",
                    log=True, label=f"demod {snr}dB")
        lm = np.clip((est["lm"][min(snrs)] - TRUTH_VEC).ravel(), -0.5, 0.5)
        ax.hist(lm[np.isfinite(lm)], bins=bins, histtype="step",
                log=True, ls="--", color="k", label=f"lm {min(snrs)}dB")
        ax.set_xlabel("coefficient error (waves, clipped)")
        ax.set_title(key)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Error histograms: unwrap catastrophes vs LM")
    fig.tight_layout()
    fig.savefig(OUT / "error_histograms.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def load_estimates(npz_path: Path, designs, snrs):
    """--plots-only 时从 npz 重建 est 字典。"""
    d = np.load(npz_path)
    all_est = {}
    for design in designs:
        all_est[design.key] = {
            r: {s: d[f"{design.key}__{r}__{s}"] for s in snrs}
            for r in ROUTES
        }
    return all_est


def paired_compare(all_est, snrs):
    """同一噪声实现下的逐种子对比：每运行 max|Δc|，LM vs 解调链。"""
    rows = []
    for dk, est in all_est.items():
        for s in snrs:
            dm = np.nanmax(np.abs(est["demod"][s] - TRUTH_VEC), axis=1)
            lm = np.nanmax(np.abs(est["lm"][s] - TRUTH_VEC), axis=1)
            ok = np.isfinite(dm) & np.isfinite(lm)
            rows.append(dict(
                design=dk, snr=s, n_paired=int(ok.sum()),
                lm_wins=float(np.mean(lm[ok] < dm[ok])),
                median_ratio=float(np.median(dm[ok] / np.clip(lm[ok], 1e-30, None))),
            ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--snrs", type=int, nargs="*", default=[60, 50, 40, 30, 20])
    ap.add_argument("--plots-only", action="store_true",
                    help="跳过 MC，从 output/mc_estimates.npz 重建统计")
    args = ap.parse_args()
    snrs = sorted(args.snrs, reverse=True)
    OUT.mkdir(exist_ok=True)

    designs = make_designs()
    for d in designs:
        print(f"[design] {d.key}: {d.label}  frames={d.frames.shape}")

    # ---------------------------------------------------------- 1. FIM / CRB
    print("\n===== 1. Fisher 信息 / CRB =====")
    spectra, dz_spectra, corrs = {}, {}, {}
    crb_tables, crb_covs = {}, {}
    spec_rows, corr_rows = [], []
    for d in designs:
        fim = assemble_fim(d)
        cov = np.linalg.inv(fim)
        w, V = np.linalg.eigh(fim)          # 升序
        sv = np.sqrt(np.maximum(w, 0.0))    # J 的奇异值 = sqrt(eigval)
        spectra[d.key] = sv[::-1]
        corrs[d.key] = corr_from_cov(cov)
        print(f"\n{d.key}: cond(J^T J) = {w[-1]/w[0]:.3e}, "
              f"cond(J) = {sv[-1]/sv[0]:.2e}")
        print("  奇异值 σ_i(J)（升序）:", np.array2string(sv, precision=3))
        for i in range(3):  # 三个最弱方向
            v = V[:, i]
            top = np.argsort(-np.abs(v))[:4]
            desc = ", ".join(f"Z{INDICES[t]}:{v[t]:+.2f}" for t in top)
            print(f"    弱方向 {i}: σ={sv[i]:.3e}  [{desc}]")
        for rank in range(len(sv)):
            v = V[:, rank]
            top = int(np.argmax(np.abs(v)))
            spec_rows.append(dict(
                design=d.key, rank_asc=rank, eigval=w[rank],
                singular_value=sv[rank],
                leading_term=f"Z{INDICES[top]}", leading_weight=v[top],
            ))
        # ΔZ 对照
        A_dz = diff_zernike_design(d.fm)
        sdz = np.linalg.svd(A_dz, compute_uv=False)
        dz_spectra[d.key] = sdz
        print(f"  ΔZ 名义设计矩阵: 形状 {A_dz.shape}, "
              f"cond = {sdz[0]/sdz[-1]:.2e}")
        # CRB 表（各 SNR）；同时保存协方差供 Wishart 对照
        crb_tables[d.key] = {
            s: d.sigma(s) ** 2 * np.diag(cov) for s in snrs
        }
        crb_covs[d.key] = {
            s: d.sigma(s) ** 2 * cov for s in snrs
        }
        C = corrs[d.key]
        for i in range(len(INDICES)):
            for j in range(i + 1, len(INDICES)):
                corr_rows.append(dict(design=d.key, zi=int(INDICES[i]),
                                      zj=int(INDICES[j]), corr=C[i, j]))

    pd.DataFrame(spec_rows).to_csv(OUT / "fim_spectrum.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(OUT / "crb_correlation.csv", index=False)

    crb_rows = []
    for d in designs:
        for s in snrs:
            for k, j in enumerate(INDICES):
                crb_rows.append(dict(
                    design=d.key, snr=s, z=int(j),
                    crb_var=crb_tables[d.key][s][k],
                    crb_std=float(np.sqrt(crb_tables[d.key][s][k])),
                ))
    pd.DataFrame(crb_rows).to_csv(OUT / "crb.csv", index=False)

    # 强耦合对
    print("\n  强耦合系数对 |corr| > 0.5:")
    for r in corr_rows:
        if abs(r["corr"]) > 0.5:
            print(f"    {r['design']:12s} Z{r['zi']}-Z{r['zj']}: {r['corr']:+.3f}")

    # ---------------------------------------------------------- 2. Monte Carlo
    if args.plots_only:
        all_est = load_estimates(OUT / "mc_estimates.npz", designs, snrs)
        all_runs = []
        print("\n===== 2. MC 跳过（plots-only，复用 npz） =====")
    else:
        print(f"\n===== 2. Monte Carlo ({args.seeds} seeds × {len(snrs)} SNR) =====")
        all_est, all_runs = {}, []
        for d in designs:
            t0 = time.time()
            est, runs = run_monte_carlo(d, snrs, args.seeds)
            all_est[d.key] = est
            all_runs += runs
            print(f"  {d.key}: {time.time()-t0:.0f}s")
        np.savez(OUT / "mc_estimates.npz",
                 **{f"{dk}__{r}__{s}": all_est[dk][r][s]
                    for dk in all_est for r in ROUTES for s in snrs},
                 indices=INDICES, truth=TRUTH_VEC, snrs=snrs)
        pd.DataFrame(all_runs).to_csv(OUT / "mc_runs.csv", index=False)

    # ---------------------------------------------------------- 3. 汇总
    summary_rows = []
    for d in designs:
        summary_rows += summarize(
            all_est[d.key], crb_tables[d.key], snrs, d.key)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "mc_summary.csv", index=False)

    wish_rows = []
    for d in designs:
        wish_rows += wishart_check(all_est[d.key], crb_covs[d.key], snrs, d.key)
    pd.DataFrame(wish_rows).to_csv(OUT / "wishart_spectrum.csv", index=False)
    print("\n  Wishart 广义谱（C_emp vs C_crb）：")
    print(pd.DataFrame(wish_rows).round(3).to_string(index=False))

    pc = pd.DataFrame(paired_compare(all_est, snrs))
    pc.to_csv(OUT / "paired_compare.csv", index=False)
    print("\n  配对对比（逐种子 max|Δc|，LM 胜率 / 误差比中位数）:")
    print(pc.round(3).to_string(index=False))

    # 控制台摘要：中位效率（截尾/未截尾两个口径并列——3σ_rob 截尾在高斯
    # 误差下方差 ≈0.973σ²，eta_clean 系统性偏高 ~3%）
    print("\n===== 3. 效率汇总（中位数，跨 Z2..Z13）=====")
    print("-- η_clean（剔除 3σ_robust 离群）:")
    piv = (summary.groupby(["design", "route", "snr"])["eta_clean"]
           .median().unstack())
    print(piv.round(3).to_string())
    print("-- η（未截尾）:")
    piv = (summary.groupby(["design", "route", "snr"])["eta"]
           .median().unstack())
    print(piv.round(3).to_string())
    print("\n  灾难率（run 级 max|Δc| > 0.05 wave）:")
    cat = (summary.groupby(["design", "route", "snr"])["catastrophe_rate"]
           .first().unstack())
    print(cat.round(3).to_string())
    print("\n  系数级离群率（|Δc-med|>3σ_robust，均值）:")
    out = summary.groupby(["design", "route", "snr"])["out_rate"].mean().unstack()
    print(out.round(3).to_string())
    if all_runs:
        print("\n  失败率（异常或 NaN）:")
        runs_df = pd.DataFrame(all_runs)
        fail = runs_df.groupby(["design", "route", "snr"])["ok"].apply(
            lambda s: 1 - s.mean()).unstack()
        print(fail.round(3).to_string())
        print("\n  LM 未收敛比例:")
        nc = (runs_df[runs_df.route == "lm"]
              .groupby(["design", "snr"])["converged"]
              .apply(lambda s: 1 - s.mean()).unstack())
        print(nc.round(3).to_string())

    # ---------------------------------------------------------- 4. 图
    fig_spectrum(spectra, dz_spectra)
    fig_correlation(corrs)
    fig_eta(summary, snrs)
    fig_var_vs_crb(summary, snrs)
    fig_outlier_hist(all_est, snrs)

    params = dict(
        indices=INDICES.tolist(), truth_idx=TRUTH_IDX.tolist(),
        truth_c=TRUTH_C.tolist(), snrs=snrs, n_seeds=args.seeds,
        sigma_rule="sigma = max(I_clean) / 10^(SNR/20)",
        designs={d.key: dict(label=d.label, shape=d.fm.shape,
                             s=d.fm.s, f0=d.fm.config.carrier_f0,
                             n_frames=len(d.frames)) for d in designs},
    )
    (OUT / "crb_run_params.json").write_text(json.dumps(params, indent=2))
    print(f"\n产物已写入 {OUT}/")


if __name__ == "__main__":
    main()
