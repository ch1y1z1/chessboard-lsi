"""可观测性曲线：堆叠雅可比 J 的 sigma_min / cond(J^T J) 随系统参数与拟合阶数。

J 在真值 c* 处取值（LM 的收敛点邻域）；帧设计取信息量最大的 8+8 相移
（t=i/8 两方向），并对载频单帧（f0=1/2s，随配置自适应）做对照。

    (a) 剪切量扫描：period_um x na 网格 -> s = sqrt(2) lam / (2 NA p)
    (b) 拟合阶数扫描：Z2..Z_N，N = 13..36

退化模式判定：最小 3 个右奇异向量的主导分量（|v| 加权）。

输出：output/observability_{shear,order}.csv，output/observability.png

运行：  uv run python analysis/observability.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lsi.lm import _frame_and_jacobian  # noqa: E402
from lsi.model import FRINGE_MODES, ForwardModel, Grid, SystemConfig  # noqa: E402

from common import (  # noqa: E402
    GRID,
    SAMPLES,
    TRUTH_C,
    TRUTH_IDX,
    sample_rows,
)

OUT = Path("output")
OUT.mkdir(exist_ok=True)


def truth_vec(indices):
    m = dict(zip(TRUTH_IDX.tolist(), TRUTH_C.tolist()))
    return np.array([m.get(int(j), 0.0) for j in indices])


def shift_mods(fm):
    """8+8 相移帧的调制序列（不含载频）。"""
    return (
        [fm.phase_shift_deltas(i / 8, 0.0) for i in range(8)]
        + [fm.phase_shift_deltas(0.0, i / 8) for i in range(8)]
    )


def stacked_jacobian(fm, indices, coeffs, mods, samples):
    """堆叠各帧解析雅可比 J（在采样像素上），返回 (J, n_frames)。"""
    rows = sample_rows(fm, samples)
    cache = fm.zernike_samples(indices, rows)
    parts = []
    for mod in mods:
        m = np.asarray(mod, dtype=float)
        if m.ndim == 3:
            m = m.reshape(len(cache), -1)[:, rows]
        parts.append(_frame_and_jacobian(cache, coeffs, m)[1])
    return np.vstack(parts), len(parts)


def svd_report(J, indices):
    """奇异值谱 + 最小 3 个右奇异向量的主导模式。"""
    _, sv, Vt = np.linalg.svd(J, full_matrices=False)
    cond_J = sv[0] / sv[-1]
    bottom = []
    for k in range(3):
        v = np.abs(Vt[-1 - k])
        order = np.argsort(v)[::-1][:3]
        bottom.append(
            [(int(indices[i]), float(v[i])) for i in order]
        )
    return sv, cond_J, bottom


def fmt_bottom(bottom):
    """[(Z5,0.62),(Z8,0.44)] -> 'Z5:0.62+Z8:0.44'"""
    return "; ".join(
        "+".join(f"Z{j}:{w:.2f}" for j, w in comp) for comp in bottom
    )


def main() -> None:
    # ------------------------------------------------------------ (a) 剪切量
    print("== (a) 剪切量扫描：period x na，拟合 Z2..Z13 ==")
    indices = np.arange(2, 14)
    c_star = truth_vec(indices)
    shear_rows = []
    print(f"  {'p(um)':>6s} {'NA':>5s} {'s':>7s} {'design':>9s} "
          f"{'sig_min':>9s} {'sig_max':>9s} {'cond(J)':>10s} {'cond(JtJ)':>12s}")
    for period in (9.0, 18.0, 30.0, 60.0):
        for na in (0.25, 0.34, 0.6):
            cfg = SystemConfig(grid=GRID, period_um=period, na=na)
            fm = ForwardModel(cfg)
            for tag, mods in (
                ("ps_8+8", shift_mods(fm)),
                ("carrier", [fm.carrier_phases(cfg.carrier_f0)]),
            ):
                J, nf = stacked_jacobian(fm, indices, c_star, mods, SAMPLES)
                sv, cond_J, bottom = svd_report(J, indices)
                shear_rows.append(dict(
                    period_um=period, na=na, s=cfg.s, f0=cfg.carrier_f0,
                    design=tag, n_frames=nf, n_rows=J.shape[0],
                    sigma_min=sv[-1], sigma_max=sv[0],
                    cond_J=cond_J, cond_JtJ=cond_J**2,
                    bottom_modes=fmt_bottom(bottom),
                ))
                print(f"  {period:6.0f} {na:5.2f} {cfg.s:7.4f} {tag:>9s} "
                      f"{sv[-1]:9.3e} {sv[0]:9.3e} {cond_J:10.3e} "
                      f"{cond_J**2:12.3e}", flush=True)

    with open(OUT / "observability_shear.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(shear_rows[0]))
        w.writeheader()
        w.writerows(shear_rows)

    # ------------------------------------------------------------ (b) 拟合阶数
    print("\n== (b) 拟合阶数扫描：Z2..Z_N @ p=18um NA=0.34 ==")
    cfg = SystemConfig(grid=GRID)
    fm = ForwardModel(cfg)
    mods = shift_mods(fm)
    order_rows = []
    spectra = {}
    print(f"  {'N':>3s} {'n_terms':>7s} {'sig_min':>9s} {'cond(J)':>10s} "
          f"{'cond(JtJ)':>12s}  退化模式")
    for N in (13, 16, 21, 26, 31, 36):
        indices = np.arange(2, N + 1)
        J, nf = stacked_jacobian(fm, indices, truth_vec(indices), mods,
                                 SAMPLES)
        sv, cond_J, bottom = svd_report(J, indices)
        spectra[N] = sv
        order_rows.append(dict(
            max_index=N, n_terms=len(indices), n_rows=J.shape[0],
            sigma_min=sv[-1], sigma_max=sv[0],
            cond_J=cond_J, cond_JtJ=cond_J**2,
            bottom_modes=fmt_bottom(bottom),
        ))
        print(f"  {N:3d} {len(indices):7d} {sv[-1]:9.3e} {cond_J:10.3e} "
              f"{cond_J**2:12.3e}  {fmt_bottom(bottom)}", flush=True)

    with open(OUT / "observability_order.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(order_rows[0]))
        w.writeheader()
        w.writerows(order_rows)
    with open(OUT / "observability_spectrum.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["max_index", "k", "sigma"])
        for N, sv in spectra.items():
            for k, s_ in enumerate(sv):
                w.writerow([N, k, f"{s_:.8e}"])

    # ------------------------------------------------------------ PNG
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for tag, marker in (("ps_8+8", "o"), ("carrier", "s")):
        for na in (0.25, 0.34, 0.6):
            sel = [r for r in shear_rows if r["na"] == na
                   and r["design"] == tag]
            sel.sort(key=lambda r: r["s"])
            ax.semilogy([r["s"] for r in sel], [r["cond_JtJ"] for r in sel],
                        marker=marker, ms=4,
                        label=f"{tag} NA={na}")
    ax.set_xlabel("shear s (fraction of pupil radius)")
    ax.set_ylabel("cond(J^T J) = (sig_max/sig_min)^2")
    ax.set_title("observability vs shear (fit Z2..Z13)")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.semilogy([r["max_index"] for r in order_rows],
                [r["cond_JtJ"] for r in order_rows], "o-")
    for r in order_rows:
        ax.annotate(r["bottom_modes"].split(";")[0],
                    (r["max_index"], r["cond_JtJ"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=6)
    ax.set_xlabel("max Fringe index fitted (Z2..Z_N)")
    ax.set_ylabel("cond(J^T J)")
    ax.set_title("observability vs model order (8+8 phase-shift frames)")
    fig.tight_layout()
    fig.savefig(OUT / "observability.png", dpi=150)
    print(f"\nwrote {OUT}/observability_shear.csv, observability_order.csv, "
          f"observability_spectrum.csv, observability.png")


if __name__ == "__main__":
    main()
