"""混合反演管线实验：解调初始化 + LM 精修。

把论文解调路线（相移 / 载频瓣）变成第三条路线（LM 直接光强反演）的
初始化器，回答五个问题：

    1. 基线正确性：无噪声下解调输出是否落在 LM 吸引盆内，LM 精修把
       系数误差降到多少。
    2. 吸引盆半径：c0 = c* + r·u（u 为按波前 RMS 归一化的随机方向），
       扫 r 得 LM 收敛到真值的临界半径；对照均匀随机初值。
    3. 噪声增益：SNR = 60/40/20 dB 下比较 (a) 纯解调 (b) 解调+LM
       (c) 零初值 / 随机初值 LM。
    4. 效率：各组 LM 的迭代次数与收敛率。
    5. 结论见 reports/hybrid-init-lm.md。

约定
----
* 扰动/随机方向的 RMS 标度沿用 experiment.py 第 5 节协议：随机方向 d
  生成波前 W(d)，按光瞳内 RMS 归一化为 u，使扰动波前 RMS = r（waves）。
* 成功 = 全部拟合系数 max|Δc| < 1e-6 wave（相移/载频帧破缺 ±W 简并，
  不需允许整体变号）。
* LM 全部使用解析雅可比 + ``samples`` 低差异子采样。

运行：  uv run python analysis/hybrid_init.py            # 全部小节
        uv run python analysis/hybrid_init.py --sections 1 3
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lsi.invert import (  # noqa: E402
    fourier_to_wavefront,
    phase_shift_to_wavefront,
)
from lsi.lm import (  # noqa: E402
    fit_wavefront_from_carrier_frame,
    fit_wavefront_from_frames,
)
from lsi.model import (  # noqa: E402
    ForwardModel,
    Grid,
    SystemConfig,
    zernike_wavefront,
)

OUT = Path("output")
OUT.mkdir(exist_ok=True)

# 与 experiment.py 相同的真值与拟合基底
INDICES = np.arange(2, 14)                      # Z2..Z13
TRUTH_IDX = np.array([4, 5, 6, 7, 8])
TRUTH_C = np.array([0.31, -0.12, 0.07, 0.42, 0.05])
_TRUTH_MAP = dict(zip(TRUTH_IDX.tolist(), TRUTH_C.tolist()))
TRUTH_VEC = np.array([_TRUTH_MAP.get(j, 0.0) for j in INDICES])
TRUTH = zernike_wavefront(TRUTH_C, TRUTH_IDX)

N_STEPS = 8               # 每方向相移步数
LM_SAMPLES = 8192         # LM 子采样像素数（基线 / 噪声实验）
BASIN_SAMPLES = 512       # 吸引盆扫描的 LM 子采样（噪声自由，失拟合跑满迭代）
SUCCESS_TOL = 1e-6        # 收敛到真值的判据（waves, max|Δc|）
FAIL_TOL = 0.1            # 解调链灾难性失败判据（waves, max|Δc|）


# --------------------------------------------------------------------------- #
# 公共工具
# --------------------------------------------------------------------------- #
def make_phase_shift_setup(n: int = 128):
    """相移模式配置（s = 0.0731）：模型、16 帧无噪声数据、相移量。"""
    cfg = SystemConfig(grid=Grid(n=n, extent=1.10))
    fm = ForwardModel(cfg)
    fx = fm.phase_shift_frames(TRUTH, "x", N_STEPS)
    fy = fm.phase_shift_frames(TRUTH, "y", N_STEPS)
    mods = [fm.phase_shift_deltas(k / N_STEPS, 0.0) for k in range(N_STEPS)]
    mods += [fm.phase_shift_deltas(0.0, k / N_STEPS) for k in range(N_STEPS)]
    return cfg, fm, np.concatenate([fx, fy], axis=0), mods


def make_carrier_setup(n: int = 256):
    """傅里叶（载频）模式配置：p = 30 um -> s = 0.0439。"""
    cfg = SystemConfig(grid=Grid(n=n, extent=1.10), period_um=30.0)
    fm = ForwardModel(cfg)
    return cfg, fm, fm.carrier_frame(TRUTH)


def max_err(coeffs: np.ndarray) -> float:
    """拟合系数对真值的最大绝对误差（waves）。"""
    return float(np.max(np.abs(np.asarray(coeffs, dtype=float) - TRUTH_VEC)))


def coeff_errs(coeffs: np.ndarray) -> np.ndarray:
    return np.asarray(coeffs, dtype=float) - TRUTH_VEC


def add_noise(frames: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """experiment.py 的加噪约定：sigma = max(I) / 10^(SNR/20)。"""
    rng = np.random.default_rng(seed)
    sigma = frames.max() / (10.0 ** (snr_db / 20.0))
    return frames + rng.normal(0.0, sigma, size=frames.shape)


def normalized_directions(n_dir: int, rng: np.random.Generator):
    """生成 n_dir 个按光瞳内波前 RMS 归一化的随机方向（experiment.py §5）。

    返回 (n_dir, n_terms)：u = d / rms_pupil(W(d))，即系数向量 u 对应的
    波前在光瞳内 RMS 恰为 1 wave。
    """
    cfg, _, _, _ = make_phase_shift_setup()
    x, y = cfg.grid.coords()
    pupil = cfg.grid.pupil()
    dirs = rng.normal(size=(n_dir, len(INDICES)))
    out = np.empty_like(dirs)
    for i, d in enumerate(dirs):
        W = zernike_wavefront(d, INDICES)(x, y)[pupil]
        out[i] = d / float(np.sqrt(np.mean((W - W.mean()) ** 2)))
    return out


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72, flush=True)


def write_csv(path: Path, header: list[str], rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #
# 第 1 节：基线正确性 —— 解调输出作为 LM 初值
# --------------------------------------------------------------------------- #
def sec1_baseline() -> list[dict]:
    section("1. 基线正确性：解调初始化 -> LM 精修（无噪声）")
    rows = []

    # --- 路线 A：8+8 相移帧 ------------------------------------------------
    cfg, fm, frames, mods = make_phase_shift_setup()
    t0 = time.time()
    fit_ps, _ = phase_shift_to_wavefront(fm, frames[:N_STEPS], frames[N_STEPS:],
                                       indices=INDICES)
    t_demod = time.time() - t0
    e_demod = max_err(fit_ps.coeffs)
    print(f"  相移解调链        : max|Δc| = {e_demod:.3e}  ({t_demod:.2f} s)")

    for tag, x0 in (("zero", None), ("demod", fit_ps.coeffs)):
        t0 = time.time()
        res = fit_wavefront_from_frames(fm, INDICES, frames, mods,
                                       samples=LM_SAMPLES, x0=x0)
        dt = time.time() - t0
        print(f"  LM x0={tag:<5}     : max|Δc| = {max_err(res.coeffs):.3e}  "
              f"n_iter = {res.n_iter:3d}  converged = {res.converged}  "
              f"rms = {res.rms_residual:.2e}  ({dt:.2f} s)")
        rows.append(dict(route="phase_shift", init=tag, init_err=e_demod,
                         final_err=max_err(res.coeffs), n_iter=res.n_iter,
                         converged=res.converged, rms=res.rms_residual,
                         coeffs=res.coeffs))

    # --- 路线 B：单帧载频 --------------------------------------------------
    cfg_ft, fm_ft, frame_ft = make_carrier_setup()
    t0 = time.time()
    fit_ft, _ = fourier_to_wavefront(fm_ft, frame_ft, indices=INDICES)
    t_demod = time.time() - t0
    e_demod = max_err(fit_ft.coeffs)
    print(f"  载频解调链        : max|Δc| = {e_demod:.3e}  ({t_demod:.2f} s)")

    for tag, x0 in (("zero", None), ("demod", fit_ft.coeffs)):
        t0 = time.time()
        res = fit_wavefront_from_carrier_frame(fm_ft, INDICES, frame_ft,
                                              samples=LM_SAMPLES, x0=x0)
        dt = time.time() - t0
        print(f"  LM x0={tag:<5}     : max|Δc| = {max_err(res.coeffs):.3e}  "
              f"n_iter = {res.n_iter:3d}  converged = {res.converged}  "
              f"rms = {res.rms_residual:.2e}  ({dt:.2f} s)")
        rows.append(dict(route="carrier", init=tag, init_err=e_demod,
                         final_err=max_err(res.coeffs), n_iter=res.n_iter,
                         converged=res.converged, rms=res.rms_residual,
                         coeffs=res.coeffs))

    write_csv(
        OUT / "hybrid_baseline.csv",
        ["route", "init", "init_max_err", "final_max_err", "n_iter",
         "converged", "rms_residual"]
        + [f"Z{j}" for j in INDICES],
        [[r["route"], r["init"], f"{r['init_err']:.6e}", f"{r['final_err']:.6e}",
          r["n_iter"], int(r["converged"]), f"{r['rms']:.6e}"]
         + [f"{c:.6e}" for c in r["coeffs"]] for r in rows],
    )
    return rows


# --------------------------------------------------------------------------- #
# 第 2 节：吸引盆半径扫描
# --------------------------------------------------------------------------- #
def sec2_basin(n_dir: int = 12, seed: int = 7) -> None:
    section("2. 吸引盆半径：c0 = c* + r·u vs 均匀随机初值")
    cfg, fm, frames, mods = make_phase_shift_setup()
    cfg_ft, fm_ft, frame_ft = make_carrier_setup()

    # 预探查显示该问题吸引盆边缘约在 1~3 wavefront-RMS（r>2 大量失拟合，
    # 每次失败跑满 _MAX_ITER），r 网格延到 3 以覆盖整个过渡带。
    r_grid = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7,
                       1.0, 1.5, 2.0, 3.0])
    dirs = normalized_directions(n_dir, np.random.default_rng(seed))

    # 真值波前 RMS（随机初值的实际偏移量级参考）
    x, y = cfg.grid.coords()
    pupil = cfg.grid.pupil()
    Wt = TRUTH(x, y)[pupil]
    truth_rms = float(np.sqrt(np.mean((Wt - Wt.mean()) ** 2)))
    print(f"  真值波前 RMS = {truth_rms:.4f} wave；方向数 = {n_dir}，"
          f"r 档 = {list(r_grid)}")

    header = (["setup", "init_kind", "direction", "r_rms", "dc_l2",
               "n_iter", "converged", "rms_residual", "max_err", "success"])
    rows = []

    # 逐行落盘：大批失败拟合（跑满 _MAX_ITER）耗时不可控，中途可查。
    csv_path = OUT / "hybrid_basin.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        def run(setup, init_kind, dir_id, r, c0, fit_fn):
            res = fit_fn(x0=c0)
            e = max_err(res.coeffs)
            row = [setup, init_kind, dir_id, r,
                   f"{np.linalg.norm(np.asarray(c0) - TRUTH_VEC):.6e}",
                   res.n_iter, int(res.converged), f"{res.rms_residual:.6e}",
                   f"{e:.6e}", int(e < SUCCESS_TOL)]
            rows.append(row)
            writer.writerow(row)
            fh.flush()
            return e < SUCCESS_TOL

        for setup, fit_fn in (
            ("phase_shift_16f",
             lambda x0: fit_wavefront_from_frames(fm, INDICES, frames, mods,
                                                 samples=BASIN_SAMPLES, x0=x0)),
            ("carrier_1f",
             lambda x0: fit_wavefront_from_carrier_frame(
                 fm_ft, INDICES, frame_ft, samples=BASIN_SAMPLES, x0=x0)),
        ):
            t0 = time.time()
            for i, u in enumerate(dirs):
                for r in r_grid:
                    run(setup, "perturbed", i, r, TRUTH_VEC + r * u, fit_fn)
                    run(setup, "random", i, r, r * u, fit_fn)
            print(f"  {setup}: {2 * n_dir * len(r_grid)} 次 LM，"
                  f"{time.time() - t0:.1f} s", flush=True)

    print(f"  wrote {csv_path}")
    arr = {k: np.array([row[k] for row in
                        [dict(zip(header, r)) for r in rows]]) for k in header}

    # 每档成功率与临界半径曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, setup in zip(axes, ("phase_shift_16f", "carrier_1f")):
        for kind, marker, label in (
            ("perturbed", "o", "c0 = c* + r*u"),
            ("random", "s", "c0 = r*u (random init)"),
        ):
            sel = (arr["setup"] == setup) & (arr["init_kind"] == kind)
            rate = [np.mean(arr["success"][sel & (arr["r_rms"] == r)].astype(float))
                    for r in r_grid]
            ax.semilogx(r_grid, rate, marker + "-", label=label)
        ax.set_xlabel("initial perturbation RMS r (waves)")
        ax.set_title(setup)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"success rate (max|dC| < {SUCCESS_TOL:g})")
    fig.suptitle(f"LM basin radius ({n_dir} random dirs x {len(r_grid)} r)")
    fig.tight_layout()
    fig.savefig(OUT / "hybrid_basin.png", dpi=150)
    print(f"  wrote {OUT}/hybrid_basin.png")

    for setup in ("phase_shift_16f", "carrier_1f"):
        for kind in ("perturbed", "random"):
            sel = (arr["setup"] == setup) & (arr["init_kind"] == kind)
            rate = [np.mean(arr["success"][sel & (arr["r_rms"] == r)].astype(float))
                    for r in r_grid]
            print(f"  {setup:16s} {kind:9s}: " + "  ".join(
                f"r={r:g}:{v:.2f}" for r, v in zip(r_grid, rate)))


# --------------------------------------------------------------------------- #
# 第 3/4 节：噪声下的精修增益 + 效率
# --------------------------------------------------------------------------- #
def sec3_noise(n_seeds: int = 32, seed: int = 11) -> None:
    section("3. 噪声下的精修增益（相移 16 帧, SNR = 60/40/20 dB）")
    cfg, fm, frames_clean, mods = make_phase_shift_setup()
    dirs = normalized_directions(n_seeds, np.random.default_rng(seed + 1000))

    header = (["snr_db", "seed", "method", "max_err", "success"]
              + [f"err_Z{j}" for j in INDICES]
              + ["n_iter", "converged", "rms_residual", "init_err"])
    rows = []
    t_all = time.time()

    csv_path = OUT / "hybrid_noise.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for snr in (60, 40, 20):
            t0 = time.time()
            for sd in range(n_seeds):
                noisy = add_noise(frames_clean, snr_db=snr,
                                  seed=seed * 1000 + sd)
                fx, fy = noisy[:N_STEPS], noisy[N_STEPS:]

                def lm(x0):
                    return fit_wavefront_from_frames(fm, INDICES, noisy, mods,
                                                    samples=LM_SAMPLES, x0=x0)

                fit_d, _ = phase_shift_to_wavefront(fm, fx, fy,
                                                   indices=INDICES)
                res_h = lm(fit_d.coeffs)
                res_z = lm(None)
                # 随机初值对照：r = 0.3 与 experiment.py §5 最大档一致；
                # r = 1.0 贴近无噪声盆缘，检验噪声下随机初值是否掉出盆地。
                res_r3 = lm(0.3 * dirs[sd])
                res_r10 = lm(1.0 * dirs[sd])

                for method, c, res, ie in (
                    ("demod", fit_d.coeffs, None, np.nan),
                    ("hybrid", res_h.coeffs, res_h, max_err(fit_d.coeffs)),
                    ("lm_zero", res_z.coeffs, res_z,
                     float(np.max(np.abs(TRUTH_VEC)))),
                    ("lm_rand0.3", res_r3.coeffs, res_r3,
                     max_err(0.3 * dirs[sd])),
                    ("lm_rand1.0", res_r10.coeffs, res_r10,
                     max_err(1.0 * dirs[sd])),
                ):
                    e = max_err(c)
                    row = ([snr, sd, method, f"{e:.6e}", int(e < SUCCESS_TOL)]
                           + [f"{v:.6e}" for v in coeff_errs(c)]
                           + [res.n_iter if res else "",
                              int(res.converged) if res else "",
                              f"{res.rms_residual:.6e}" if res else "",
                              f"{ie:.6e}"])
                    rows.append(row)
                    writer.writerow(row)
                fh.flush()
                if sd % 8 == 0:
                    print(f"    SNR={snr} seed={sd} ({time.time() - t0:.0f} s)",
                          flush=True)
            print(f"  SNR={snr}: {n_seeds} seeds done, {time.time() - t0:.0f} s",
                  flush=True)

    print(f"  wrote {csv_path}  total {time.time() - t_all:.0f} s")

    # ---- 汇总表 ----------------------------------------------------------
    data = [dict(zip(header, r)) for r in rows]
    methods = ("demod", "hybrid", "lm_zero", "lm_rand0.3", "lm_rand1.0")
    print("\n  汇总：max|Δc| 中位数 / 最大值 / 成功率(<1e-6) / 灾难失败率(>0.1)")
    for snr in (60, 40, 20):
        for method in methods:
            sel = [d for d in data if d["snr_db"] == snr and d["method"] == method]
            errs = np.array([float(d["max_err"]) for d in sel])
            iters = [d["n_iter"] for d in sel if d["n_iter"] != ""]
            nconv = [d["converged"] for d in sel if d["converged"] != ""]
            print(f"  SNR={snr:3d} {method:8s}: med={np.median(errs):.3e}  "
                  f"max={errs.max():.3e}  ok={np.mean(errs < SUCCESS_TOL):.2f}  "
                  f"fail={np.mean(errs > FAIL_TOL):.2f}"
                  + (f"  iter_med={np.median(iters):.0f} conv={np.mean(nconv):.2f}"
                     if iters else ""))

    # ---- 图 --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for snr_i, snr in enumerate((60, 40, 20)):
        for m_i, m in enumerate(methods):
            errs = np.array([float(d["max_err"]) for d in data
                             if d["snr_db"] == snr and d["method"] == m])
            pos = snr_i * (len(methods) + 1) + m_i
            axes[0].scatter([pos] * len(errs), errs, s=9, alpha=0.5,
                            label=m if snr_i == 0 else None)
            axes[0].plot([pos - 0.25, pos + 0.25],
                         [np.median(errs)] * 2, "k-", lw=1.5)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("max|Δc| (waves)")
    axes[0].axhline(SUCCESS_TOL, color="g", ls="--", lw=0.8)
    axes[0].axhline(FAIL_TOL, color="r", ls="--", lw=0.8)
    ticks, labels = [], []
    for snr_i, snr in enumerate((60, 40, 20)):
        for m_i, m in enumerate(methods):
            ticks.append(snr_i * (len(methods) + 1) + m_i)
            labels.append(f"{m}\n{snr}dB")
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_title("max|dC| under noise (tick = median)")
    axes[0].grid(alpha=0.3)

    for m_i, m in enumerate(("hybrid", "lm_zero", "lm_rand0.3", "lm_rand1.0")):
        iters = np.array([int(d["n_iter"]) for d in data
                          if d["method"] == m])
        axes[1].hist(iters, bins=np.arange(0, iters.max() + 2) - 0.5,
                     alpha=0.5, label=m, density=True)
    axes[1].set_xlabel("LM n_iter")
    axes[1].set_title("LM iteration count (all SNR pooled)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "hybrid_noise.png", dpi=150)
    print(f"  wrote {OUT}/hybrid_noise.png")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--n-dir", type=int, default=12)
    ap.add_argument("--n-seeds", type=int, default=32)
    args = ap.parse_args()

    if 1 in args.sections:
        sec1_baseline()
    if 2 in args.sections:
        sec2_basin(n_dir=args.n_dir)
    if 3 in args.sections:
        sec3_noise(n_seeds=args.n_seeds)


if __name__ == "__main__":
    main()
