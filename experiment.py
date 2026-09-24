"""棋盘光栅二维剪切干涉：最小可复现实验。

复现刘志祥博士论文《二维剪切干涉检测技术研究》（电子科技大学，2018）
核心计算链条，三条反演路线共享同一个真值波前：

    干涉强度 I(x,y)
        ├─ 相移模式：8 步闭式解调 psi = atan2(-S, C)           （论文 2.3）
        ├─ 傅里叶模式：二维 FFT 提取 +f0 载频瓣 -> arg c(x,y)  （论文 2.4）
        └─ LM 直接反演：min ||I - I(c)||^2                     （不经解调）
                ↓（前两条路线）
    差分波前 dW_x = W(x+s,y)-W(x-s,y)，dW_y 同理
                ↓ 差分 Zernike 最小二乘（式 2-28~2-31）
    波前 Zernike 系数（waves）

运行：  uv run python experiment.py      （图片写入 output/）
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lsi.invert import (  # noqa: E402
    fourier_to_wavefront,
    phase_shift_to_wavefront,
    wavefront_on_grid,
)
from lsi.lm import fit_wavefront_from_carrier_frame, fit_wavefront_from_frames  # noqa: E402
from lsi.model import (  # noqa: E402
    ForwardModel,
    Grid,
    SystemConfig,
    chessboard_orders,
    diffraction_efficiency,
    zernike_wavefront,
)

OUT = Path("output")
OUT.mkdir(exist_ok=True)

INDICES = np.arange(2, 14)          # 拟合 Z2..Z13（Z1 平移不可观）
TRUTH_IDX = np.array([4, 5, 6, 7, 8])
TRUTH_C = np.array([0.31, -0.12, 0.07, 0.42, 0.05])   # waves；Z2/Z3 留零看泄漏
truth = zernike_wavefront(TRUTH_C, TRUTH_IDX)


# --------------------------------------------------------------------------- #
# 报告工具（仅供本脚本使用）
# --------------------------------------------------------------------------- #
def pv(W: np.ndarray, pupil: np.ndarray) -> float:
    """光瞳内峰谷值（waves）。"""
    v = np.asarray(W, dtype=float)[pupil]
    return float(v.max() - v.min())


def rms(W: np.ndarray, pupil: np.ndarray) -> float:
    """光瞳内去平移均方根（waves）。"""
    v = np.asarray(W, dtype=float)[pupil]
    return float(np.sqrt(np.mean((v - v.mean()) ** 2)))


def add_noise(frames: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """加高斯噪声；``snr_db`` 为峰值光强对噪声标准差之比 20 log10(max I / sigma)。"""
    rng = np.random.default_rng(seed)
    sigma = frames.max() / (10.0 ** (snr_db / 20.0))
    return frames + rng.normal(0.0, sigma, size=frames.shape)


def log_spectrum(image: np.ndarray) -> np.ndarray:
    """实图像的对数幅度谱（画图用）。"""
    return np.log10(np.abs(np.fft.fftshift(np.fft.fft2(image))) + 1e-9)


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def show(ax, data, title="", cmap="viridis"):
    arr = np.ma.masked_invalid(np.asarray(data, float))
    im = ax.imshow(arr, origin="lower", cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def report(fit, tag: str) -> None:
    """打印拟合系数与误差汇总（fit 为 ZernikeFit 或 LMResult）。

    误差对全部拟合模式取 max；真值中缺省的模式按 0 处理（零模泄漏）。
    """
    tab = fit.as_dict()
    truth = dict(zip(TRUTH_IDX.tolist(), TRUTH_C.tolist()))
    errors = {j: v - truth.get(j, 0.0) for j, v in tab.items()}
    max_err = max(map(abs, errors.values()), default=0.0)
    leakage = max(
        (abs(e) for j, e in errors.items() if j not in truth), default=0.0
    )
    print(f"  {tag}: " + ", ".join(
        f"Z{j}={tab[j]:+.4f}" for j in TRUTH_IDX))
    print(f"    最大 |系数误差| = {max_err:.2e} wave"
          f"（零模泄漏 {leakage:.2e}）")


# --------------------------------------------------------------------------- #
section("1. 棋盘光栅衍射级次（论文表 2-3）")

eff = diffraction_efficiency(chessboard_orders(max_index=25))
print(f"  0 级         : {eff['dc']*100:.2f} %   (论文 25 %)")
print(f"  每个 ±1 级   : {eff['first_order_total']*100/4:.2f} %   (论文 4.11 %)")
print(f"  全部级次之和 : {sum(eff['all'].values()):.4f}   (Parseval; 解析值 0.5)")
print(f"  二维交叉光栅 : {(0.5/np.pi)**2*100:.2f} %   (论文 2.53 %)")

# --------------------------------------------------------------------------- #
section("2. 路线 A：8 步相移 -> 解调 -> 解包裹 -> 差分 Zernike")

cfg = SystemConfig(grid=Grid(n=128, extent=1.10))
fm = ForwardModel(cfg)
print(f"  {cfg.describe()}")

fx = fm.phase_shift_frames(truth, "x", 8)
fy = fm.phase_shift_frames(truth, "y", 8)
fit_ps, diff = phase_shift_to_wavefront(fm, fx, fy, indices=INDICES)
report(fit_ps, "相移路线")
print(f"    差分残差 rms = {fit_ps.rms_residual:.2e} wave,  条件数 = {fit_ps.cond:.1f}")

x, y = cfg.grid.coords()
pupil = cfg.grid.pupil()
W_true = truth(x, y)
W_ps = wavefront_on_grid(fit_ps.coeffs, fit_ps.indices, x, y, pupil)
print(f"    重构 W: PV = {pv(W_ps, pupil):.4f} wave, RMS = {rms(W_ps, pupil):.4f} wave"
      f"  (真值 PV = {pv(W_true, pupil):.4f}, RMS = {rms(W_true, pupil):.4f})")

# --------------------------------------------------------------------------- #
section("3. 路线 B：单帧载频（傅里叶变换模式）")

cfg_ft = SystemConfig(grid=Grid(n=256, extent=1.10), period_um=30.0)
fm_ft = ForwardModel(cfg_ft)
print(f"  {cfg_ft.describe()}")

frame_ft = fm_ft.carrier_frame(truth)
fit_ft, diff_ft = fourier_to_wavefront(fm_ft, frame_ft, indices=INDICES)
report(fit_ft, "傅里叶路线")
print(f"    差分残差 rms = {fit_ft.rms_residual:.2e} wave")

# --------------------------------------------------------------------------- #
section("4. 路线 C：LM 直接光强反演（不经解调/解包裹）")

frames = np.concatenate([fx, fy], axis=0)
deltas = [fm.phase_shift_deltas(k / 8, 0.0) for k in range(8)]
deltas += [fm.phase_shift_deltas(0.0, k / 8) for k in range(8)]
res_lm = fit_wavefront_from_frames(fm, INDICES, frames, deltas, samples=6000)
report(res_lm, f"LM（16 相移帧, {res_lm.n_iter} 次迭代）")
print(f"    cost = {res_lm.cost:.2e},  rms 残差 = {res_lm.rms_residual:.2e}")

res_lc = fit_wavefront_from_carrier_frame(fm_ft, INDICES, frame_ft, samples=6000)
report(res_lc, f"LM（单帧载频, {res_lc.n_iter} 次迭代）")

# --------------------------------------------------------------------------- #
section("5. 噪声敏感性（相移路线）")

for snr in (60, 40, 20):
    nx = add_noise(fm.phase_shift_frames(truth, "x", 8), snr_db=snr, seed=1)
    ny = add_noise(fm.phase_shift_frames(truth, "y", 8), snr_db=snr, seed=2)
    f, _ = phase_shift_to_wavefront(fm, nx, ny, indices=INDICES)
    tab = f.as_dict()
    truth_map = dict(zip(TRUTH_IDX.tolist(), TRUTH_C.tolist()))
    err = max(abs(v - truth_map.get(j, 0.0)) for j, v in tab.items())
    print(f"  SNR {snr:3d} dB : 最大 |系数误差| = {err:.4f} wave")

# --------------------------------------------------------------------------- #
section("6. 图 -> output/")

fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
show(axes[0, 0], fx[0], "phase-shift frame t=0 (x scan)")
show(axes[0, 1], diff.wrapped_phase["x"], "wrapped demod phase x (rad)")
show(axes[0, 2], np.where(diff.mask["x"], diff.phase["x"], np.nan),
     "unwrapped dW_x (rad)")
show(axes[1, 0], np.where(pupil, W_true, np.nan), "true W (waves)", cmap="jet")
show(axes[1, 1], W_ps, "reconstructed W (waves)", cmap="jet")
show(axes[1, 2], W_ps - W_true, "reconstruction error (waves)")
fig.tight_layout()
fig.savefig(OUT / "phase_shift_route.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
show(axes[0], frame_ft, "carrier frame (Fourier mode)")
show(axes[1], log_spectrum(frame_ft),
     f"log|spectrum|, f0 = {cfg_ft.carrier_f0:.2f} cyc/unit")
xf, yf = cfg_ft.grid.coords()
W_ft = wavefront_on_grid(fit_ft.coeffs, fit_ft.indices, xf, yf, cfg_ft.grid.pupil())
show(axes[2], W_ft - truth(xf, yf), "Fourier route: error (waves)")
fig.tight_layout()
fig.savefig(OUT / "fourier_route.png", dpi=150)

print(f"  wrote {OUT}/phase_shift_route.png, {OUT}/fourier_route.png")
