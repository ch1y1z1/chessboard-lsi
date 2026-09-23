"""01 - 二维棋盘剪切干涉仪的前向模型。

逐步复现论文第二章：

* 45° 旋转振幅棋盘光栅的衍射级次
  （表 2-3：0 级 25%，每个 ±1 级 4.11%，交叉光栅 2.53%），
* 2.3.1 的相移律（沿 +x 方向 8 帧一个周期），
* 叠加前向模型 I = |sum A exp(i[2 pi W + delta])|^2
  与显式 4/5 光束区域公式 (2-12)...(2-16) 的逐点对照。

运行：  python3 scripts/01_forward_model.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lsi.config import Grid, SystemConfig, PRESET_FOURIER, PRESET_PHASE_SHIFT
from lsi.forward import ForwardModel, ZernikeWavefront, paper_region_intensity
from lsi.ftmode import spectrum
from lsi.grating import chessboard_orders, diffraction_efficiency
from lsi.metrics import pv, rms
from lsi.plotting import imshow, new_fig, savefig
from lsi.zernike import zernike

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT, exist_ok=True)
REPORT: dict = {}


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --------------------------------------------------------------------------- #
section("1. 棋盘光栅的衍射级次（表 2-3）")

ideal = chessboard_orders(max_index=25)
eff = diffraction_efficiency(ideal)
first = eff["first_order_total"] * 100 / 4
crossed = (0.5 / np.pi) ** 2 * 100
print(f"  0 级             : {eff['dc']*100:.2f} %   (论文 25 %)")
print(f"  每个 ±1 级       : {first:.2f} %   (论文 4.11 %)")
print(f"  全部级次之和     : {sum(eff['all'].values()):.4f}   (Parseval; 解析值 0.5)")
print(f"  二维交叉光栅     : {crossed:.2f} %   (论文 2.53 %)")
print("  闭式 |A| = 2/(pi^2 |a^2-b^2|):")
for a, b in [(1, 0), (0, 1), (2, 1), (-2, -1)]:
    print(f"    (a,b)=({a:+d},{b:+d})  |A| = {abs(ideal[(float(a), float(b))]):.6f}")
REPORT["grating"] = {
    "dc_pct": eff["dc"] * 100,
    "first_order_each_pct": first,
    "crossed_pct": crossed,
    "parseval": float(sum(eff["all"].values())),
}

# --------------------------------------------------------------------------- #
section("2. 叠加模型 vs 论文区域公式 (2-12) ... (2-16)")

cfg = SystemConfig(grid=Grid(n=160, extent=1.10))
s = cfg.s
A0, A1 = 0.5, 2.0 / np.pi**2


def order_dict(orders) -> dict:
    return {tuple(map(float, o)): (A0 if tuple(o) == (0, 0) else A1) for o in orders}


class SmoothWavefront:
    """不可分离的测试波前（彗差 + 离焦 + 像散 + 局部凸起）。"""

    def w(self, x, y):
        return (
            0.45 * zernike(7, x, y)
            - 0.20 * zernike(4, x, y)
            + 0.15 * zernike(11, x, y)
            + 0.10 * zernike(5, x, y)
            + 0.12 * np.exp(-3.0 * (x**2 + y**2)) * np.cos(3.0 * y)
        )


wf = SmoothWavefront()
x, y = cfg.grid.coords()
regions = {
    "x1 (光束 0,±1_x,+1_y)": ([(0, 0), (1, 0), (-1, 0), (0, 1)], "x1"),
    "x2 (光束 0,±1_x,-1_y)": ([(0, 0), (1, 0), (-1, 0), (0, -1)], "x2"),
    "x5 (全部五光束)": ([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)], "x5"),
    "y1 (光束 0,±1_y,+1_x)": ([(0, 0), (0, 1), (0, -1), (1, 0)], "y1"),
}
for label, (orders, tag) in regions.items():
    fm = ForwardModel(cfg, order_dict(orders))
    for t in (0.0, 0.125):
        # x 区域对 x 扫描，y 区域对 y 扫描
        deltas = fm.phase_shift_deltas(t, 0.0) if tag.startswith("x") else fm.phase_shift_deltas(0.0, t)
        I_model = fm.intensity(wf, deltas=deltas)
        mask = np.ones(cfg.grid.shape, dtype=bool)
        for a, b in orders:
            mask &= (x + a * s) ** 2 + (y + b * s) ** 2 <= 1.0
        I_paper = paper_region_intensity(tag, wf.w, x, y, s, 2 * np.pi * t, A0=A0, A1=A1)
        err = np.abs(I_model[mask] - I_paper[mask]).max() / np.abs(I_paper[mask]).max()
        print(f"  {label:26s} t={t:5.3f}  最大相对差 = {err:.2e}")
        REPORT.setdefault("region_agreement", []).append(
            {"region": label, "t": t, "rel_err": float(err)}
        )

# --------------------------------------------------------------------------- #
section("3. 2.3.1 的相移律（沿 +x 方向 8 帧一个周期）")

cfg_ps = PRESET_PHASE_SHIFT
fm_ps = ForwardModel(cfg_ps)
truth = ZernikeWavefront(np.array([1.0]), np.array([7]))
frames = fm_ps.phase_shift_frames(truth, "x", 8)
print("  8 步 x 扫描：每帧的 p2v 与 (+1,0) 级的 delta")
for k, frame in enumerate(frames):
    d = fm_ps.phase_shift_deltas(k / 8, 0.0)[fm_ps.order_list.index((1.0, 0.0))]
    print(f"    第 {k} 帧: p2v = {np.ptp(frame):.5f}   delta(+1,0) = {d:+.4f} rad")
print("  第 8 帧将回到同一光栅位置（相移律的周期）。")
REPORT["phase_shift_law"] = {"period_frames": 8}

# --------------------------------------------------------------------------- #
section("4. W = Z7（1 波长）的前向输出：图样与频谱")

cfg_ft = PRESET_FOURIER
fm_ft = ForwardModel(cfg_ft)
frame_ft = fm_ft.carrier_frame(truth)
frames4 = fm_ps.phase_shift_frames(truth, "x", 4)

x_ps, y_ps = cfg_ps.grid.coords()
pupil = cfg_ps.grid.pupil()
W_true = truth.w(x_ps, y_ps)
print(f"  相移配置 : {cfg_ps.describe()}")
print(f"  傅里叶配置: {cfg_ft.describe()}")
print(f"  输入 W   : PV = {pv(W_true, pupil):.4f} wave, RMS = {rms(W_true, pupil):.4f} wave")
print("  (Z7=1 理论值: PV = 2.000, RMS = 0.35355；采样值随网格分辨率变化)")

fig, axes = new_fig(2, 4, figsize=(16, 7.5))
for k in range(4):
    imshow(axes[0, k], frames4[k], cfg_ps.grid, title=f"phase-shift frame t={k}/4")
imshow(axes[1, 0], frame_ft, cfg_ft.grid, title="Fourier mode: single frame")
imshow(axes[1, 1], np.log10(np.abs(spectrum(frame_ft)) + 1e-9), cfg_ft.grid,
       title=f"log|spectrum|, f0 = {cfg_ft.carrier_f0:.2f} cyc/unit")
imshow(axes[1, 2], 2 * np.pi * (truth.w(x_ps + cfg_ps.s, y_ps) - truth.w(x_ps - cfg_ps.s, y_ps)),
       cfg_ps.grid, title="true 2-sided dW_x (rad)")
imshow(axes[1, 3], W_true, cfg_ps.grid, title="input W (waves)")
savefig(fig, "01_forward.png")
print(f"\n  -> wrote {OUT}/01_forward.png and 01_forward_model.json")

with open(os.path.join(OUT, "01_forward_model.json"), "w") as fh:
    json.dump(REPORT, fh, indent=2)
