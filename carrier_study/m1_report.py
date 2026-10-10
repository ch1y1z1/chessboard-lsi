"""把 results/carrier/m1/predictions.json 渲染成备忘用的表格（打印到 stdout）。"""

from __future__ import annotations

import json
from pathlib import Path

from .analysis import fmt, md_table
from .common import NOISES

PRED = Path("results/carrier/m1/predictions.json")


def main() -> None:
    P = json.loads(PRED.read_text())
    for p in (35, 12):
        d = P[f"p{p}"]
        print(f"\n## p = {p}\n")
        print(f"κ(JᵀJ) 中位 {fmt(d['kappa_JtJ_median'])}，范围 {[fmt(x) for x in d['kappa_JtJ_range']]}；"
              f"GN 二次收敛常数 C 中位 {fmt(d['C_gn_median'])} /wave；γ_N 中位 {fmt(d['gamma_N_max_median'])}")
        print(f"每步成本（前向当量）：{ {k: round(v, 2) for k, v in d['unit_cost_per_iter'].items()} }")
        print(f"梯度尺度 g_x = {fmt(d['gradient_scale_g_x'])}；κ 随周期：{d.get('kappa_by_period')}")
        print(f"r50 预测（c = 0.15/0.25/0.35）：{d['basin_r50_pred']}\n")
        rows = []
        for nz in NOISES:
            v = d["noise"][nz.label]
            c = v["cost"]
            rows.append([nz.label, fmt(v["crb_wf_median"]), fmt(v["pred_d2_fourier_median"]),
                         fmt(v["pred_d2_unweighted_ls_median"]), fmt(v["curv_rel_max"]),
                         fmt(v.get("fourier_phase_breakdown_fraction", float("nan"))),
                         f"{fmt(c['t0.1']['gn'])} / {fmt(c['t0.1']['lbfgs'])} / {fmt(c['t0.1']['accel'])} / {fmt(c['t0.1']['gd'])}",
                         f"{fmt(c['t0.1']['n_gn'])} / {fmt(c['t0.1']['n_accel'])} / {fmt(c['t0.1']['n_gd'])}"])
        print(md_table(["噪声", "CRB 波前 RMS", "傅里叶 E[d²] 预测", "不加权 LS E[d²]", "γ_N·σ·√p 最大",
                        "傅里叶相位噪声失效比例", "τ=0.1 成本 GN/L-BFGS/加速/GD", "迭代数 GN/加速/GD"], rows))
        print("\n傅里叶 E[d²] 按 RMS：")
        rms_keys = list(d["noise"]["G40"]["pred_d2_fourier_by_rms"])
        print(md_table(["噪声"] + rms_keys,
                       [[nz.label] + [fmt(d["noise"][nz.label]["pred_d2_fourier_by_rms"][k]) for k in rms_keys] for nz in NOISES]))
        print("\n一阶方法更便宜的噪声档：", d["first_order_cheaper_at"])
        print("\n定义域（按 RMS）：")
        print(md_table(["RMS", "瓣局部频率最大（中位）", "拍频/奈奎斯特（最大）", "瓣频率 > 4 的比例"],
                       [[k, fmt(v["lobe_freq_max_median"]), fmt(v["beat_over_nyquist_max"]), fmt(v["frac_lobe_over_window"])]
                        for k, v in d["domain"].items()]))
        print("\n失配一阶预测（中位，waves）：")
        print(md_table(["失配", "MLE 偏差", "傅里叶偏差", "MLE（去倾斜）", "傅里叶（去倾斜）", "傅里叶更小的比例"],
                       [[k, fmt(v["mle_bias_wf_median"]), fmt(v["fourier_bias_wf_median"]),
                         fmt(v["mle_bias_wf_notilt_median"]), fmt(v["fourier_bias_wf_notilt_median"]),
                         fmt(v["frac_fourier_better"])] for k, v in d["mismatch"].items()]))


if __name__ == "__main__":
    main()
