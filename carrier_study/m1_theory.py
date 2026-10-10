"""阶段 1：理论预测（只用 dev 真值），主实验前冻结。

运行：  uv run python -m carrier_study.m1_theory
输出：  results/carrier/m1/theory_truths.csv.gz   每个 dev 真值的理论量
        results/carrier/m1/mismatch_linear.csv.gz 失配偏差的一阶预测（P3）
        results/carrier/m1/predictions.json       汇总后的 P1–P7 预测数值
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import config
from .calibrate import load_units
from .common import NOISES, mismatch_scenes, pmap, write_csv, write_json
from .fourier import FourierChain
from .generator import Generator, Noise
from .nominal import NominalModel
from .theory import (
    curvature,
    domain_stats,
    gradient_scale,
    iters_accel,
    iters_gd,
    iters_gn,
)
from .truths import tail_q, truth_set

OUT = Path("results/carrier/m1")
TAUS = (1.0, 0.3, 0.1)
MLE_CURV_CRIT = 0.3          # 预测 MLE 非线性失效：gamma_N * sigma * sqrt(p) >= 0.3
FOURIER_PHASE_CRIT = np.pi / 4  # 预测傅里叶链失效：掩膜内 99 分位相位噪声 std >= pi/4
BASIN_C = (0.15, 0.25, 0.35)    # 吸引盆：差分波前误差 RMS 达到 c（waves）


def _sigma(img_max: float, snr_db: float) -> float:
    return img_max / 10.0 ** (snr_db / 20.0)


def _poisson_var(I: np.ndarray, nz: Noise) -> np.ndarray:
    scale = nz.level / I.max()
    return (I * scale + nz.read_e**2) / scale**2


def truth_worker(args) -> dict:
    p, tr = args
    idx = config.indices_for(p)
    cfg = config.system()
    model = NominalModel(cfg, idx)
    q = tr.q
    row = {"tid": tr.tid, "p": p, "shape": tr.shape, "rms": tr.rms, "rep": tr.rep}
    row.update(domain_stats(cfg, idx, q))
    row.update(curvature(model, q, n_dirs=150, seed=1))
    img = Generator(cfg, idx).clean(q)
    I_pix = model.take(img)
    _, st = model.forward(q)
    J = model.jacobian(st)
    JtJ = J.T @ J
    crb1 = np.linalg.inv(JtJ)
    row["crb_wf_rms_per_sigma"] = float(np.sqrt(np.trace(crb1)))
    row["img_max"] = float(img.max())

    chain = FourierChain(cfg, idx)
    lin = chain.linearize(img)
    b = lin.q0 - q
    row["fourier_bias_wf"] = float(np.linalg.norm(b))
    L = lin.matrix(batch=64)
    LLt = L @ L.T
    row["fourier_noise_wf_per_sigma"] = float(np.sqrt(np.trace(LLt)))
    row["fourier_var_ratio_gauss"] = float(np.trace(JtJ @ LLt) / p)
    # 每像素相位噪声（单位 sigma）：std(Im(dc/c0)) = sqrt(mean|win|^2 / 2) / |c0|
    fac = np.sqrt(np.mean(chain.win_u**2) / 2.0)
    ph = np.concatenate([fac / np.abs(lin.parts[d]["c"][lin.parts[d]["mask"]]) for d in ("x", "y")])
    row["phase_std_p99_per_sigma"] = float(np.percentile(ph, 99))
    row["mask_px"] = int(sum(lin.n_rows.values()))

    for nz in NOISES:
        if nz.kind == "gauss":
            sig = _sigma(img.max(), nz.level)
            fimm = JtJ / sig**2
            d2_four = float(b @ fimm @ b / p) + row["fourier_var_ratio_gauss"]
            d2_ls = 1.0
            crb_wf = sig * row["crb_wf_rms_per_sigma"]
            curv_rel = row["gamma_N_max"] * sig * np.sqrt(p)
            four_noise_wf = sig * row["fourier_noise_wf_per_sigma"]
        else:
            var_full = _poisson_var(img, nz).ravel()
            var_pix = var_full[model.pix]
            fimm = (J / var_pix[:, None]).T @ J
            cov_four = (L * var_full[None, :]) @ L.T
            d2_four = float((b @ fimm @ b + np.trace(fimm @ cov_four)) / p)
            cov_ls = crb1 @ (J.T * var_pix[None, :]) @ J @ crb1
            d2_ls = float(np.trace(fimm @ cov_ls) / p)
            crb_wf = float(np.sqrt(np.trace(np.linalg.inv(fimm))))
            curv_rel = row["gamma_N_max"] * float(np.sqrt(var_pix.mean())) * np.sqrt(p)
            four_noise_wf = float(np.sqrt(np.trace(cov_four)))
        lab = nz.label
        row[f"crb_wf_{lab}"] = crb_wf
        row[f"pred_d2_fourier_{lab}"] = d2_four
        row[f"pred_d2_unweighted_ls_{lab}"] = d2_ls
        row[f"curv_rel_{lab}"] = float(curv_rel)
        # 成本预测（傅里叶初值 -> 各精度目标）
        e0 = float(np.sqrt(b @ b + four_noise_wf**2))
        for tau in TAUS:
            et = tau * crb_wf
            ratio = max(e0 / et, 1.0)
            row[f"pred_n_gn_{lab}_t{tau:g}"] = iters_gn(e0, et, row["C_gn_median"])
            row[f"pred_n_accel_{lab}_t{tau:g}"] = iters_accel(row["kappa_JtJ"], ratio)
            row[f"pred_n_gd_{lab}_t{tau:g}"] = iters_gd(row["kappa_JtJ"], ratio)
    return row


def mismatch_worker(args) -> list[dict]:
    """P3：失配偏差的一阶预测，MLE 与傅里叶链。"""
    p, tr = args
    idx = config.indices_for(p)
    cfg = config.system()
    model = NominalModel(cfg, idx)
    q = tr.q
    img0 = Generator(cfg, idx).clean(q)
    I0 = model.take(img0)
    _, st = model.forward(q)
    J = model.jacobian(st)
    proj = np.linalg.solve(J.T @ J, J.T)
    chain = FourierChain(cfg, idx)
    lin = chain.linearize(img0)
    b0 = lin.q0 - q
    rows = []
    for name, level, scene in mismatch_scenes(seed=config.SEED_MISMATCH):
        tq = None
        if name == "tail_frac":
            tq = tail_q(level * np.linalg.norm(q), config.rng(config.SEED_MISMATCH, 1, tr.rep))
        img = Generator(cfg, idx, scene).clean(q, tq)
        dI = img - img0
        db_mle = proj @ model.take(dI)
        db_four = b0 + lin.apply(dI)[0]
        rows.append({
            "tid": tr.tid, "p": p, "shape": tr.shape, "rms": tr.rms, "factor": name, "level": level,
            "pred_bias_wf_mle": float(np.linalg.norm(db_mle)),
            "pred_bias_wf_mle_notilt": float(np.linalg.norm(db_mle[2:])),
            "pred_bias_wf_fourier": float(np.linalg.norm(db_four)),
            "pred_bias_wf_fourier_notilt": float(np.linalg.norm(db_four[2:])),
            "pred_bias_mle": db_mle, "pred_bias_fourier": db_four,
        })
    return rows


def shear_worker(args) -> dict:
    period, p, tr = args
    idx = config.indices_for(p)
    cfg = config.system(period)
    model = NominalModel(cfg, idx)
    c = curvature(model, tr.q, n_dirs=60, seed=2)
    return {"period_um": period, "s": cfg.s, "p": p, "tid": tr.tid, "kappa_JtJ": c["kappa_JtJ"],
            "C_gn_median": c["C_gn_median"]}


def summarize(rows: list[dict], mis_rows: list[dict], shear_rows: list[dict]) -> dict:
    pred: dict = {"criteria": {
        "mle_curvature": MLE_CURV_CRIT, "fourier_phase_noise_rad": FOURIER_PHASE_CRIT,
        "basin_c_waves": BASIN_C, "taus": TAUS}}
    for p in (35, 12):
        R = [r for r in rows if r["p"] == p]
        units = load_units(p)
        c_gn = units["F"] + units["J"] + units["NE"]
        c_acc = 2 * units["F"] + units["G"]        # 加速梯度：y 处 F+G，回溯检查一次 F
        c_lbfgs = units["F"] + units["G"]
        c_gd = 2 * units["F"] + units["G"]
        P = {"n_truths": len(R),
             "kappa_JtJ_median": float(np.median([r["kappa_JtJ"] for r in R])),
             "kappa_JtJ_range": [float(min(r["kappa_JtJ"] for r in R)), float(max(r["kappa_JtJ"] for r in R))],
             "C_gn_median": float(np.median([r["C_gn_median"] for r in R])),
             "gamma_N_max_median": float(np.median([r["gamma_N_max"] for r in R])),
             "unit_cost_per_iter": {"gn": c_gn, "accel": c_acc, "lbfgs": c_lbfgs, "gd": c_gd},
             "noise": {}}
        for nz in NOISES:
            lab = nz.label
            d = {
                "crb_wf_median": float(np.median([r[f"crb_wf_{lab}"] for r in R])),
                "pred_d2_mle": 1.0,
                "pred_d2_fourier_median": float(np.median([r[f"pred_d2_fourier_{lab}"] for r in R])),
                "pred_d2_fourier_by_rms": {
                    f"{rms:g}": float(np.median([r[f"pred_d2_fourier_{lab}"] for r in R if r["rms"] == rms]))
                    for rms in sorted({r["rms"] for r in R})},
                "pred_d2_unweighted_ls_median": float(np.median([r[f"pred_d2_unweighted_ls_{lab}"] for r in R])),
                "curv_rel_max": float(np.max([r[f"curv_rel_{lab}"] for r in R])),
                "mle_nonlinear_predicted": bool(np.max([r[f"curv_rel_{lab}"] for r in R]) >= MLE_CURV_CRIT),
                "cost": {},
            }
            if nz.kind == "gauss":
                sig_rel = 10.0 ** (-nz.level / 20.0)
                frac = np.mean([r["phase_std_p99_per_sigma"] * sig_rel * r["img_max"] >= FOURIER_PHASE_CRIT for r in R])
                d["fourier_phase_breakdown_fraction"] = float(frac)
            for tau in TAUS:
                k = f"{lab}_t{tau:g}"
                n_gn = np.median([r[f"pred_n_gn_{k}"] for r in R])
                n_acc = np.median([r[f"pred_n_accel_{k}"] for r in R])
                n_gd = np.median([r[f"pred_n_gd_{k}"] for r in R])
                d["cost"][f"t{tau:g}"] = {
                    "gn": float(1 + n_gn * c_gn), "accel": float(1 + n_acc * c_acc),
                    "lbfgs": float(1 + 0.5 * n_acc * c_lbfgs), "gd": float(1 + n_gd * c_gd),
                    "n_gn": float(n_gn), "n_accel": float(n_acc), "n_gd": float(n_gd),
                }
            P["noise"][lab] = d
        # P5：交叉点
        cross = {}
        for tau in TAUS:
            first_cheaper = [lab for lab, d in P["noise"].items()
                             if min(d["cost"][f"t{tau:g}"][m] for m in ("accel", "lbfgs", "gd")) < d["cost"][f"t{tau:g}"]["gn"]]
            cross[f"t{tau:g}"] = first_cheaper
        P["first_order_cheaper_at"] = cross
        # P7：傅里叶链失效（定义域）
        P["domain"] = {
            f"{rms:g}": {
                "lobe_freq_max_median": float(np.median([r["lobe_freq_max"] for r in R if r["rms"] == rms])),
                "beat_over_nyquist_max": float(np.max([r["beat_over_nyquist"] for r in R if r["rms"] == rms])),
                "frac_lobe_over_window": float(np.mean([r["lobe_freq_max"] > 4.0 for r in R if r["rms"] == rms])),
            } for rms in sorted({r["rms"] for r in R})}
        # P6：吸引盆
        gs = gradient_scale(config.system(), config.indices_for(p))
        P["gradient_scale_g_x"] = gs["g_x"]
        P["basin_r50_pred"] = {
            f"{per:g}": {f"c{c:g}": c / (2.0 * config.system(per).s * gs["g_x"]) for c in BASIN_C}
            for per in config.PERIODS_SHEAR_SCAN}
        # P3：失配
        M = [m for m in mis_rows if m["p"] == p]
        mis = {}
        for name in sorted({m["factor"] for m in M}):
            for lv in sorted({m["level"] for m in M if m["factor"] == name}):
                mm = [m for m in M if m["factor"] == name and m["level"] == lv]
                mis[f"{name}={lv:g}"] = {
                    "mle_bias_wf_median": float(np.median([m["pred_bias_wf_mle"] for m in mm])),
                    "fourier_bias_wf_median": float(np.median([m["pred_bias_wf_fourier"] for m in mm])),
                    "mle_bias_wf_notilt_median": float(np.median([m["pred_bias_wf_mle_notilt"] for m in mm])),
                    "fourier_bias_wf_notilt_median": float(np.median([m["pred_bias_wf_fourier_notilt"] for m in mm])),
                    "frac_fourier_better": float(np.mean([m["pred_bias_wf_fourier"] < m["pred_bias_wf_mle"] for m in mm])),
                }
        P["mismatch"] = mis
        S = [s for s in shear_rows if s["p"] == p]
        P["kappa_by_period"] = {f"{per:g}": float(np.median([s["kappa_JtJ"] for s in S if s["period_um"] == per]))
                                for per in config.PERIODS_SHEAR_SCAN} if S else {}
        pred[f"p{p}"] = P
    return pred


def main() -> None:
    t0 = time.time()
    jobs = [(p, tr) for p in (35, 12) for tr in truth_set("dev", config.indices_for(p))]
    rows = pmap(truth_worker, jobs)
    print(f"truths done {time.time() - t0:.0f}s")
    sel = [(p, tr) for p, tr in jobs if tr.rms in (0.1, 0.3, 1.0)]
    mis_rows = [r for rr in pmap(mismatch_worker, sel) for r in rr]
    print(f"mismatch done {time.time() - t0:.0f}s")
    sh_jobs = [(per, 35, tr) for per in config.PERIODS_SHEAR_SCAN
               for tr in truth_set("dev", config.INDICES_MAIN, rms_levels=(0.1, 0.3), shapes=("pow2", "flat"), reps=1)]
    shear_rows = pmap(shear_worker, sh_jobs)
    write_csv(OUT / "theory_truths.csv.gz", rows)
    write_csv(OUT / "mismatch_linear.csv.gz", mis_rows)
    write_csv(OUT / "shear_kappa.csv", shear_rows)
    pred = summarize(rows, mis_rows, shear_rows)
    write_json(OUT / "predictions.json", pred)
    print(f"all done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
