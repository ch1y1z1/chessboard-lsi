"""Part 2：代表 runs 的曲率/端点诊断 + 小噪声二阶项 + 条件 trust-exact。

输入：旧基准 runs.csv / trajectories.jsonl.gz / dataset.npz（只读）。
口径：q=RMS 坐标、r 归一化（同 StackedIntensityProblem）、全像素。
输出：$ARTIFACTS/part2_curvature_diagnostics.json、part2_points.csv、
      part3_noise_second_order.json、part3_trust_exact.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.carrier_curvature import common
from analysis.carrier_curvature.diagnostics import (
    classify_endpoint, perturbation_probe, point_diagnostics, trust_exact)
from analysis.carrier_curvature.hessian import fd_hessian_check
from analysis.carrier_curvature.instrument import run_benchmark_lm

TRUST_EXACT_MAX = 6
NOISE_SNRS = (40.0, 60.0)   # dB，sigma = max/10^(snr/20)，同旧基准定义
NOISE_SEEDS = (20261009, 20261010)


def _slim(diag):
    """压缩谱量存盘：保留关键标量与谱端值。"""
    keep = {k: v for k, v in diag.items() if k not in ("j_sv", "gn_eig",
                                                     "h_eig")}
    keep["j_sv_head"] = diag["j_sv"][:3]
    keep["j_sv_tail"] = diag["j_sv"][-3:]
    keep["h_eig_head"] = diag["h_eig"][:3]
    keep["h_eig_tail"] = diag["h_eig"][-3:]
    keep["gn_eig_head"] = diag["gn_eig"][:3]
    return keep


def scenario_diagnostics(fm, dataset, rows, rng, spec):
    from lsi.optimizers import SolverOptions

    case_id, initial_id, method, roles = spec
    row = common.find_run(rows, case_id, initial_id, method)
    record = common.load_trajectory(case_id, initial_id, method)
    problem = common.carrier_problem(fm, dataset, case_id)
    truth = dataset["truths"][case_id]
    final_q = problem.to_parameters(common.fitted_coeffs(row))
    points = common.trajectory_points(record, roles, final_q)

    out = {"case_id": case_id, "initial_id": initial_id, "method": method,
           "truth_rms": float(row["truth_rms"]),
           "initial_rms": float(row["initial_rms"]),
           "status": row["status"], "old_loss": float(row["loss"]),
           "old_wavefront_error": float(row["wavefront_rms_error"]),
           "old_intensity_rms": float(row["intensity_rms"]),
           "truth_sign": int(row["truth_sign"]),
           "strict_success": row["strict_success"] == "True",
           "n_forward": int(row["n_forward"]),
           "loss_curve": [float(ev["loss"]) for ev in record["history"]],
           "points": {}}

    for role, q in points.items():
        diag = point_diagnostics(problem, q)
        rng_dirs = np.random.default_rng(7000 + case_id * 100 + initial_id)
        fd_err = fd_hessian_check(
            problem, q, [rng_dirs.normal(size=q.size) for _ in range(3)])
        entry = {"q": np.asarray(q).tolist(), "diag": _slim(diag),
                 "fd_hessian_rel_err": fd_err}
        if role == "final":
            probe = perturbation_probe(problem, q, rng)
            entry["probe"] = probe
            entry["labels"] = classify_endpoint(diag, probe)
        out["points"][role] = entry

    # LM 场景的阻尼/接受率：重跑真实 instrumented 求解（小规模）。
    if method == "lm":
        c0 = dataset["initials"][initial_id]
        _, trials = run_benchmark_lm(
            problem, c0, SolverOptions(max_forward=500, max_jacobian=500))
        out["lm_damping"] = {
            "lam_seq": [t["lam"] for t in trials],
            "rho_seq": [t["rho"] for t in trials],
            "accepted": [t["accepted"] for t in trials],
            "n_rejected": sum(1 for t in trials if not t["accepted"]),
        }
    # GN 等线搜索方法：由 history 推导每接受步的评价数（含回溯）。
    hist = record["history"]
    improvements = sum(
        1 for a, b in zip(hist, hist[1:]) if b["loss"] < a["loss"] - 1e-300)
    out["eval_stats"] = {"n_events": len(hist),
                         "loss_improvements": improvements}
    return out


def noise_second_order(fm, dataset, case_id=0):
    """小噪声下真值点与拟合端点的残差二阶项规模（单独小节/分表）。"""
    frame = dataset["frames"][case_id, 0]
    mod = dataset["modulations"][case_id, 0]
    truth = dataset["truths"][case_id]
    results = []
    sigma0 = float(frame.max())
    variants = [(None, None)] + [(s, seed) for s in NOISE_SNRS
                                 for seed in NOISE_SEEDS]
    for snr, seed in variants:
        if snr is None:
            image = frame
        else:
            sigma = sigma0 / 10 ** (snr / 20)
            image = frame + np.random.default_rng(seed).normal(
                0, sigma, frame.shape)
        problem = common.make_problem(fm, image, mod)
        truth_q = problem.to_parameters(truth)
        diag_t = point_diagnostics(problem, truth_q)
        # 拟合端点：真实 instrumented benchmark LM（同预算）
        from lsi.optimizers import SolverOptions
        result, _ = run_benchmark_lm(
            problem, np.zeros(len(common.INDICES)),
            SolverOptions(max_forward=500, max_jacobian=500))
        end_q = problem.to_parameters(result.coeffs)
        diag_e = point_diagnostics(problem, end_q)
        results.append({"snr_db": snr, "seed": seed,
                        "at_truth": _slim(diag_t),
                        "at_endpoint": _slim(diag_e),
                        "endpoint_loss": result.loss,
                        "endpoint_status": result.status,
                        "endpoint_wavefront_error": float(min(
                            np.std(problem.pupil_basis
                                   @ (result.coeffs - truth)),
                            np.std(problem.pupil_basis
                                   @ (result.coeffs + truth))))})
    return results


def main():
    dataset = common.load_dataset()
    rows = common.load_runs()
    fm = common.make_forward()
    rng = np.random.default_rng(20261009)

    common.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    diagnostics = [scenario_diagnostics(fm, dataset, rows, rng, spec)
                   for spec in common.PART2_SCENARIOS]
    (common.ARTIFACTS / "part2_curvature_diagnostics.json").write_text(
        json.dumps({"scenarios": diagnostics}, indent=1, allow_nan=False))

    import csv
    with (common.ARTIFACTS / "part2_points.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "initial_id", "method", "point", "loss",
                    "g_inf", "j_cond", "h_min_eig", "h_neg_count",
                    "second_over_gn", "residual_rms", "labels",
                    "probe_random_hits"])
        for s in diagnostics:
            for role, p in s["points"].items():
                d = p["diag"]
                w.writerow([s["case_id"], s["initial_id"], s["method"],
                            role, d["loss"], d["g_inf"], d["j_cond"],
                            d["h_min_eig"], d["h_neg_count"],
                            d["second_over_gn"], d["residual_rms"],
                            "|".join(p.get("labels", [])),
                            p.get("probe", {}).get("random_hits")])

    # Part 3a：小噪声残差二阶项（独立于无噪结论，分表）
    noise = noise_second_order(fm, dataset, case_id=0)
    (common.ARTIFACTS / "part3_noise_second_order.json").write_text(
        json.dumps({"case_id": 0, "snr_db_list": list(NOISE_SNRS),
                    "seeds": list(NOISE_SEEDS), "results": noise},
                   indent=1, allow_nan=False))

    # Part 3b：trust-exact 仅用于有二阶诊断价值的端点（≤6）
    candidates = []
    for s in diagnostics:
        fin = s["points"].get("final")
        if fin is None:
            continue
        interesting = ("indefinite_hessian" in fin.get("labels", [])
                       or "descent_found" in fin.get("labels", [])
                       or s["method"] == "momentum")
        if interesting:
            candidates.append((s, fin))
    trust_runs = []
    for s, fin in candidates[:TRUST_EXACT_MAX]:
        problem = common.carrier_problem(fm, dataset, s["case_id"])
        q0 = np.asarray(fin["q"])
        trace = trust_exact(problem, q0, seed=0)
        truth = dataset["truths"][s["case_id"]]
        c = problem.to_coefficients(trace[-1]["q"])
        trust_runs.append({
            "case_id": s["case_id"], "initial_id": s["initial_id"],
            "method": s["method"], "start_loss": trace[0]["loss"],
            "end_loss": trace[-1]["loss"], "n_iter": trace[-1]["iter"],
            "end_g_inf": trace[-1]["g_inf"], "end_min_eig": trace[-1]["min_eig"],
            "end_wavefront_error": float(min(
                np.std(problem.pupil_basis @ (c - truth)),
                np.std(problem.pupil_basis @ (c + truth)))),
            "end_truth_sign_distance": float(min(
                np.max(np.abs(c - truth)), np.max(np.abs(c + truth)))),
            "trace": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in t.items() if k != "q"} | {"q": t["q"].tolist()}
                      for t in trace]})
    (common.ARTIFACTS / "part3_trust_exact.json").write_text(
        json.dumps({"runs": trust_runs,
                    "note": "诊断用途：精确二阶信息在端点邻域的行为，"
                            "不构成局部极小的判定或部署证据"},
                   indent=1, allow_nan=False))

    for s in diagnostics:
        fin = s["points"].get("final", {})
        print(s["case_id"], s["initial_id"], s["method"],
              "labels:", fin.get("labels"),
              "g_inf:", fin.get("diag", {}).get("g_inf"),
              "h_min:", fin.get("diag", {}).get("h_min_eig"))
    print("artifacts ->", common.ARTIFACTS)


if __name__ == "__main__":
    main()
