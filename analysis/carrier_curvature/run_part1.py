"""Part 1：主线 LM / benchmark LM / SciPy least_squares(method='lm') 比较。

两张表口径：
A. 原生默认配置——各实现自己的像素采样、归一化、参数坐标、容差与预算；
   场景 (case, init) 共享。
B. 统一口径——同一全像素集、同一原始残差 f(c)=I-I_obs、同一 c 坐标、
   同一 c0；benchmark LM 经 scaling='none' 的 StackedIntensityProblem 接入
   （其残差为 f/sqrt(M)，轨迹按尺度不变性等价，数值验证前缀一致）。

运行：python analysis/carrier_curvature/run_part1.py
输出：$ARTIFACTS/part1_lm_comparison.json + part1_lambda_rho.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.carrier_curvature import common
from analysis.carrier_curvature.instrument import (
    run_benchmark_lm, run_mainline_lm, run_scipy_lm)


def _raw_residual(problem):
    """统一口径的原始残差/雅可比闭包：f(c)=I(c)-I_obs，全像素。"""
    from lsi.lm import _frame_and_jacobian

    mod = problem.mods[0]
    obs = problem.blocks[0]

    def residual(c):
        I, _ = _frame_and_jacobian(problem.cache, np.asarray(c, float), mod,
                                   compute_jacobian=False)
        return I - obs

    def jacobian(c):
        _, J = _frame_and_jacobian(problem.cache, np.asarray(c, float), mod)
        return J

    return residual, jacobian


def _wavefront_error(problem, coeffs, truth):
    """离线评价：固定符号外的整体符号对齐波前 RMS（waves）。"""
    basis = problem.pupil_basis
    c, t = np.asarray(coeffs), np.asarray(truth)
    return float(min(np.std(basis @ (c - t)), np.std(basis @ (c + t))))


def _intensity_rms(residual, coeffs):
    f = residual(np.asarray(coeffs, float))
    return float(np.sqrt(np.mean(f * f)))


def run_scenario(fm, dataset, rows, case_id, initial_id):
    from lsi.optimizers import SolverOptions

    frame = dataset["frames"][case_id, 0]
    mod = dataset["modulations"][case_id, 0]
    truth = dataset["truths"][case_id]
    c0 = dataset["initials"][initial_id]
    run_row = common.find_run(rows, case_id, initial_id, "lm")
    out = {"case_id": case_id, "initial_id": initial_id,
           "truth_rms": float(run_row["truth_rms"]),
           "initial_rms": float(run_row["initial_rms"]),
           "old_lm": {"status": run_row["status"], "loss": float(run_row["loss"]),
                      "n_forward": int(run_row["n_forward"]),
                      "wavefront_rms_error": float(run_row["wavefront_rms_error"]),
                      "strict_success": run_row["strict_success"] == "True"}}

    prob_rms = common.make_problem(fm, frame, mod, scaling="rms")
    prob_none = common.make_problem(fm, frame, mod, scaling="none")
    residual, jacobian = _raw_residual(prob_none)

    # ---------- 表 A：原生默认配置 ----------
    a = {}
    res_m, trials_m, rows_sel, cost_log = run_mainline_lm(
        fm, common.INDICES, frame, c0=c0, samples=4096)
    a["mainline"] = {
        "cost_fsq": float(res_m.cost), "n_iter": res_m.n_iter,
        "converged": bool(res_m.converged),
        "rms_residual_subsample": res_m.rms_residual,
        "n_samples": int(rows_sel.size), "coeffs": res_m.coeffs.tolist(),
        "intensity_rms_full": _intensity_rms(residual, res_m.coeffs),
        "wavefront_error": _wavefront_error(prob_rms, res_m.coeffs, truth),
        "trials": trials_m}

    opt_def = SolverOptions(max_forward=500, max_jacobian=500)
    res_b, trials_b = run_benchmark_lm(prob_rms, c0, opt_def)
    a["benchmark"] = {
        "loss": res_b.loss, "status": res_b.status,
        "n_forward": res_b.n_forward, "n_jacobian": res_b.n_jacobian,
        "n_iter": res_b.n_iter, "coeffs": res_b.coeffs.tolist(),
        "intensity_rms_full": _intensity_rms(residual, res_b.coeffs),
        "wavefront_error": _wavefront_error(prob_rms, res_b.coeffs, truth),
        "trials": trials_b}

    def prob_residual(q):
        r, _ = prob_none.evaluate(q, jacobian=False)
        return r

    def prob_jacobian(q):
        _, j = prob_none.evaluate(q)
        return j

    res_s, calls_s = run_scipy_lm(prob_residual, prob_jacobian, c0)
    a["scipy_lm"] = {
        "cost_half_fsq_normalized": float(res_s.cost),
        "status": int(res_s.status), "message": res_s.message,
        "nfev": int(res_s.nfev), "njev": int(res_s.njev),
        "coeffs": res_s.x.tolist(),
        "intensity_rms_full": _intensity_rms(residual, res_s.x),
        "wavefront_error": _wavefront_error(prob_rms, res_s.x, truth),
        "eval_costs": [c["cost"] for c in calls_s]}
    out["native"] = a

    # ---------- 表 B：统一口径 ----------
    b = {}
    res_mu, trials_mu, _, _ = run_mainline_lm(
        fm, common.INDICES, frame, c0=c0, samples=None)
    b["mainline_full"] = {
        "cost_fsq": float(res_mu.cost), "n_iter": res_mu.n_iter,
        "converged": bool(res_mu.converged), "coeffs": res_mu.coeffs.tolist(),
        "intensity_rms_full": _intensity_rms(residual, res_mu.coeffs),
        "wavefront_error": _wavefront_error(prob_rms, res_mu.coeffs, truth),
        "trials": trials_mu}

    res_bn, trials_bn = run_benchmark_lm(prob_none, c0, opt_def)
    b["benchmark_none"] = {
        "loss": res_bn.loss, "status": res_bn.status,
        "n_forward": res_bn.n_forward, "n_iter": res_bn.n_iter,
        "coeffs": res_bn.coeffs.tolist(),
        "intensity_rms_full": _intensity_rms(residual, res_bn.coeffs),
        "wavefront_error": _wavefront_error(prob_rms, res_bn.coeffs, truth),
        "trials": trials_bn}

    # 同口径 + 宽松容差：检验轨迹是否在主线停机前完全一致
    opt_tight = SolverOptions(max_forward=2000, max_jacobian=2000,
                              gtol=1e-15, ftol=1e-15, xtol=1e-15)
    res_bt, trials_bt = run_benchmark_lm(prob_none, c0, opt_tight)
    b["benchmark_none_tight"] = {
        "loss": res_bt.loss, "status": res_bt.status,
        "n_forward": res_bt.n_forward, "n_iter": res_bt.n_iter,
        "coeffs": res_bt.coeffs.tolist(),
        "trials": trials_bt}

    res_su, calls_su = run_scipy_lm(residual, jacobian, c0)
    b["scipy_lm_raw"] = {
        "cost_fsq": 2.0 * float(res_su.cost), "status": int(res_su.status),
        "message": res_su.message, "nfev": int(res_su.nfev),
        "njev": int(res_su.njev), "coeffs": res_su.x.tolist(),
        "intensity_rms_full": _intensity_rms(residual, res_su.x),
        "wavefront_error": _wavefront_error(prob_rms, res_su.x, truth),
        "eval_costs": [c["cost"] for c in calls_su]}

    # 前缀一致性：主线 λ/ρ/迭代点 vs benchmark（scaling='none'，等价轨迹）
    def _prefix_cmp(trials_x, trials_y):
        n = min(len(trials_x), len(trials_y))
        mism = []
        for i in range(n):
            a_, b_ = trials_x[i], trials_y[i]
            cx = a_.get("c_trial"); cy = b_.get("q_trial")
            d_lam = abs(a_["lam"] - b_["lam"])
            ra, rb = a_.get("rho"), b_.get("rho")
            d_rho = (None if ra is None or rb is None else abs(ra - rb))
            d_x = (float(np.max(np.abs(np.asarray(cx) - np.asarray(cy))))
                   if cx is not None else None)
            mism.append({"i": i, "d_lam": d_lam, "d_rho": d_rho, "d_x": d_x,
                         "accepted_match": a_["accepted"] == b_["accepted"]})
        return {"prefix_len": n, "len_x": len(trials_x), "len_y": len(trials_y),
                "max_d_lam": max(m["d_lam"] for m in mism) if mism else None,
                "max_d_x": max(m["d_x"] for m in mism) if mism else None,
                "accept_all_match": all(m["accepted_match"] for m in mism),
                "per_iter": mism}

    b["prefix_vs_default"] = _prefix_cmp(trials_mu, trials_bn)
    b["prefix_vs_tight"] = _prefix_cmp(trials_mu, trials_bt)
    b["endpoint_dist_mainline_vs_scipy"] = float(
        np.max(np.abs(res_mu.coeffs - res_su.x)))
    b["endpoint_dist_mainline_vs_bench_tight"] = float(
        np.max(np.abs(res_mu.coeffs - res_bt.coeffs)))
    out["unified"] = b
    return out


def main():
    dataset = common.load_dataset()
    rows = common.load_runs()
    fm = common.make_forward()
    # 旧数据调制与本模型载频相位一致性（溯源校验）
    np.testing.assert_allclose(
        dataset["modulations"][0, 0],
        fm.carrier_phases(fm.config.carrier_f0), rtol=0, atol=1e-12)
    results = [run_scenario(fm, dataset, rows, c, i)
               for c, i in common.PART1_SCENARIOS]
    common.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    payload = {"scenarios": results}
    out_json = common.ARTIFACTS / "part1_lm_comparison.json"
    out_json.write_text(json.dumps(payload, indent=1, allow_nan=False))
    # λ/ρ 轨迹表
    import csv
    with (common.ARTIFACTS / "part1_lambda_rho.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "initial_id", "impl", "iter", "lam", "rho",
                    "accepted", "loss_or_cost", "step_norm"])
        for s in results:
            for impl, key in (("mainline_native", "mainline"),
                              ("benchmark_native", "benchmark")):
                for k, t in enumerate(s["native"][key]["trials"]):
                    w.writerow([s["case_id"], s["initial_id"], impl, k,
                                t["lam"], t.get("rho"), t["accepted"],
                                t.get("cost_trial", t.get("loss_trial")),
                                t["step_norm"]])
            for impl, key in (("mainline_unified", "mainline_full"),
                              ("bench_none_default", "benchmark_none"),
                              ("bench_none_tight", "benchmark_none_tight")):
                for k, t in enumerate(s["unified"][key]["trials"]):
                    w.writerow([s["case_id"], s["initial_id"], impl, k,
                                t["lam"], t.get("rho"), t["accepted"],
                                t.get("cost_trial", t.get("loss_trial")),
                                t["step_norm"]])
    print(json.dumps([{k: s[k] for k in ("case_id", "initial_id")} | {
        "native": {m: {"wfe": s["native"][m].get("wavefront_error"),
                       "status": s["native"][m].get("status",
                            s["native"][m].get("converged"))}
                   for m in ("mainline", "benchmark", "scipy_lm")},
        "unified_max_dx": s["unified"]["prefix_vs_tight"]["max_d_x"],
        "unified_accept_match": s["unified"]["prefix_vs_tight"]["accept_all_match"]}
        for s in results], indent=1))
    print("wrote", out_json)


if __name__ == "__main__":
    main()
