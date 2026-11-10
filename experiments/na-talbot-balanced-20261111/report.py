"""Read saved coefficients and arrays; never propagate or optimize."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
from scipy import fft
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from lsi.basis import basis

LABELS = {"single_ft": "Single FT", "single_lm": "Single LM",
          "multi64_ft": "64-source FT", "multi64_lm": "64-source LM"}
COLORS = {"single_ft": "#aa4c26", "single_lm": "#21628c",
          "multi64_ft": "#bb8e24", "multi64_lm": "#27836e"}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+"\n")


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))


def pupil_basis(cfg):
    n = cfg["N_lattice"]*cfg["N_super_lattice"]
    freq = fft.fftshift(fft.fftfreq(n, cfg["super_lattice_period"]/cfg["N_lattice"]))
    bound = cfg["NA_out"]/cfg["wavelength"]
    active = np.nonzero(abs(freq) <= bound*(1+1e-12))[0]
    fx, fy = np.meshgrid(freq[active], freq[active])
    rho = cfg["wavelength"]*np.hypot(fx, fy)/cfg["NA_out"]
    eps = cfg["NA_in"]/cfg["NA_out"]
    mask = (rho >= eps) & (rho <= 1)
    return basis(tuple(range(1, 36)), rho, np.arctan2(fy, fx), eps, cfg["Nz"]), mask, rho


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--provenance", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-commit", required=True)
    args = p.parse_args()
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    assert head == args.expected_commit
    dirty = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain",
                                    "--untracked-files=no"], text=True)
    assert not dirty, dirty
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output/"analysis-version.json", dict(commit=head,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        command=sys.argv, python=sys.version, numpy=np.__version__,
        generated_utc=datetime.now(timezone.utc).isoformat()))
    rows = read(args.results/"summary.json")["results"]
    assert len(rows) == 112
    flight = read(args.preflight/"preflight.json")
    assert flight["completed_na"] == 4 and flight["passed"]
    audit = []
    bases = {}
    for na in (.020, .066, .110, .150):
        cfg = read(args.results/f"na{na:.3f}_single/config.json")
        bases[na] = pupil_basis(cfg)
    for row in rows:
        Z, mask, rho = bases[row["na_out"]]
        error = np.tensordot(np.array(row["coefficients"])-row["truth"], Z, axes=1)
        true = np.tensordot(row["truth"], Z, axes=1)
        computed = dict(raw_rms_waves=rms(error[mask]),
                        raw_pv_waves=float(np.ptp(error[mask])),
                        outer10_rms_waves=rms(error[mask & (rho >= .9)]),
                        common_na002_rms_waves=rms(error[mask & (rho*row["na_out"] <= .02+1e-14)]))
        for key, value in computed.items():
            delta = abs(value-row[key])
            assert delta < 1e-10, (row["na_out"], row["method"], key, delta)
            audit.append(dict(check=key, difference=delta, passed=True))
        assert abs(rms(true[mask])-.05) < 1e-12
        assert int(mask.sum()) == row["evaluation_pixels"]
        audit.extend([dict(check="truth_raw_rms", passed=True),
                      dict(check="complete_pupil_count", passed=True)])
    pairs = {}
    for row in rows:
        key = (row["na_out"], row["source_kind"], row["condition"])
        pairs.setdefault(key, []).append(row)
    for key, pair in pairs.items():
        assert len(pair) == 2 and pair[0]["observed_sha256"] == pair[1]["observed_sha256"]
        assert {x["method"] for x in pair} == {"ft", "lm"}
        audit.append(dict(check="paired_observation_sha256", passed=True))
        na, kind, condition = key
        observed = np.load(args.results/f"na{na:.3f}_{kind}"/f"observation_{condition}.npy")
        actual = hashlib.sha256(np.ascontiguousarray(observed).tobytes()).hexdigest()
        assert actual == pair[0]["observed_sha256"]
        audit.append(dict(check="saved_observation_array_sha256", passed=True))
        del observed
    for field in flight["rows"]:
        edges = read(args.preflight/f'na{field["na_out"]:.3f}'/"source_edges.json")
        assert len(edges) == 128 and all(x["finite"] for x in edges)
        audit.append(dict(check="128_finite_source_fields", passed=True))
    save(args.output/"audit.json", dict(passed=True, checks=len(audit), items=audit,
         description="Independent basis/metric recalculation and saved SHA pairing; not independent physical truth."))
    columns = ["na_out", "source_kind", "method", "condition", "snr_db", "seed",
               "raw_rms_waves", "raw_rms_nm", "raw_pv_waves", "relative_rms",
               "outer10_rms_waves", "common_na002_rms_waves",
               "coefficient_max_error_waves", "evaluation_pixels", "observed_sha256"]
    with (args.output/"metrics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows({k: r[k] for k in columns} for r in rows)
    with (args.output/"coefficients.csv").open("w") as f:
        names = ["na_out", "source_kind", "method", "condition", "mode",
                 "truth_waves", "estimate_waves", "error_waves"]
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        for row in rows:
            for j, (target, estimate) in enumerate(zip(row["truth"], row["coefficients"]), 1):
                w.writerow(dict({k: row[k] for k in names[:4]}, mode=j,
                    truth_waves=target, estimate_waves=estimate, error_waves=estimate-target))
    with (args.output/"radial.csv").open("w") as f:
        names = ["na_out", "source_kind", "method", "condition",
                 "low", "high", "pixels", "rms_waves"]
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        for row in rows:
            for ring in row["radial"]:
                w.writerow({**{k: row[k] for k in names[:4]}, **ring})
    stats = []
    for na in bases:
        for method in LABELS:
            kind, solver = method.rsplit("_", 1)
            subset = [r for r in rows if r["na_out"] == na and
                      r["source_kind"] == kind and r["method"] == solver]
            samples = [r["raw_rms_waves"] for r in subset if r["snr_db"] == 40]
            clean = next(r for r in subset if r["condition"] == "clean")
            noisy30 = next(r for r in subset if r["snr_db"] == 30)
            noisy42 = next(r for r in subset if r["condition"] == "40db_seed42")
            stats.append(dict(na_out=na, pipeline=method, clean=clean["raw_rms_waves"],
                mean40=float(np.mean(samples)), sd40=float(np.std(samples, ddof=1)),
                seed42_40=noisy42["raw_rms_waves"], seed42_30=noisy30["raw_rms_waves"],
                outer10_40=noisy42["outer10_rms_waves"],
                common_na002_40=noisy42["common_na002_rms_waves"]))
    save(args.output/"statistics.json", stats)
    with (args.output/"statistics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    def finish(fig, name):
        fig.savefig(args.output/(name+".png"), dpi=180, bbox_inches="tight")
        fig.savefig(args.output/(name+".pdf"), bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for method in LABELS:
        sub = [r for r in stats if r["pipeline"] == method]
        x = [r["na_out"] for r in sub]
        ax[0].semilogy(x, [max(r["clean"], 1e-17) for r in sub], "o-", color=COLORS[method], label=LABELS[method])
        ax[1].errorbar(x, [r["mean40"] for r in sub], yerr=[r["sd40"] for r in sub],
                       color=COLORS[method], marker="o", capsize=3, label=LABELS[method])
        ax[2].semilogy(x, [r["seed42_30"] for r in sub], "o-", color=COLORS[method])
    ax[1].set_yscale("log")
    for a, title in zip(ax, ("Clean (model matched)", "40 dB: mean +/- sample SD (5 seeds)", "30 dB: seed 42")):
        a.set(xlabel="NA_out", ylabel="Raw full-pupil RMS (waves)", title=title)
        a.grid(alpha=.18)
    ax[1].legend(fontsize=8)
    finish(fig, "rms_vs_na")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True, sharey=True)
    for a, na in zip(axes.flat, bases):
        for method in LABELS:
            kind, solver = method.rsplit("_", 1)
            row = next(r for r in rows if r["na_out"] == na and r["source_kind"] == kind
                       and r["method"] == solver and r["condition"] == "40db_seed42")
            a.semilogy([(r["low"]+r["high"])/2 for r in row["radial"]],
                       [r["rms_waves"] for r in row["radial"]], "o-", color=COLORS[method], label=LABELS[method])
        a.set(title=f"NA_out={na:.3f}", xlabel="Normalized pupil radius", ylabel="RMS (waves)")
        a.grid(alpha=.18)
    axes.flat[0].legend(fontsize=8)
    finish(fig, "radial_40db_seed42")

    global_wave_limit, global_error_limit = 0., 0.
    for na, (Z, mask, rho) in bases.items():
        for row in rows:
            if row["na_out"] != na or row["condition"] != "40db_seed42":
                continue
            true = np.tensordot(row["truth"], Z, axes=1)
            fitted = np.tensordot(row["coefficients"], Z, axes=1)
            global_wave_limit = max(global_wave_limit, float(abs(true[mask]).max()),
                                    float(abs(fitted[mask]).max()))
            global_error_limit = max(global_error_limit, float(abs((fitted-true)[mask]).max()))
    for na, (Z, mask, rho) in bases.items():
        chosen = []
        for method in LABELS:
            kind, solver = method.rsplit("_", 1)
            chosen.append(next(r for r in rows if r["na_out"] == na and r["source_kind"] == kind
                and r["method"] == solver and r["condition"] == "40db_seed42"))
        true = np.tensordot(chosen[0]["truth"], Z, axes=1)
        errors = [np.tensordot(np.array(r["coefficients"])-r["truth"], Z, axes=1) for r in chosen]
        amp, errlim = global_wave_limit, global_error_limit
        fig, ax = plt.subplots(4, 3, figsize=(9, 10), constrained_layout=True)
        maps = []
        for i, (method, error, row) in enumerate(zip(LABELS, errors, chosen)):
            for j, arr in enumerate((true, true+error, error)):
                im = ax[i, j].imshow(np.where(mask, arr, np.nan), origin="lower", cmap="RdBu_r",
                    vmin=-(errlim if j == 2 else amp), vmax=(errlim if j == 2 else amp))
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                if i == 0:
                    ax[i, j].set_title(("Truth", "Reconstruction", "Error")[j])
                    maps.append(im)
                if j == 0:
                    ax[i, j].set_ylabel(LABELS[method])
            ax[i, 2].set_xlabel(f'RMS={row["raw_rms_waves"]:.5g} waves')
        fig.colorbar(maps[0], ax=ax[:, :2], shrink=.65, label="Wavefront (waves)")
        fig.colorbar(maps[2], ax=ax[:, 2], shrink=.65, label="Error (waves)")
        fig.suptitle(f"NA_out={na:.3f}; 40 dB seed 42; scales shared across all NA/methods")
        finish(fig, f"wavefronts_na{na:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for k, key in enumerate(("image_outer5_energy", "grating_outer5_energy", "camera_outer5_energy")):
        ax[k].semilogy([r["na_out"] for r in flight["rows"]],
                       [100*r["maxima"][key] for r in flight["rows"]], "o-", color="#21628c")
        limit = read(args.results/"version.json")["protocol"]["limits"][key]*100
        ax[k].axhline(limit, color="#aa4c26", linestyle="--", label="Declared limit")
        ax[k].set(title=key.replace("_outer5_energy", "").title(),
                  xlabel="NA_out", ylabel="Worst outer 5% energy (%)")
        ax[k].grid(alpha=.18)
        ax[k].legend(fontsize=8)
    finish(fig, "boundary_diagnostics")

    spectra = []
    fig, ax = plt.subplots(4, 2, figsize=(10, 13), constrained_layout=True)
    for i, na in enumerate(bases):
        for j, kind in enumerate(("single", "multi64")):
            folder = args.results/f"na{na:.3f}_{kind}"
            zero = np.load(folder/"zero.npy")
            n = zero.shape[0]
            spectrum = abs(fft.fftshift(fft.fft2((zero-zero.min())/zero.max())))
            dc = float(spectrum[n//2, n//2])
            bins = read(folder/"geometry.json")["fundamental_bins"]
            k = round(bins)
            peaks = []
            for axis in (1, 0):
                cy, cx = n//2+(k if axis == 0 else 0), n//2+(k if axis == 1 else 0)
                local = spectrum[cy-3:cy+4, cx-3:cx+4]
                dy, dx = np.unravel_index(np.argmax(local), local.shape)
                peaks.append(dict(axis="x" if axis == 1 else "y",
                    peak_xy_bins=[cx+int(dx)-3-n//2, cy+int(dy)-3-n//2],
                    peak_over_dc=float(local[dy, dx]/dc)))
            bound = int(np.ceil(2.4*bins))
            crop = spectrum[n//2-bound:n//2+bound+1, n//2-bound:n//2+bound+1]
            spectral_im = ax[i, j].imshow(np.log10(np.maximum(crop/dc, 1e-7)), origin="lower",
                extent=(-bound, bound, -bound, bound), cmap="magma", vmin=-7, vmax=0)
            ax[i, j].set(title=f"NA={na:.3f}, {kind}; log10 amplitude / DC",
                          xlabel="Horizontal FFT bin", ylabel="Vertical FFT bin")
            spectra.append(dict(na_out=na, source_kind=kind,
                geometry_fundamental_bins=bins, geometry_selected_bin=k,
                local_peaks=peaks,
                scope="Local +/-3-bin amplitude/DC diagnostic; not Michelson visibility or a global carrier search."))
            del zero, spectrum, crop
    fig.colorbar(spectral_im, ax=ax, shrink=.7, label="log10 FFT amplitude / DC")
    finish(fig, "zero_reference_spectra")
    save(args.output/"spectrum_diagnostics.json", spectra)

    calibration_rows = []
    for na in bases:
        for kind in ("single", "multi64"):
            folder = args.results/f"na{na:.3f}_{kind}"
            cal = read(folder/"calibration.json")
            holdout = read(folder/"holdout.json")
            calibration_rows.append(dict(na_out=na, source_kind=kind, **cal,
                holdout_rms_waves=[r["raw_rms_waves"] for r in holdout]))
    save(args.output/"calibration_summary.json", calibration_rows)
    failures = [dict(na_out=r["na_out"], source_kind=r["source_kind"], condition=r["condition"],
                message=r["diagnostics"]["message"]) for r in rows if r["method"] == "lm" and not r["diagnostics"]["success"]]
    branches = [dict(na_out=r["na_out"], source_kind=r["source_kind"], condition=r["condition"])
                for r in rows if r["method"] == "ft" and not r["diagnostics"]["small_phase_valid"]]
    save(args.output/"limitations.json", dict(lm_not_converged=failures,
         ft_branch_diagnostic_failed=branches))
    prov = read(args.provenance)
    version = read(args.results/"version.json")
    status = read(args.results/"status.json")
    zone = ZoneInfo("Asia/Shanghai")
    started_bj = datetime.fromisoformat(version["start_utc"]).astimezone(zone).isoformat()
    finished_bj = datetime.fromisoformat(status["timestamp_utc"]).astimezone(zone).isoformat()
    lines = ["# 高 NA 波前恢复：泰伯与视场配平后的四组合实验", "",
        f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}", "",
        f"正式计算（北京时间）：{started_bj}—{finished_bj}；"
        f"主运行耗时 {status['elapsed_seconds']/60:.3f} 分钟；峰值 RSS {status['peak_rss_gib']:.3f} GiB。", "",
        f"112 条主结果已生成；独立复算与记录核对共 {len(audit)} 项通过。"
        f"LM 优化器成功标志未通过 {len(failures)} 条，FT 主值相位诊断未通过 {len(branches)} 条。", "",
        "## 研究问题与比较范围", "",
        "沿用 NA_out=0.020、0.066、0.110、0.150，比较单源 FT、单源 LM、64 源 FT、64 源 LM。"
        "本实验同步配平周期、传播距离、采样及光源间距，比较这一组系统内的恢复表现。"
        "它不是固定装置中只改变 NA 的单因素实验。NA_in=0.00769 保持不变，因此中心遮拦比随 NA 改变。", "",
        "源间距按 1/q 缩放，轴向失调按 1/q² 缩放，q=NA_out/0.020。"
        "环形 Zernike 的定义随遮拦比改变，四模式真值分别归一化为完整光瞳上的 0.05 waves RMS。"
        "因此各档共享模式编号与权重比例，不是完全相同的物理波前与源间距。", "",
        "## 主要结果", "", "误差为完整离散物镜光瞳上的原始 RMS，单位 waves；不去除均值、倾斜或离焦。"
        "40 dB 为 5 个独立噪声种子的均值和样本标准差。30 dB 只有 seed=42。", "",
        "| NA_out | 组合 | clean | 40 dB 均值 ± SD | 40 dB seed42 | 30 dB seed42 |",
        "|---:|---|---:|---:|---:|---:|"]
    for row in stats:
        lines.append(f'| {row["na_out"]:.3f} | {LABELS[row["pipeline"]]} | {row["clean"]:.10g} | '
            f'{row["mean40"]:.10g} ± {row["sd40"]:.4g} | {row["seed42_40"]:.10g} | {row["seed42_30"]:.10g} |')
    lines += ["", "![NA 与重建误差](rms_vs_na.png)", "",
        "完整 112 条 RMS、PV、nm、系数误差和 SHA256 在 metrics.csv；"
        "coefficients.csv 保存全部 35 项系数，radial.csv 保存径向分区数值。"
        "statistics.csv 同时列出外圈和共同绝对 NA≤0.020 区域的误差；共同区域的采样点数随档位改变，不能当作相同采样的独立试验。", "",
        "## 参数及泰伯条件", "",
        "R=z_grating−M²·shift_z，M=f2/f1=0.2；z_eff=RD/(R+D)=2p²/λ，D=z_cam。"
        "所有档位均按实际离散光栅周期核对。只改变 NA 本身并不改变这一近轴恒等式；"
        "旧扫描沿用的 1.34 µm 基准接近四分之一泰伯，后续单独改周期也未同步配平有效距离。", "",
        "| NA_out | p (µm) | z_grating (µm) | R (µm) | D (mm) | shift_z (µm) | 源距 (µm) | 泰伯比 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in flight["rows"]:
        na, g = row["na_out"], row["geometry"]
        cfg = read(args.results/f"na{na:.3f}_single/config.json")
        lines.append(f'| {na:.3f} | {cfg["grating_period"]*1e6:.10g} | {cfg["z_grating"]*1e6:.10g} | '
            f'{g["paraxial_radius_m"]*1e6:.10g} | {cfg["z_cam"]*1e3:.10g} | '
            f'{cfg["shift_z"]*1e6:.10g} | {cfg["super_lattice_period"]*1e6:.10g} | {g["talbot_ratio"]:.15g} |')
    lines += ["", "像面为 1680×1680，相机为 7560×7560，光栅为 16800×16800；光栅周期 200 点、方孔边长 100 点。"
        "像面、光栅、相机全宽分别为 9.38 µm/q、56.28 µm/q、75.6 mm/q。"
        "相比 11 月 9 日基准，本实验扩大了相机范围并加密光栅采样。", "",
        "## 逐源边界及原算子核对", "",
        "| NA_out | 像面外缘最大 (%) | 光栅外缘最大 (%) | 相机外缘最大 (%) | 光栅几何余量 (µm) | 相机一级包络余量 (mm) | 原参考光强 L2 |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in flight["rows"]:
        g, m = row["geometry"], row["maxima"]
        lines.append(f'| {row["na_out"]:.3f} | {100*m["image_outer5_energy"]:.8g} | '
            f'{100*m["grating_outer5_energy"]:.8g} | {100*m["camera_outer5_energy"]:.8g} | '
            f'{g["grating_sine_angle_margin_m"]*1e6:.8g} | {g["camera_sine_angle_first_order_margin_m"]*1e3:.8g} | '
            f'{row["reference_intensity_relative_l2"]:.5g} |')
    lines += ["", "每档检查 64 光源×零波前/目标波前，共 128 个独立场，四档共 512 个。"
        "几何余量还按 tan(arcsin(NA_out/M)) 计算；这只是一项更严格的包络检查，实际传播仍使用原参考 Fresnel 算子。"
        "所有相机场有限、非零；周期单元复振幅复现检查与原 Stage1/Stage2/Stage3 光强核对分别保存。", "",
        "光栅入射强度的边界诊断使用相同物理范围的独立 3360² 网格，间距小于有限像面支持所决定的强度 Nyquist 间隔。"
        "每档另以一个角落光源的零波前和目标波前对照 16800² 密网格，外缘能量绝对差须小于 10⁻⁵。"
        "相机场、正式观测、标定和拟合始终使用 16800² 光栅的精确可分离离散算子；"
        "原三段 CZT 传播用于逐档核对相机中央 1680² 公共坐标。"
        "独立相机网格另有整幅 CZT 和解析导数的小场景核对。诊断网格不替换实际光栅掩模。", "",
        "有限衍射场有非零尾部，不能保证所有衍射级在有限窗口外严格为零。"
        "这里的“覆盖通过”指几何包络、外缘能量和采样检查通过，不是重建误差上界。"
        "外缘阈值在开跑前固定为像面 0.3%、光栅 0.1%、相机 0.1%。", "",
        "![边界诊断](boundary_diagnostics.png)", "",
        "## 标定、相位与收敛", "",
        "FT 使用一倍基频、相邻级次差分 Z(x−s)−Z(x)、相应光瞳交集和独立零波前复参考。"
        "每档每种光源数均重新生成 70 张 ±0.02 waves 单模式标定图。"
        "3 个随机 0.05 waves 混合真值只做留出检验，不参与取窗、标定和参数选择。", "",
        "观测 FFT 和边带逆 FFT 使用完整相机网格；基函数和相位平滑限定在按几何确定的区域内，"
        "其边界与所有有效像素之间至少留出 4 个高斯标准差的零填充。"
        "check_features.py 将该实现与整幅网格实现对照，避免在光瞳外反复计算恒为零的相位。", "",
        "这条 FT 流程沿用 11 月 10 日基频匹配诊断的结论；本轮重新构造窗口、差分基和响应矩阵，未复用旧标定。"
        "干净观测按各自峰值归一化；噪声为独立高斯噪声，σ=10^(−SNR/20)·RMS(clean)。"
        "FT 与 LM 共用同一观测数组及 SHA256。源数比较采用相同归一化 SNR，"
        "没有固定总光子数、引入 Poisson 噪声或模拟实际曝光预算，不能据此推断 64 光源的实际光子效率。", "",
        "本轮相机视场与旧报告不同，基于整幅图 RMS 定义的同名 SNR 也不能直接当作相同曝光条件进行跨报告比较。"
        "单源位置为 (7,7)，64 源的横纵索引均为 3…10；源坐标约定与基准一致。", "",
        "| NA_out | 光源 | 标定秩 | 条件数 | 训练最大系数误差 | 留出 RMS (3 个) | 主值诊断均通过 |",
        "|---:|---|---:|---:|---:|---|---|"]
    for c in calibration_rows:
        lines.append(f'| {c["na_out"]:.3f} | {c["source_kind"]} | {c["rank"]}/35 | {c["condition"]:.7g} | '
            f'{c["training_max_coefficient_error"]:.7g} | '
            + ", ".join(f"{x:.7g}" for x in c["holdout_rms_waves"])
            + f' | {c["all_small_phase_valid"]} |')
    lines += ["", "FT 采用小像差范围的主值相位特征，并保存每张图的相位接近 ±π 的诊断。"
        "它不代替一般大像差的掩膜相位展开。出现诊断失败时，应将对应结果视为流程适用范围之外。", "",
        "LM 从零波前启动，35 项波前、正增益和常数背景共同拟合；解析 Jacobian，"
        "scipy.optimize.least_squares(method='lm')。4096 个固定随机相机像素参与拟合，"
        "候选区为按几何参数确定、覆盖全部源偏移光瞳并增加两倍一级剪切余量的圆盘，未用目标波前选择像素。"
        "并另算完整相机图残差。未让全部相机像素参与 LM。"
        "优化器停止标志和波前恢复误差分别记录。", "",
        f"优化器成功标志未通过列表：{json.dumps(failures, ensure_ascii=False)}", "",
        f"FT 观测主值诊断失败列表：{json.dumps(branches, ensure_ascii=False)}", "",
        "零参考频谱的横纵基频位置按几何计算；图中幅值均除以 DC，"
        "spectrum_diagnostics.json 记录预期位置附近 ±3 bin 的局部峰。"
        "该幅值比不等于 Michelson 可见度，也不是全频谱自动选峰。", "",
        "![零参考频谱](zero_reference_spectra.png)", "",
        "![径向误差](radial_40db_seed42.png)", "",
        "## 波前对照", ""]
    for na in bases:
        lines += [f"![NA_out={na:.3f} 真值、恢复与误差](wavefronts_na{na:.3f}.png)", ""]
    lines += ["## 结论边界", "",
        "这是原标量 4f/Fresnel 前向模型内的算法比较。NA_out/M 最高为 0.75，"
        "此时近轴传播的物理适用性需要另外验证。本轮没有加入矢量衍射、严格高 NA 物镜模型、"
        "严格非近轴光栅传播或真实仪器观测。近轴泰伯核对不能证明高 NA 的严格自成像。", "",
        "配平后横向和轴向尺度近似满足 Fresnel 相似律，相对剪切量和像素采样保持一致。"
        "若误差随 NA 变化不大，应结合相似律、固定 NA_in 导致的遮拦比变化以及物侧角谱高阶相位解释；"
        "不能据此断言任何固定装置在高 NA 下都同样准确。", "",
        "无噪声同模型 LM 的浮点级误差反映模型内部一致性，不代表实际仪器精度。"
        "五个种子只描述本真值和噪声模型下的重复性，不证明所有波前上的普遍算法优劣。", "",
        "## 代码版本与留存", "",
        f"运行编号：{prov.get('run_id', 'run-01')}。正式计算提交：`{version['commit']}`。", ""]
    for item in prov["repositories"]:
        lines += [f"- 仓库：{item['repository']}；用途：{item['purpose']}；完整 commit：`{item['commit']}`；"
                  f"Git tag：`{item['tag']}`；提交：{item['commit_url']}；标签：{item['tag_url']}。"]
    lines += ["", f"后处理提交：`{head}`；脚本及环境核对见 analysis-version.json。", "",
              "### 预检查与历史尝试", ""]
    for attempt in prov.get("attempt_history", []):
        lines.append(f"- `{attempt['id']}`：{attempt['status']}。")
    lines += ["", "运行命令、配置、源码 SHA256、环境与机器信息见 version.json 和 versions.json。"
        "正式运行前已上传并核对标签；后处理如有改动，另列提交和标签。"
        "历史运行记录保留，未修改 11 月 4 日、9 日、10 日结果。", "",
        "## 文件与复核", "",
        f"- `audit.json`：{len(audit)} 项独立复算和记录核对。独立指从保存系数重新构造基与指标，不是独立物理模型。",
        "- `metrics.csv`、`statistics.csv`、`coefficients.csv`、`radial.csv`：完整数值、系数及重复噪声统计。",
        "- `calibration_summary.json`、`limitations.json`：标定、留出检验与失败记录。",
        "- 服务器原始观测、逐源边界明细、标定特征、LM 轨迹与 30 秒状态日志随各运行目录保留。", ""]
    (args.output/"report-draft.md").write_text("\n".join(lines))
    save(args.output/"artifact-manifest.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.iterdir() if p.is_file() and p.name != "artifact-manifest.json"})
    print(json.dumps(dict(rows=len(rows), audit_checks=len(audit), lm_failures=len(failures),
                         ft_branch_failures=len(branches), output=str(args.output))))


if __name__ == "__main__":
    main()
