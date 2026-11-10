"""Create figures and a Chinese report exclusively from audited saved results."""
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np

from talbot_common import ROOT, RESULTS, REFERENCE_COMMIT, save

REPORT = ROOT/"report"
NAMES = dict(single_ft="Single FT (Lasso)", single_ft_matched="Single FT (OLS control)",
             single_lm="Single LM", multi64_ft="64-source FT (OLS)", multi64_lm="64-source LM")
MAIN = ("single_ft", "single_lm", "multi64_ft", "multi64_lm")
COLORS = dict(single_ft="#a74435", single_ft_matched="#ae7e27",
              single_lm="#006b6b", multi64_ft="#446eab", multi64_lm="#554f64")
FILES = dict(single_ft=("single", "ft_original"), single_ft_matched=("single", "ft_matched"),
             single_lm=("single", "lm"), multi64_ft=("multi64", "ft_matched"),
             multi64_lm=("multi64", "lm"))


def read(path):
    return json.loads(path.read_text())


def get(method, condition):
    folder, prefix = FILES[method]
    stem = RESULTS/folder/f"{prefix}_{condition}"
    return read(stem.with_suffix(".json")), np.load(stem.with_name(stem.name+"_wavefront.npz"))


def finish(fig, name):
    fig.savefig(REPORT/f"{name}.png", dpi=180)
    fig.savefig(REPORT/f"{name}.pdf")
    plt.close(fig)


def wavefronts(condition):
    pairs = [get(method, condition) for method in MAIN]
    mask = pairs[0][1]["pupil"]
    truth = pairs[0][1]["truth"]
    waves = [truth, *[a["reconstruction"] for _, a in pairs]]
    errors = [a["error"] for _, a in pairs]
    maximum = max(np.max(abs(w[mask])) for w in waves)
    error_maximum = max(np.max(abs(e[mask])) for e in errors)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.3), layout="constrained")
    norm, error_norm = Normalize(-maximum, maximum), Normalize(-error_maximum, error_maximum)
    for ax, wave, label in zip(axes[0], waves, ["Truth", *[NAMES[m] for m in MAIN]]):
        image = ax.imshow(np.ma.array(wave, mask=~mask), cmap="RdBu_r", norm=norm,
                          origin="lower", interpolation="nearest")
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(image, ax=list(axes[0]), shrink=.8, label="Wavefront [waves]; shared scale")
    axes[1,0].axis("off")
    axes[1,0].text(.02, .85,
        f"{condition}\nTrue RMS: 0.05 waves\n35 free modes, piston = 0\n"
        "Raw error: no removal\nDisplay cropped to pupil\nPropagation: 1680 / 3360\n"
        "LM enlarged errors:\nsee separate figure",
        va="top", fontsize=11, linespacing=1.8)
    for ax, (row, arrays), method in zip(axes[1,1:], pairs, MAIN):
        image = ax.imshow(np.ma.array(arrays["error"], mask=~mask), cmap="RdBu_r",
                          norm=error_norm, origin="lower", interpolation="nearest")
        ax.set_title(f"Error RMS {row['raw_rms_waves']:.4g} waves", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(image, ax=list(axes[1,1:]), shrink=.8,
                 label="Reconstruction - truth [waves]; shared scale")
    fig.suptitle(f"New Talbot baseline | {condition} | common objective pupil", fontsize=15)
    finish(fig, f"wavefronts_{condition}")
    fig, axes = plt.subplots(1, 2, figsize=(10,4.3), layout="constrained")
    lm_pairs = [get(method,condition) for method in ("single_lm","multi64_lm")]
    maximum = max(np.max(abs(a["error"][mask])) for _, a in lm_pairs)
    for ax, (row, arrays), method in zip(axes, lm_pairs, ("single_lm","multi64_lm")):
        image = ax.imshow(np.ma.array(arrays["error"], mask=~mask), cmap="RdBu_r",
                          norm=Normalize(-maximum,maximum), origin="lower", interpolation="nearest")
        ax.set_title(f"{NAMES[method]}\nRMS {row['raw_rms_waves']:.5g} waves")
        ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(image, ax=list(axes), label="LM error [waves]; separate enlarged scale")
    fig.suptitle(f"LM errors | {condition} | scale differs from main comparison")
    finish(fig, f"lm_error_zoom_{condition}")


def radial_and_coefficients():
    fig, axes = plt.subplots(1,2,figsize=(13,4.8),layout="constrained")
    for ax, condition in zip(axes, ("clean","40db")):
        for method in (*MAIN,"single_ft_matched"):
            row,_ = get(method,condition)
            x = [(r["low"]+r["high"])/2 for r in row["radial"]]
            y = [max(r["rms_waves"],1e-18) for r in row["radial"]]
            ax.semilogy(x,y,"o--" if method=="single_ft_matched" else "o-",
                        color=COLORS[method], label=NAMES[method], markersize=4)
        ax.set_title(condition);ax.set_xlabel("Normalized pupil radius rho")
        ax.set_ylabel("Annular-bin raw error RMS [waves]")
        ax.grid(alpha=.2,which="both")
    axes[1].legend(fontsize=8)
    fig.suptitle("Same full pupil; 10 equal-radius annuli; no low-order removal")
    finish(fig,"radial_errors")
    fig,axes=plt.subplots(2,1,figsize=(13,7),layout="constrained")
    for ax, condition in zip(axes,("clean","40db")):
        for method in (*MAIN,"single_ft_matched"):
            row,_=get(method,condition)
            errors=np.array(row["coefficient_error_waves_36"])[1:]
            ax.plot(np.arange(1,36),errors,".--" if method=="single_ft_matched" else ".-",
                    label=NAMES[method],color=COLORS[method])
        ax.axhline(0,color="black",lw=.6);ax.set_title(condition)
        ax.set_xlabel("Reference zero-based mode index j (piston omitted)")
        ax.set_ylabel("Coefficient error [waves]");ax.grid(alpha=.2)
    axes[0].legend(ncol=3,fontsize=8)
    finish(fig,"coefficient_errors")


def calibration_and_spectra():
    fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
    for kind,tag,method in (("single","original","single_ft"),
                            ("single","matched","single_ft_matched"),
                            ("multi64","matched","multi64_ft")):
        row=read(RESULTS/kind/f"{tag}_calibration.json")
        singular=np.array(row["singular_values"])
        ax.semilogy(np.arange(1,36),np.maximum(singular/singular[0],1e-18),".-",
                    color=COLORS[method],label=f"{NAMES[method]}: rank {row['effective_rank']}/35")
        ax.axhline(row["rcond"],color=COLORS[method],alpha=.4,ls="--")
    ax.set_xlabel("Singular-value index");ax.set_ylabel("sigma / sigma_max")
    ax.set_title("Fresh independent FT response matrices; dashed = SVD cutoffs")
    ax.legend(fontsize=9);ax.grid(alpha=.2,which="both")
    finish(fig,"calibration_singular_values")
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout="constrained")
    for col,kind in enumerate(("single","multi64")):
        data=np.load(RESULTS/kind/"sideband_diagnostics.npz")
        geometry=read(RESULTS/kind/"sideband_geometry.json")
        zero=np.load(RESULTS/kind/"zero_raw.npy")
        zero=zero-zero.min()
        spectrum=abs(np.fft.fftshift(np.fft.fft2(zero/zero.max())))
        middle=zero.shape[0]//2
        crop=spectrum[middle-220:middle+221,middle-220:middle+221]
        np.savez_compressed(REPORT/f"sideband_plot_data_{kind}.npz",
                            spectrum_crop=crop,x_profile=data["x_profile"],y_profile=data["y_profile"])
        radius=(crop.shape[0]-1)//2
        axes[0,col].imshow(np.log10(np.maximum(crop/crop.max(),1e-8)),
            origin="lower",extent=(-radius-.5,radius+.5,-radius-.5,radius+.5),
            vmin=-6,vmax=0,cmap="magma",interpolation="nearest")
        carrier=geometry["selected_carrier_bins"];width=geometry["half_width_bins"]
        for x,y in ((carrier,0),(0,carrier)):
            axes[0,col].add_patch(Rectangle((x-width,y-width),2*width,2*width,
                                           fill=False,edgecolor="#8ce5e5",lw=1.5))
        axes[0,col].set_title(f"{kind}: zero-wavefront FFT; selected +/-1 bands")
        axes[0,col].set_xlabel("x FFT offset [bins]");axes[0,col].set_ylabel("y FFT offset [bins]")
        for key,color in (("x_profile","#446eab"),("y_profile","#a74435")):
            profile=data[key]
            axes[1,col].semilogy(np.arange(1,220),profile[1:220]/profile[0],
                                label=key[0]+" axis",color=color)
        axes[1,col].axvline(geometry["shadow_fundamental_bins"],color="gray",ls=":",
                            label="0/+1 carrier (~63)")
        axes[1,col].axvline(carrier,color="teal",ls="--",label="selected -1/+1 (~125)")
        axes[1,col].axvspan(carrier-width,carrier+width,color="teal",alpha=.12)
        axes[1,col].set_ylim(1e-7,1);axes[1,col].set_xlabel("Positive FFT offset [bins]")
        axes[1,col].set_ylabel("Amplitude / DC amplitude");axes[1,col].grid(alpha=.2)
        axes[1,col].legend(fontsize=8)
    finish(fig,"sideband_diagnostics")


def convergence():
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout="constrained")
    for ax,condition in zip(axes,("clean","40db")):
        for kind,method in (("single","single_lm"),("multi64","multi64_lm")):
            lines=(RESULTS/kind/f"lm_{condition}_trajectory.jsonl").read_text().splitlines()
            trace=[json.loads(line) for line in lines]
            ax.semilogy([r["joint_evaluation"] for r in trace],
                        [max(r["cost"],1e-35) for r in trace],".-",color=COLORS[method],
                        label=NAMES[method])
        ax.set_title(condition);ax.set_xlabel("Joint residual/Jacobian evaluations")
        ax.set_ylabel("Sum of squared sampled intensity residuals")
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle("Zero-wavefront starts; trial evaluations included")
    finish(fig,"lm_convergence")


def number(value):
    return "—" if value is None else f"{value:.7g}"


def write_report():
    rows=[get(method,condition)[0] for condition in ("clean","40db")
          for method in (*MAIN,"single_ft_matched")]
    fields=["method","condition","raw_rms_waves","raw_rms_nm","raw_pv_waves",
            "relative_rms","coefficient_rms_waves","coefficient_max_abs_waves",
            "outer10_rms_waves","full_intensity_rmse","full_intensity_nrmse","seconds"]
    with (REPORT/"metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)
    with (REPORT/"coefficients.csv").open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["method","condition","index","truth_waves","fitted_waves","error_waves"])
        for row in rows:
            for j in range(36):
                writer.writerow([row["method"],row["condition"],j,
                    row["truth_coefficients_waves_36"][j],row["coefficients_waves_36"][j],
                    row["coefficient_error_waves_36"][j]])
    preflight=read(RESULTS/"preflight.json")
    audit=read(RESULTS/"audit.json")
    single=read(RESULTS/"single/observation.json")
    multi=read(RESULTS/"multi64/observation.json")
    geometry=read(RESULTS/"single/sideband_geometry.json")
    main_table=["| 条件 | 方法 | raw RMS / waves | RMS / nm | raw PV / waves | 系数 RMS / waves |",
                "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        main_table.append("| "+" | ".join([row["condition"],NAMES[row["method"]],
            number(row["raw_rms_waves"]),number(row["raw_rms_nm"]),number(row["raw_pv_waves"]),
            number(row["coefficient_rms_waves"])])+" |")
    calibration_table=["| FT 流程 | 步长 / waves | 有效秩 | 全秩条件数 | 保留子空间条件数 | dZ 条件数 |",
                       "|---|---:|---:|---:|---:|---:|"]
    holdout_table=["| 流程 | 独立混合真值 RMS | 重建 raw RMS / waves |",
                   "|---|---:|---:|"]
    training_table=["| FT 流程 | 单模式训练回解最大系数误差 / waves |",
                    "|---|---:|"]
    for kind,tag,method in (("single","original","single_ft"),("single","matched","single_ft_matched"),
                            ("multi64","matched","multi64_ft")):
        row=read(RESULTS/kind/f"{tag}_calibration.json")
        calibration_table.append("| "+" | ".join([NAMES[method],number(row["amplitude_waves"]),
            f"{row['effective_rank']}/35",number(row["full_rank_condition_2"]),
            number(row["retained_condition_2"]),number(row["dZ_condition_2"])])+" |")
        training_table.append(f"| {NAMES[method]} | {number(row['train_max_coefficient_error_waves'])} |")
        for validation in read(RESULTS/kind/f"{tag}_holdout_validation.json"):
            holdout_table.append(f"| {NAMES[method]} | {validation['target_rms_waves']} | "
                                 f"{number(validation['raw_rms_waves'])} |")
    lm_table=["| 条件 | 方法 | 收敛 | nfev / njev | 抽样残差 RMS | 全图残差 RMS | 投影波前 J 条件数 |",
              "|---|---|---|---:|---:|---:|---:|"]
    for condition in ("clean","40db"):
        for method in ("single_lm","multi64_lm"):
            row,_=get(method,condition)
            lm_table.append(f"| {condition} | {NAMES[method]} | {row['success']} | "
                f"{row['nfev']} / {row['njev']} | {number(row['sampled_intensity_rmse'])} | "
                f"{number(row['full_intensity_rmse'])} | {number(row['projected_wavefront_condition_2'])} |")
    noisy={method:get(method,"40db")[0] for method in (*MAIN,"single_ft_matched")}
    source_ratio=noisy["single_lm"]["raw_rms_waves"]/noisy["multi64_lm"]["raw_rms_waves"]
    same_ft_ratio=noisy["single_ft_matched"]["raw_rms_waves"]/noisy["multi64_ft"]["raw_rms_waves"]
    pipeline_ratio=noisy["single_ft"]["raw_rms_waves"]/noisy["single_ft_matched"]["raw_rms_waves"]
    timing_table=["| 流程 | 标定实测 / s | clean 反演及回代 / s | 40 dB 反演及回代 / s |",
                  "|---|---:|---:|---:|"]
    for method in (*MAIN,"single_ft_matched"):
        kind,prefix=FILES[method]
        cal=(read(RESULTS/kind/f"{'original' if method=='single_ft' else 'matched'}_calibration.json")
             if "ft" in method else None)
        timing_table.append(f"| {NAMES[method]} | "
            f"{number(cal['calibration_wall_seconds']) if cal else '不需要'} | "
            f"{number(get(method,'clean')[0]['seconds'])} | {number(get(method,'40db')[0]['seconds'])} |")
    memory={kind:read(RESULTS/kind/"completed.json")["peak_rss_gib"] for kind in ("single","multi64")}
    text=f"""# 新泰伯基准下的 FT 与 LM 重跑报告

报告生成时间：{datetime.now(timezone.utc).isoformat()}。基准提交：`{REFERENCE_COMMIT}`。
全部结果来自 SJTU 本次独立重跑。原 1200/2400 网格、OFF 配置的旧报告保持原样，旧数据没有混入新表。

**主要结论：**40 dB 下，单源与 64 源 LM 的 raw RMS 分别为
{noisy['single_lm']['raw_rms_waves']:.7g}、{noisy['multi64_lm']['raw_rms_waves']:.7g} waves，
差别约 {abs(source_ratio-1)*100:.3g}%。这一单次噪声样本没有建立多源 LM 的普遍优势。
相同 OLS FT 流程下，单源和 64 源误差分别为
{noisy['single_ft_matched']['raw_rms_waves']:.7g}、{noisy['multi64_ft']['raw_rms_waves']:.7g} waves。
原单源 Lasso FT 标定的训练自检失败，单独标记为流程失效。

## 主要结果

真值为原第 7、25、11、15 项的组合，按新网格完整物镜光瞳缩放到 **0.05 waves raw RMS**。
以下误差保留倾斜、离焦和误差均值；piston 固定为零。所有 35 个非 piston 模式自由。
1 wave = 6.7 nm。表中“系数 RMS”只对 35 个非 piston 系数求均方根。

{chr(10).join(main_table)}

40 dB 下，单源 LM 与 64 源 LM 的 raw RMS 比为 {source_ratio:.5g}。
该比值只描述本次给定真值和 seed=42 的结果；方向以表中实际数值为准。
单源 OLS FT 是附加控制组，与 64 源 FT 使用相同提取、参考扣除和标定流程。
原单源 Lasso FT 与 OLS FT 的差异单独保留，避免将回归流程差异全部归因于光源数。

**光源数对照：**相同 OLS FT 流程下，单源／64 源误差比为 {same_ft_ratio:.5g}；
相同完整模型 LM 下，该比值为 {source_ratio:.5g}。大于 1 表示此次 64 源误差更低，小于 1 表示单源更低。

**FT 流程对照：**同一单源观测下，原流程／控制组误差比为 {pipeline_ratio:.5g}。
控制组同时使用参考复信号扣除、OLS 和 0.02 waves 标定步长；
原流程不在展开前扣参考，使用 Lasso 和 0.005 waves 步长，因此不能把差异只归因于回归器。

![40 dB 真值、四组主重建与误差，共用色标](wavefronts_40db.png)

![无噪声真值、四组主重建与误差，共用色标](wavefronts_clean.png)

主图第一行共用波前色标，第二行共用误差色标。所有面板只裁切显示范围，评价仍覆盖整个物镜光瞳。
LM 误差另给放大图，**采用独立的、更窄色标**。

![40 dB LM 误差独立放大色标](lm_error_zoom_40db.png)

## 已统一的物理配置与观测

| 项目 | 本次设置 |
|---|---|
| 光栅 | 未旋转 Square 方孔；0.670 µm 周期；25% 面积透过率 |
| 波长 / 光栅采样 | 6.7 nm / 16.75 nm；40 点周期、20 点孔宽 |
| 误差 | ON；物面轴向偏移 −100 µm；其余失调为零 |
| 名义像面到光栅 | 130.335839599 µm |
| 近轴实际焦面到光栅 | 134.335839599 µm；轴向焦移约 −4 µm |
| 光栅到相机 | 53.6 mm |
| 等效传播距离 / 第一完整泰伯长度 | 134 µm / 134 µm |
| 相机 / 光栅网格 | 1680² / 3360² |
| 单源 | 新索引 (7,7)，物面 (+1.675,+1.675) µm，对应旧 (5,5) |
| 64 源 | 每轴索引 3…10；物理位置仍为每轴 −11.725…+11.725 µm |
| 评价光瞳 | NA 0.00769…0.02；新离散物镜光瞳 {preflight['pupil_pixels']} 点 |
| 噪声 | clean 固定按自身无噪声峰值归一化；sigma=.01×RMS(clean)；seed=42 |
| 单源 sigma / 64 源 sigma | {single['noise_sigma']:.10g} / {multi['noise_sigma']:.10g} |
| LM | 真正的 SciPy LM；解析强度 Jacobian；零波前初值；2048 个固定随机像素 |
| LM 其他未知量 | 正曝光 gain 与常数背景；无波前正则化、无真值初始化 |

单源两种算法读取同一观测；单源 OLS 控制组也读取该观测。64 源 FT 与 LM 读取同一观测。
两种光源数的观测本身不同；本比较固定定义的图像 RMS 信噪比，没有固定总光子数。
仿真包含基准模型规定的角谱传播和失调。近轴泰伯等式不等于证明整个角谱模型在任意 NA 下精确自成像。

## 侧带重新定位：必须区分衍射级次

在完整泰伯新配置下，旧的 `N*p*dx/(lambda*T)` 公式给出约 31.265 bin，不能沿用。
将光栅 x 向 Fourier 级次 **0、+1、−1** 单独送入原 CZT 传播，得到独立数值验证：

| 干涉级次 | 几何载频 / bin | 直接隔离场的实测频谱峰 / bin |
|---|---:|---:|
| 0 与 +1 | 62.686567 | 63 |
| −1 与 +1 | 125.373134 | 125 |

原 FT 的差分基是正负剪切，因此与 **−1/+1** 交叉项对应。本次选择 **125 bin**。
设实际焦点到光栅为 R、光栅到相机为 D，球面二次相位近似给出：

`f_(m,n) = (m-n) * N * dx_cam * R / (p * (R+D))`。

正负一级的空间差分仍按两个副本的 ±shear 构造；每个副本位移约 `lambda*(R+D)/p/dx_cam`，
沿用原离散整数剪切规则，实际整数值见各标定 JSON。新基和掩膜均重建。
窗口半宽预先按相邻谐波间距的 40% 向下取整，实际为 **{geometry['half_width_bins']} bin**；
窗口平滑 sigma=2、相位平滑 sigma=10，保持原提取流程。没有按真值误差调窗。
零波前频谱中的强 63 bin 峰属于相邻级次，不将其误当成正负一级的频谱位置。

31 bin 的早期诊断已移入 `diagnostics/legacy-carrier-31-single/`，不参与任何正式表格或结论。
直接级次验证保存在 `results/diffraction_order_validation.json/.npz`。

![零波前频谱、窗口及轴向谱线](sideband_diagnostics.png)

## 独立标定与可辨识性

每个标定由零波前及全部 35 个单模式正负扰动组成，共 71 张独立训练图。
原单源 FT 保留 Lasso alpha=0.02、1000 次迭代上限、tol=1e-4，以及 0.005 waves 标定步长和 rcond=1e-10。
OLS FT 使用独立零波前复信号扣除、0.02 waves 正负扰动及 NumPy 默认 SVD 截断。
每个流程均重新生成响应矩阵，没有复用旧响应、旧侧带或旧剪切矩阵。

{chr(10).join(calibration_table)}

“—”表示按声明阈值不满秩，因此不报告有限的全秩条件数。原始 35 个奇异值、截断阈值、
保留子空间条件数以及训练自检误差都保存在对应 `*_calibration.json`。
秩满不保证抗噪稳定；小标定步长下的局部线性映射也不保证能准确反演更大混合波前。
原单源 R 由 Lasso、相位展开和平滑后的特征形成，其秩不是完整物理强度模型的可辨识秩。
Lasso 阈值与展开分支都可能让“单一线性响应矩阵”失效。

{chr(10).join(training_table)}

训练回解误差很大时，即使该响应矩阵具有较多非零奇异值，也不能将它称为有效的定量波前标定。
本报告保留这些失败结果；不将其解释为已达到的仪器精度。

![新响应矩阵的全部归一化奇异值](calibration_singular_values.png)

另外用 seed=1807 生成未参与标定的三种混合波前，检查标定在不同幅度下的外推误差：

{chr(10).join(holdout_table)}

这些控制波前不用于选择窗口、调正则化、初始化或修改主实验真值。

## LM 收敛与光强残差

{chr(10).join(lm_table)}

LM 波前 35 列的奇异值先投影去除曝光及背景两个 nuisance 列，再计算秩与条件数。
每次拟合从零波前开始。预算为先 80 次函数评价；只在未收敛时从同一优化轨迹续算最多 160 次。
`success` 只表示满足数值停止条件。波前是否准确须同时查看 raw RMS 和系数误差。
完整轨迹记录每个候选点，包含可能被拒绝的试探步；不将轨迹行数等同于接受迭代数。
FT 回代强度采用固定 clean 尺度，未额外拟合曝光和背景，因此其光强残差不能与 LM 残差视为完全相同的拟合目标。

![全部 LM 候选点评价的残差轨迹](lm_convergence.png)

## 计算开销

{chr(10).join(timing_table)}

标定时间为训练图传播与特征提取的实测累计值；独立验证、建模及报告绘图不包含在该列。
单源计算进程峰值 RSS 为 {memory['single']:.4g} GiB，64 源为 {memory['multi64']:.4g} GiB。
所有性能数字来自本次服务器实测，不从单源耗时简单外推多源耗时。

## 径向误差与系数误差

径向曲线在同一环形光瞳内按归一化半径等分 10 桶。每桶 RMS 是该桶全部有效离散像素的 raw RMS。
每桶边界和像素数保存在每个结果 JSON 的 `radial` 字段，便于重算。
外圈指标固定为 rho≥0.9。曲线和主表均保留全部非 piston 模式。

![全光瞳径向误差，含同流程 OLS 控制组](radial_errors.png)

![35 项系数误差](coefficient_errors.png)

## 数值核验与复现

原参考类的数值代码保持不变。为了在没有 PyTorch 的私人 Conda 环境读取这些 NumPy/SciPy 类，
验证适配器仅移除未参与计算的 torch 导入及 Module 基类；此修改仅存在于内存中的 AST。
正式快速算子是同一个离散 Fresnel/CZT 算子的可分离组合，未降低传播网格或减少物理源数。

- 单源零波前强度与原参考的相对 L2 误差：{preflight['errors']['zero']['errors']['all_sources_intensity_relative_l2']:.6g}。
- 单源真值强度与原参考的相对 L2 误差：{preflight['errors']['truth']['errors']['all_sources_intensity_relative_l2']:.6g}。
- 64 源逐源原参考叠加相对 L2 误差：{preflight['errors']['multi64_truth']['errors']['all_sources_intensity_relative_l2']:.6g}。
- 单源解析方向导数与中心差分相对误差：{preflight['errors']['jacobian_directional_relative_l2']:.6g}。
- 抽样传播与完整图像取同一像素的相对误差：{preflight['errors']['sampled_vs_full_relative_l2']:.6g}。
- 保存数据的独立审计：**{len(audit['checks'])} 项检查通过，10 条结果齐全**。

审计逐项复核输入 SHA256、物理源索引、噪声公式和 seed、同观测哈希、piston=0、
真值 RMS=0.05、raw RMS/PV、全部环带 RMS/点数、系数误差、完整图像残差及 FT 标定回解。
所有保存的波前数组为 float64；图像显示裁切不影响计算。

SJTU 工作目录：`/data/yk/shear-reference-lm/talbot-rerun-20261109`。
私人 Conda：`/data/yk/conda/envs/shear-reference-lm-py312`。
一个数值进程串行执行、最多 4 个 BLAS 线程，不使用 GPU。
30 秒日志位于 `results/eta.jsonl`；`results/status.json` 给出当前阶段。
标定 ETA 只外推当前标定剩余工作，LM 未知迭代数不伪造准确完成时间。

复查命令：

```bash
cd /data/yk/shear-reference-lm/talbot-rerun-20261109
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONPATH="$PWD/code"
/data/yk/conda/envs/shear-reference-lm-py312/bin/python code/audit.py
/data/yk/conda/envs/shear-reference-lm-py312/bin/python code/make_report.py
```

## 解释范围

这是一个给定真值、固定 seed 的模型匹配仿真比较。未建立多次噪声试验的成功率或真实仪器精度。
无噪声数值零只用于验证离散模型一致性；不代表皮米级仪器测量能力。
当前比较没有做高 NA 扫描、波前正则化寻优或按真值误差选侧带。
没有额外做更细网格的收敛性试验；误差数值是指定离散模型内部的重建误差。
本报告优先报告实际误差和病态方向；不通过去掉倾斜、离焦或只拟合四个真值项改善结果。

## 文件索引

- `report/metrics.csv`：全部主指标；`report/coefficients.csv`：全部 36 项系数及误差。
- `results/single/`、`results/multi64/`：观测、响应、训练/独立验证、系数、波前和优化轨迹。
- `results/preflight.json`：与原三段传播逐段核对及性能测量。
- `results/audit.json`：独立审计；`results/summary.json`：机器可读结果汇总。
- `provenance/input-manifest.json`：基准提交与部署文件 SHA256。
- `report/*.png`、`report/*.pdf`：科学图件；本目录内图件均由已保存结果生成。
"""
    (REPORT/"report.md").write_text(text)


def main():
    if not read(RESULTS/"audit.json").get("passed"):
        raise RuntimeError("Audit must pass before generating the final report")
    REPORT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    for condition in ("clean","40db"):
        wavefronts(condition)
    radial_and_coefficients()
    calibration_and_spectra()
    convergence()
    write_report()
    files={str(path.relative_to(ROOT)):dict(bytes=path.stat().st_size,
        sha256=sha256(path.read_bytes()).hexdigest()) for path in REPORT.iterdir() if path.is_file()}
    save(REPORT/"artifact-manifest.json",files)
    print(f"Report generated: {REPORT/'report.md'}",flush=True)


if __name__ == "__main__":
    main()
