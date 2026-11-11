"""Build Chinese report drafts from audited CSVs; no hand-written HTML."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess


def load(path):
    return list(csv.DictReader(Path(path).open()))


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--verification", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    primary = load(a.analysis / "primary_statistics.csv")
    regions = load(a.analysis / "regions.csv")
    geometry = load(a.analysis / "mask_geometry.csv")
    bins = load(a.analysis / "bin_sensitivity.csv")
    holdouts = load(a.analysis / "holdouts.csv")
    labels = {"single_ft": "单源 FT", "single_lm": "单源 LM",
              "multi64_ft": "64 源 FT", "multi64_lm": "64 源 LM"}
    nas = (.02, .066, .11, .15)

    def find(na, key, mask="geometric_outer"):
        return next(r for r in primary if float(r["na_out"]) == na
                    and r["pipeline"] == key and r["mask"] == mask)

    def value(row, key, factor=1, digits=2):
        return f'{float(row[key + "_mean"]) * factor:.{digits}f}'

    def pm(row, key, factor=1, digits=2):
        return value(row, key, factor, digits) + " ± " + f'{float(row[key + "_sd"]) * factor:.{digits}f}'

    def image(name, caption, directory=None):
        path = ((directory or a.analysis) / (name + ".png")).resolve()
        return f"![{caption}]({path})"

    main_rows = []
    for na in nas:
        for key in labels:
            row = find(na, key)
            main_rows.append([
                f"{na:.3f}", labels[key], pm(row, "full_rms", 1e4, 3),
                pm(row, "region_rms", 1e4, 3),
                value(row, "interior_rms", 1e4, 3),
                pm(row, "squared_error_fraction", 100),
                pm(row, "region_over_interior"),
                value(row, "trimmed_over_full", 100),
            ])
    main_table = table([
        "NA_out", "组合", "全瞳 RMS", "外侧 10% RMS", "内部 90% RMS",
        "外侧平方误差 (%)", "外侧/内部 RMS", "去外侧后/全瞳 (%)",
    ], main_rows)
    concise_table = table(
        ["NA_out", "单源 FT：外侧份额", "64 源 FT：外侧份额", "单源 LM：外侧份额", "64 源 LM：外侧份额"],
        [[f"{na:.3f}", *[value(find(na, key), "squared_error_fraction", 100) + "%"
                          for key in labels]] for na in nas],
    )
    # Keep header order identical to the explicitly listed data columns.
    concise_table = table(
        ["NA_out", "单源 FT", "单源 LM", "64 源 FT", "64 源 LM"],
        [[f"{na:.3f}", *[value(find(na, key), "squared_error_fraction", 100) + "%"
                          for key in labels]] for na in nas],
    )
    hot_rows = []
    for na in nas:
        geom = next(r for r in geometry if float(r["na_out"]) == na
                    and float(r["requested_fraction"]) == .1 and r["reference"] == "single_ft")
        hot_rows.append([
            f"{na:.3f}", value(find(na, "single_ft"), "squared_error_fraction", 100),
            value(find(na, "single_ft", "hot_clean_single_ft"), "squared_error_fraction", 100),
            value(find(na, "single_lm", "hot_clean_single_ft"), "squared_error_fraction", 100),
            f'{100 * float(geom["hot_edge_overlap"]):.2f}',
            f'{100 * float(geom["hot_outer20_fraction"]):.2f}',
        ])
    hot_table = table([
        "NA_out", "几何外侧：FT份额 (%)", "冻结热点：FT份额 (%)",
        "同一热点：LM份额 (%)", "热点落在外侧10% (%)", "热点落在外侧20% (%)",
    ], hot_rows)
    area_table = table(
        ["NA_out", "有效像素", "实际外侧比例 (%)", "外侧起点 ρ", "外侧起点 NA"],
        [[f"{na:.3f}", r["evaluation_pixels"], f'{100 * float(r["actual_fraction"]):.4f}',
          f'{float(r["rho_low"]):.6f}', f'{float(r["na_low"]):.6f}']
         for na in nas for r in geometry if float(r["na_out"]) == na
         and float(r["requested_fraction"]) == .1 and r["reference"] == "single_ft"],
    )
    sensitivity = []
    for fraction in (.01, .02, .05, .1, .15, .2, .3):
        means = []
        for na in nas:
            for key in ("single_ft", "multi64_ft"):
                group = [float(r["squared_error_fraction"]) for r in regions
                         if float(r["na_out"]) == na and r["pipeline"] == key
                         and r["mask"] == "geometric_outer"
                         and float(r["requested_fraction"]) == fraction
                         and r["condition"].startswith("40db")]
                means.append(statistics.mean(group))
        sensitivity.append([f"{fraction*100:g}%", f"{100*min(means):.2f}%—{100*max(means):.2f}%"])
    sensitivity_table = table(["几何外侧像素比例", "八组 FT 的五种子平均误差份额范围"], sensitivity)
    bin_table = table(
        ["分桶方式", "10 桶", "20 桶", "50 桶", "100 桶"],
        [[name, *[f'{1e4 * max(float(r["rms_mean"]) for r in bins if float(r["na_out"]) == .15 and r["pipeline"] == "single_ft" and r["coordinate"] == coordinate and int(r["bins"]) == count):.3f}'
                  for count in (10, 20, 50, 100)]]
         for coordinate, name in (("equal_radius", "等半径"), ("equal_area", "等面积"))],
    )
    holdout_table = table(
        ["NA_out", "单源 FT 留出：外侧份额 (%)", "64 源 FT 留出：外侧份额 (%)"],
        [[f"{na:.3f}", *[
            f'{100*min(float(r["squared_error_fraction"]) for r in holdouts if float(r["na_out"]) == na and r["pipeline"] == key):.2f}—{100*max(float(r["squared_error_fraction"]) for r in holdouts if float(r["na_out"]) == na and r["pipeline"] == key):.2f}'
            for key in ("single_ft", "multi64_ft")]] for na in nas],
    )
    params = table(
        ["NA_out", "NA_in/NA_out", "光栅周期 (µm)", "相机距离 (mm)", "源距 (µm)", "泰伯比"],
        [[f"{na:.3f}", f"{.00769/na:.5f}", f"{.67/(na/.02):.9f}",
          f"{53.6/(na/.02)**2:.9f}", f"{3.35/(na/.02):.9f}", "1"] for na in nas],
    )
    analysis_version = json.loads((a.analysis / "versions.json").read_text())
    verification = json.loads((a.verification / "independent_verification.json").read_text())
    repo = Path(__file__).resolve().parents[2]
    delivery_sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    version_rows = [
        ["chessboard-lsi：原反演", "753ab12789db71d6b57946edabcccdb20bc3d14b", "exp/na-talbot-balanced-20261111/run-08"],
        ["shear-reference-lm：原传播参考", "71ab81b18c9f017c6ddd3dfc302d18b0a2ba2542", "exp/na-talbot-balanced-20261111/run-08"],
        ["chessboard-lsi：本轮主分析", analysis_version["commit"], "exp/fixed-na-edge-20261111/analysis-01"],
        ["chessboard-lsi：独立复核与控制图", verification["commit"], "exp/fixed-na-edge-20261111/verify-01"],
        ["chessboard-lsi：报告生成与同步工具", delivery_sha, "exp/fixed-na-edge-20261111/delivery-01"],
    ]
    version_table = table(["仓库及用途", "完整 commit", "固定实验 tag"], version_rows)
    title = "固定 NA_out 的外缘误差：无分桶曲线、固定面积与高误差像素对照"
    report = f"""# {title}

实验编号：`fixed-na-edge-20261111`。运行：`analysis-01`、`verify-01`。原数据：`na-talbot-balanced-20261111/run-08`。本轮在 SJTU 读取已保存结果，未重新传播、标定或优化。正式分析耗时 {analysis_version['elapsed_seconds']:.2f} 秒，峰值 RSS {analysis_version['peak_rss_mib']:.1f} MiB，仅用两条 CPU 线程，不占 GPU。

## 先看结论

- **FT 的外缘误差集中是稳健现象，不依赖某一种径向分桶。** 最外侧约 10% 有效光瞳像素承担单源 FT 72.25%—75.56%、64 源 FT 71.36%—73.71% 的总平方误差。对照 LM 为 18.10%—21.19%。
- **固定像素比例比固定半径宽度更适合跨系统比较。** 旧的“最外 10% 半径”实际占光瞳约 19%—22%，并非本轮的“最外 10% 像素”。两组数值不能混用。
- **高误差像素分区更突出热点，但不能单独证明边缘效应。** 无噪声单源 FT 图选出的最高误差 10% 像素，在五个含噪结果中承担 81.19%—82.92% 平方误差；其中约 91%—93% 位于几何外侧 20%，约 60%—62% 位于几何外侧 10%。
- **边缘集中程度随系统 NA_out 轻度增加，边缘绝对 RMS 不单调。** 单源 FT 外侧误差份额增加 3.32 个百分点，64 源增加 2.35 个百分点；从 0.110 到 0.150，外侧 RMS 分别回升 8.89% 和 7.84%。
- **不能把这些关联归因于 NA_out 单因素。** 四档属于配平后的标量 Fresnel 系统族，遮拦比、基函数和真值同时改变。当前结果确认了空间误差分布，未识别其物理或算法根因。

## 当前研究进展与本轮衔接

项目比较 FT 解调与直接光强 LM 拟合。两条链使用同一参考三段传播；64 源逐源传播后非相干等权叠加强度。当前 Square 为 25% 面积开口的周期方孔阵列，不与 50% 棋盘光栅混称。

旧 NA 扫描曾同时存在泰伯、光栅范围、相机视场与采样混杂，不能把高 NA 的失败当作 NA 本身的证据。11 月 9 日确立 0.670 µm 一阶泰伯基准，11 月 10 日修正 FT 基频与相邻级次差分基匹配，11 月 11 日配平四档参数后完成 112 条恢复结果。LM 平稳，FT 保留无噪声偏差。上一版简报已展示累计口径与 50 个径向桶，但没有量化分桶敏感性。

本轮直接回应三个问题：固定系统后向外扩大评价口径，RMS 如何变化；固定高误差像素比例是否更清楚；边缘误差强度与集中程度是否随系统 NA_out 改变。读取了当前记忆、原完整报告、图表简报、代码留存规范及报告/日报参考页，不把旧记忆中的 Notion 认证故障当作当前状态。

## 固定的系统与评价口径

{params}

波长 6.7 nm、NA_in=0.00769、M=0.2；像面 1680²、光栅 16800²、相机 7560²；35 个非 piston 模式全部保留。真值为原四模式组合，在每档完整光瞳归一为 0.05 waves RMS。使用原始波前误差，不去均值、倾斜或离焦。40 dB 使用 seeds41—45 的均值和样本标准差，FT 与 LM 共用相同观测。

系统 NA_out 在一条曲线内固定。横轴 NA_eval 只是评价区域的截止 NA：选择 NA_in≤NA≤NA_eval 的原生光瞳像素，不改变孔径、不重标定、不重新拟合。不同面计算窗口外缘的光能检查，是截断风险诊断；本报告统计光瞳上的恢复误差，两者不是同一项指标。原 512 个逐源检查通过的记录继续适用，未在本轮重跑。

## 指标与预设分区

令误差 e_i=W_rec−W_true，有效光瞳为 P，外侧像素集合为 E_p。

$$
R(A)=\\sqrt{{\\frac{{1}}{{|A|}}\\sum_{{i\\in A}}e_i^2}},\\qquad
C(p)=\\frac{{\\sum_{{i\\in E_p}}e_i^2}}{{\\sum_{{i\\in P}}e_i^2}}.
$$

主分析预先选择外侧 p=10% 的有效像素，同时保存 1%、2%、5%、15%、20%、30% 敏感性分析。圆环几何下外侧阈值近似为 ρ_p=√[1−p(1−ε²)]，ε=NA_in/NA_out；实际计算保留完整同半径像素壳层，故比例略大于 10%。

{area_table}

几何分区只依赖半径，不看误差。热点分区按无噪声 FT 的 |e| 排序，选取与几何外侧完全相同数量的像素并冻结；单源与64源各有一份参考，均施加到四种恢复方法。含噪图自身最高误差分区只作描述性上包络，不能用来独立验证误差集中。每个像素的误差是重建的全局系数在该位置的表现，不等于当地 FT 相位提取误差。

## 固定系统：不依赖分桶的 RMS–NA

{image('unbinned_cumulative', '每个不同原生半径均计算一次累计 RMS；五种子均值与样本标准差，不分径向桶。')}

曲线在接近外缘时陡升，四档均有。累计 RMS 不必单调：加入一圈时，若该圈平均平方误差低于当前累计均值，RMS 会下降；高于当前均值才会上升。累计平方误差则必定不减。因此，同时给出累计 RMS 与由外向内的误差份额 C(p)，避免把曲线中部下降解释为高 NA 反而更准确。

{image('edge_concentration', '横轴为由外向内累积的有效像素比例；纵轴为这些像素承担的总平方误差。虚线是均匀像素误差基准。')}

{sensitivity_table}

最外约 1% 的像素已承担约 15%—17% 的 FT 平方误差，最外 5% 承担约 54%—57%。这不是选择 10% 才出现的现象。达到 80% 平方误差所需外侧像素：单源 FT 从 15.27% 降到 12.52%，64 源 FT 从 17.82% 降到 15.48%；LM 则约需 63%—74%。

## 四组合主结果

本表 RMS 单位为 **10⁻⁴ waves**。± 为五种子的样本标准差；份额和比值先逐种子计算，再取均值。主分区实际面积见上表。

{main_table}

{image('edge_erosion', '仅移除评价像素，不重新拟合；纵轴为剩余区域 RMS 除以完整光瞳 RMS。')}

去掉外侧约 10% 的评价像素后，FT 剩余 RMS 为原值的 52.11%—56.41%，即下降约 44%—48%；LM 剩余 RMS 为 93.47%—95.37%。这描述误差定位，不是裁掉数据后重新运行算法的精度收益，更不代表全光瞳恢复已被改进。

## 高误差像素分区是否更清楚

{image('maps_and_frozen_masks', '前两列为同一含噪观测的单源 FT/LM 绝对误差，seed42，共用对数色标；第三列为预先冻结热点与几何外侧10%的叠加。', a.verification)}

{hot_table}

{image('frozen_hot_concentration', '实线为几何外侧，虚线为冻结的无噪声单源 FT 热点，点线为每张含噪图自身排序；后者不是独立验证。')}

固定热点捕获的误差份额高于同样大小的几何圆环，说明角向结构不能忽略。单源 FT 热点中约 7%—9% 不在外侧20%，所以不能把所有高误差像素都叫作边缘。另一方面，约 91%—93% 落在外侧20%，远高于均匀随机选点的约20%，仍支持其主要位于外缘附近。

如果每张含噪图都用自身误差排序，任何方法都必然得到偏高的“最差10% RMS”。本数据中 LM 自身最高误差10%也捕获约42%—46%平方误差，但在同一冻结 FT 热点只捕获约17%—18%。因此，正确用法是先选定区域，再交叉评价并保留空间坐标。冻结无噪声掩膜只避免噪声实现层面的重复选区，它仍依赖同一主真值，不构成独立波前泛化验证。

## 分桶敏感性有多大

{image('bin_sensitivity', '同一单源 FT 数据改用10/20/50/100个等面积桶；细桶更突出最外侧局部峰。')}

以 NA_out=0.150 的单源 FT 为例，局部桶 RMS 的最大值如下，单位仍为10⁻⁴ waves：

{bin_table}

同一张图可以因桶数改变，使局部峰从约9.37变到17.22。等半径与等面积的末桶面积也不同，峰值不能直接比较。主报告不使用“末桶最高值”作跨系统指标，而使用固定面积的 C(p)、外侧/内部 RMS 和不分桶的累计曲线。完整数值已保存，两种分桶的像素数和平方误差总和均核对。

## 边缘效应与系统 NA_out 的关系

{image('system_na_effect', '跨系统使用同一外侧像素比例；误差条是五种子样本标准差。')}

从0.020到0.150，单源 FT 外侧份额由72.25%增到75.56%，增加3.32个百分点；配对差值的样本标准差为0.09个百分点。64源从71.36%到73.71%，增加2.35个百分点，配对差值标准差为0.09个百分点。该变化大于本条件下的噪声重复波动，但不代表其他真值或真实装置也有同样趋势。

外侧绝对 RMS 则不单调：单源为11.905、11.374、11.160、12.152；64源为10.148、9.329、9.113、9.827，单位10⁻⁴ waves。从0.110到0.150分别回升8.89%、7.84%；两端比较单源只增加2.07%，64源反而下降3.16%。边缘份额、边缘/内部比和边缘绝对 RMS回答不同问题，应同时报告。

NA_out 与孔径位置的两种变化不可混同。配平维持近似 Fresnel 相似关系，同时 NA_in固定导致遮拦比变化，基与物理真值也改变。四个点只给出该系统族内的关联，既不能估计单因素因果效应，也不足以定位高NA阈值。

## 真值幅度、基函数与噪声控制

{image('basis_and_relative_control', '左：均匀像素误差与独立等方差系数误差的基函数基准；右：按每区真值RMS归一后的外侧/内部比。', a.verification)}

35模式独立等方差系数误差的代数模型，在外侧10%已预期集中了25.26%—26.67%的平方误差。这说明“10%面积超过10%误差”本身不足以证明局部边缘失效；该基准不是拟合出的物理噪声模型，也不能从实测份额中直接相减求因果贡献。FT的71%—76%远高于这一对照；LM的18%—21%反而低于它。

真值本身在外缘幅度较大。按每个区域的真值 RMS 归一后，FT 外侧/内部相对误差比约2.30—2.50，弱于绝对 RMS 比4.73—5.28，但仍有外侧偏高。LM相对误差比约0.71—0.74，未显示同样的相对外缘恶化。相对归一只控制幅度尺度，不是机制识别。

{image('bias_and_holdouts', '左：五种子均值图与有限样本波动的精确MSE分解；右：24个已有随机真值FT留出结果。')}

FT 外侧 clean MSE 为40 dB平均MSE的99.70%—99.89%，五种子均值周围的波动仅占约0.005%—0.009%。因此主真值上的外侧结构主要表现为无噪声时就存在的恢复偏差，而不是由这些噪声种子随机产生。分解保留 clean 与噪声扰动的交叉项，不把MSE比机械解释成可相加的因果份额。LM的有限种子波动约占外侧总MSE的79%—82%，五次重复不足以精确估计总体噪声分布。

三个随机留出真值、两种源数、四档系统，共24条已有 clean FT结果：

{holdout_table}

这些留出结果的外侧10%仍捕获约56.56%—73.74%平方误差，支持外侧集中不只出现在主真值；但主真值72%—76%的具体数值不能推广到所有波前。留出数据没有LM反演，也没有新增独立物理前向模型。

## 验证与结论边界

主分析4264项核对通过，独立保存图复算5776项通过。复核包括原112条完整光瞳误差、24条留出、分区像素、MSE守恒、累计终点、误差份额单调性、冻结热点重建、五种子统计及精确偏差/波动分解。输入与代码保留SHA256。独立复核指另一条保存图统计路径，不是独立光学模型。

没有做传播/反演网格收敛、非近轴或矢量模型验证；像方NA最高0.75仍使用原标量Fresnel算子。没有固定总光子预算，不能推断多源曝光效率。没有隔离频谱取窗、平滑、剪切交集、线性标定和基函数方向误差，因此不能把外缘形状直接命名为Gibbs效应。本报告不把执行成功等同准确恢复，也不把像素数当作统计重复数。

下一步应固定系统和独立真值，逐项改变FT窗口、平滑、差分覆盖或标定方式，并用留出波前评价。如果要分离NA本身，还需独立控制遮拦比、归一化采样和真值定义，并先完成传播与反演网格收敛。本轮未自动启动这些额外计算。

## 代码版本与留存

{version_table}

每份主分析与复核代码均先推送标签、核对远端SHA，再在SJTU新目录检出执行。原反演两仓库标签保留，未移动或覆盖。报告工具与数值分析版本分开记录。

主仓库提交链接：https://github.com/ch1y1z1/chessboard-lsi/commit/{analysis_version['commit']} 。主分析标签：https://github.com/ch1y1z1/chessboard-lsi/tree/exp/fixed-na-edge-20261111/analysis-01 。复核标签：https://github.com/ch1y1z1/chessboard-lsi/tree/exp/fixed-na-edge-20261111/verify-01 。参考仓库原提交：https://github.com/ch1y1z1/shear-reference-lm/commit/71ab81b18c9f017c6ddd3dfc302d18b0a2ba2542 。各仓库完整链接与运行命令保存在versions.json。

服务器根目录：`/data/yk/shear-reference-lm/fixed-na-edge-20261111`。主分析输出`analysis-01`，复核输出`verification-01`；原31GiB相机数组仍在原实验run-08目录，未复制入Git。本轮重建图与所有壳层曲线另以NPZ保存，报告附件提供紧凑CSV/图表包；较大的完整数值NPZ保留本机及SJTU，不伪称已上传Notion。

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /data/yk/conda/envs/shear-reference-lm-py312/bin/python code-01/experiments/fixed-na-edge-20261111/analyze.py --results /data/yk/shear-reference-lm/na-talbot-balanced-20261111/run-08 --output analysis-01 --expected-commit {analysis_version['commit']}
```

## 补充材料

原完整报告：https://app.notion.com/p/NA-4158b24d37dc817f9ecdf26116d10207 。原图表简报：https://app.notion.com/p/NA-4158b24d37dc816e88d1c51714e8969b 。本轮包含8张主图、控制图、11份CSV、两份数值NPZ、输入哈希、分析与独立核验回执及固定版本代码。
"""
    (a.output / "report.md").write_text(report)
    overview = f"""---
title: 固定 NA_out：FT 外缘误差集中在哪里
subtitle: 固定像素比例、冻结热点与不分桶曲线
template: doc
theme: paper
lang: zh
---

## A 核心判断

FT 外侧约10%像素承担71%—76%平方误差。
边缘集中程度略随系统NA_out增加。
边缘绝对RMS没有单调增加。
四档属于配平系统族，不能单归因于NA。

{concise_table}

## B 比较的是什么

| 项目 | 定义 |
| --- | --- |
| 系统NA_out | 每条曲线内固定 |
| NA_eval | 只改变评价截止位置，不重新拟合 |
| 几何外侧 | 按半径选最外10%有效像素 |
| 冻结热点 | 按无噪声FT选10%像素，再评价含噪结果 |
| 统计重复 | 40dB五种子；像素不作独立重复 |

原112条结果与24条留出均复用。
本轮没有新传播或反演，不占GPU。

## C 不分桶的 RMS–NA

{image('unbinned_cumulative', '每个不同原生半径均计算累计RMS。阴影为五种子样本标准差。')}

FT纳入外缘后RMS陡升，四档均存在。
累计RMS可先下降，因为新增像素可能更准确。
这不等于改变物理NA后的算法表现。

## D 固定面积衡量边缘

{image('edge_concentration', '外侧像素比例与总平方误差份额，不依赖径向桶。')}

最外5%像素承担约54%—57%的FT平方误差。
去掉外侧10%评价像素，FT的RMS下降约44%—48%。
这只是误差定位，不是重新拟合后的精度提升。

## E 高误差像素更清楚，但需交叉评价

{image('frozen_hot_concentration', '虚线使用冻结的无噪声FT热点。点线按含噪图自身排序，只作描述。')}

冻结热点10%承担约81%—83%的单源FT平方误差。
约91%—93%热点位于几何外侧20%。
热点与几何外侧10%只重合约60%—62%。
误差有角向结构，并非均匀一圈。

## F 系统 NA_out 的关联

{image('system_na_effect', '相同外侧像素比例，五种子均值与样本标准差。')}

单源FT外侧份额增加3.32个百分点。
64源FT外侧份额增加2.35个百分点。
从0.110到0.150，外侧RMS回升8.89%和7.84%。
遮拦比、基和真值也改变，不能作因果归因。

## G 偏差与幅度控制

{image('basis_and_relative_control', '相对RMS按各区真值幅度归一。基函数曲线是代数对照。', a.verification)}

FT按局部真值幅度归一后，外侧仍高约2.3—2.5倍。
LM相同比值约0.71—0.74，没有同样的相对外缘恶化。
FT外缘误差主要在无噪声结果中就已存在。
24条随机真值留出仍显示外侧集中，但程度不同。

## H 验证、版本与边界

主分析4264项、独立图统计5776项核对通过。
主提交：`{analysis_version['commit']}`。
主标签：`exp/fixed-na-edge-20261111/analysis-01`。
复核标签：`exp/fixed-na-edge-20261111/verify-01`。

未验证网格收敛或严格非近轴高NA物理。
未识别Gibbs、窗口或标定中的具体原因。
完整分桶敏感性、四组合表和留出分析见配套报告。
"""
    (a.output / "overview.md").write_text(overview)
    manifest = {
        "experiment_id": "fixed-na-edge-20261111", "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_analysis": analysis_version, "independent_verification": verification,
        "delivery_commit": delivery_sha, "version_rows": version_rows,
        "document_review": {"skill": "humanizer-zh", "status": "draft; review before publication"},
    }
    (a.output / "versions.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
