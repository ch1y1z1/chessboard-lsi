"""Build a visual-summary Markdown draft from the final audited numbers."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"results"
REPORT=ROOT/"report"


def read(path):
    return json.loads(path.read_text())


def main():
    audit=read(RESULTS/"audit.json")
    if not audit["passed"]:
        raise RuntimeError("Refusing to summarize unaudited results")
    summary=read(RESULTS/"summary.json")
    rows={(r["method"],r["condition"]):r for r in summary["results"]}
    names=dict(single_ft="单源 FT：原 Lasso 流程",single_ft_matched="单源 FT：同流程 OLS 控制",
               single_lm="单源 LM",multi64_ft="64 源 FT：OLS",multi64_lm="64 源 LM")
    table=["| 方法 | clean RMS / waves | 40 dB RMS / waves | 40 dB PV / waves |",
           "|---|---:|---:|---:|"]
    for method in ("single_ft","single_lm","multi64_ft","multi64_lm","single_ft_matched"):
        clean=rows[method,"clean"]; noisy=rows[method,"40db"]
        table.append(f"| {names[method]} | {clean['raw_rms_waves']:.6g} | "
                     f"{noisy['raw_rms_waves']:.6g} | {noisy['raw_pv_waves']:.6g} |")
    rank_table=["| FT 流程 | 标定秩 | 全秩条件数 |",
                "|---|---:|---:|"]
    for kind,tag,method in (("single","original","single_ft"),("single","matched","single_ft_matched"),
                            ("multi64","matched","multi64_ft")):
        calibration=read(RESULTS/kind/f"{tag}_calibration.json")
        condition=calibration["full_rank_condition_2"]
        rank_table.append(f"| {names[method]} | {calibration['effective_rank']}/35 | "
                          f"{condition:.5g} |" if condition is not None else
                          f"| {names[method]} | {calibration['effective_rank']}/35 | 秩亏 |")
    lm_table=["| LM | 条件 | 收敛 | nfev | J 条件数 |","|---|---|---|---:|---:|"]
    for method in ("single_lm","multi64_lm"):
        for condition in ("clean","40db"):
            r=rows[method,condition]
            lm_table.append(f"| {names[method]} | {condition} | {r['success']} | {r['nfev']} | "
                            f"{r['projected_wavefront_condition_2']:.5g} |")
    preflight=read(RESULTS/"preflight.json")
    ft_ratio=rows["single_ft_matched","40db"]["raw_rms_waves"]/rows["multi64_ft","40db"]["raw_rms_waves"]
    lm_ratio=rows["single_lm","40db"]["raw_rms_waves"]/rows["multi64_lm","40db"]["raw_rms_waves"]
    text=f"""---
title: 新泰伯基准：FT 与 LM 重跑结果
subtitle: 0.670 µm · ON · 1680 / 3360 · clean 与 40 dB
template: doc
theme: paper
lang: zh
style: off
---

全部 10 条结果已完成，并通过 {len(audit['checks'])} 项保存数据审计。

## A 主结果

真值为原四项组合，缩放到 0.05 waves RMS。
35 个非 piston 模式全部自由，piston 固定为零。
误差保留均值、倾斜和离焦。1 wave = 6.7 nm。

{chr(10).join(table)}

## B 如何解读

```callout warn 原单源 FT 标定失效
原 Lasso 流程的训练回解误差很大。
这组结果不能视为有效的定量波前标定。
该响应矩阵的秩也不等于完整强度模型的可辨识秩。
```

同流程 OLS FT 的单源／64 源误差比为 {ft_ratio:.5g}。
同模型 LM 的单源／64 源误差比为 {lm_ratio:.5g}。
比值大于 1，表示本次 64 源误差更低。
比值小于 1，表示本次单源误差更低。

原 FT 与控制组同时改变了参考扣除、回归和标定步长。
不能把两者差异只归因于回归器。
LM 的 clean 数值零只验证模型内部一致性。

## C 统一配置

```kv
基准提交: 71ab81b
波长: 6.7 nm
光栅: Square 方孔，25% 面积透过率
周期 / 采样: 0.670 µm / 16.75 nm
误差: ON，物面轴向偏移 −100 µm
名义光栅距离: 130.335839599 µm
光栅到相机: 53.6 mm
传播网格: 相机 1680²，光栅 3360²
物镜光瞳: NA 0.00769–0.02
物理源数: 1 或 64，原位置保持
```

每种源数分别生成 clean 和 40 dB 观测。
噪声 sigma=.01×RMS(clean/clean.max())，seed=42。
同一源数下，FT 与 LM 共用同一观测。
LM 从零初值拟合 2048 个固定随机像素。

## D 侧带已独立核验

| 干涉级次 | 推导载频 / bin | 原 CZT 隔离场实测峰 / bin |
|---|---:|---:|
| 0 与 +1 | 62.6866 | 63 |
| −1 与 +1 | 125.373 | 125 |

正负剪切基对应 −1/+1，正式侧带取 125 bin。
窗口半宽为 25 bin，按相邻谐波间距预先确定。
旧 44、22 bin 和早期 31 bin 诊断均未用于正式结果。
没有按真值误差调窗。

![零波前频谱与正式窗口]({(REPORT/'sideband_diagnostics.png').resolve()})

## E 重建与径向误差

真值与重建共用色标，误差共用另一组色标。
显示范围裁到光瞳，评价仍覆盖完整光瞳。

![40 dB 四组主方法与真值对照]({(REPORT/'wavefronts_40db.png').resolve()})

LM 误差放大图使用独立色标。

![LM 误差放大，色标与主图不同]({(REPORT/'lm_error_zoom_40db.png').resolve()})

径向误差按同一光瞳等分 10 个半径区间。
各区间的有效点数保存在结果 JSON。

![同光瞳径向误差]({(REPORT/'radial_errors.png').resolve()})

## F 标定与收敛

{chr(10).join(rank_table)}

每套标定使用零波前及 35 项正负扰动，共 71 图。
另外生成三种独立混合波前，检查标定外推误差。

{chr(10).join(lm_table)}

J 条件数先投影去除曝光和背景两列。
收敛标记不保证恢复真值，须结合波前误差判断。

## G 数值核验与交付

```kv
保存结果审计: {len(audit['checks'])} 项通过，10 条结果齐全
64 源强度对原参考相对误差: {preflight['errors']['multi64_truth']['errors']['all_sources_intensity_relative_l2']:.6g}
解析导数对中心差分相对误差: {preflight['errors']['jacobian_directional_relative_l2']:.6g}
传播网格: 全程保留 1680 / 3360
数值格式: float64 / complex128
执行位置: SJTU 私人 Conda，串行 CPU，最多 4 线程
完整报告: report.md
数值表: metrics.csv 与 coefficients.csv
```

审计复核源位置、噪声公式、观测哈希、raw RMS/PV、
系数误差、环带点数和标定回解。
完整报告另含全部奇异值、训练自检及收敛轨迹。

这是固定真值和 seed=42 的模型匹配仿真。
本次没有建立一般成功率，也没有网格加密收敛结论。
固定的是图像 RMS 信噪比，没有固定总光子数。
不能把数值拟合零当作真实仪器测量精度。
"""
    path=REPORT/"overview.md"
    path.write_text(text)
    print(path)


if __name__=="__main__":
    main()
