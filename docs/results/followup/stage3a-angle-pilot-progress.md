# 阶段 3a 进度：camera x +0.25° 试点

- 时间：约 2026-10-21T15:00:53Z 起，持续 10.6 分钟（15:11:30Z 左右结束）；峰值 RSS 3261 MiB。
- 结果：9 行（zero direct + zero fixed/auto FT + camera known/ignored direct 与 fixed/auto FT）。

## 关键数值（仅此测试图）
- zero direct：`0.000932681` waves；zero FT fixed/auto：`0.000932681` / `0.000843969` waves。
- known direct：`6.30659e-18` waves；known FT fixed/auto：`0.00161449` / `0.000822876` waves。
- ignored direct：`0.0298212` waves，converged=False，nfev=60；ignored FT fixed/auto：`0.263317` / `0.26308` waves。

## 解释与限制
- known 几何下 direct 恢复数值零，FT auto 优于固定中心；ignored 几何下模型失配，direct 停在非零局部解，FT 误差很大。优化停止不等于恢复成功。
- 固定/自动中心分别使用相应几何的固定名义窗标定；未把物理角度当未知量，未用真值初始化。
- 参考倾斜仍是输入场相位坡度近似，不是严格旋转平面。
