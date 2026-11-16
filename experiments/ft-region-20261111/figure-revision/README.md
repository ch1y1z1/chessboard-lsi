# FT 选区报告图表返修，2026-11-16

用户反馈：流程与数据图有重叠、画幅不常见、图意不清楚、正文缺少坐标解释。本轮只重绘保存结果，不重跑数值实验。

`redraw.py` 输出 14 张正文图和一张 PDF 补图。PNG 固定 2400 px 宽，画幅 16:9 或 4:3；禁止 tight 裁切改变比例。统一蓝色圆点为单源、橙色菱形为 64 源（掩膜几何与带宽诊断另设图例）。图例放在数据区外。标题给出该图的具体结论，副标题给出真值与噪声条件。热图改为五种候选设置的配对比值均值、完整范围和改善计数，保留所有未画配置的原数据。

脚本对全部 100 个配置的七项统计共 700 项作一致性检查；只由四个保存系数向量重建空间误差图。文字裁切、文字包围盒相交情况写入 layout-audit.json。最终还需逐张查看 900 px 预览；自动检查不能证明读者理解，也不检测所有图形和曲线遮挡。

```bash
research/.venv/bin/python research/repo/experiments/ft-region-20261111/figure-revision/redraw.py \
  --original-output research/output --source research/source \
  --output research/revision-20261116 \
  --font research/assets/NotoSansCJKsc-Regular.otf
python research/repo/experiments/ft-region-20261111/figure-revision/prepare.py \
  --output research/revision-20261116
node /home/chiyizi/.agents/skills/answer-me-with-html/scripts/am.mjs render \
  research/revision-20261116/overview.md --no-open
python research/repo/experiments/ft-region-20261111/figure-revision/publish.py \
  --output research/revision-20261116
```

`publish.py` 原位修改原报告的十个图片块，插入四张拆分图和十四条读图说明，更新简版 HTML；保留旧表格、正文和块 ID。11 月 13 日日报只更新两张图并补充返修说明，保留用户后来新增的内容。当天日报按项目与日期去重后创建或合并，用户手记块不修改、不移动。时间线仅关联已有条目，不改用户要求或完成状态。

所有图片、HTML 与报告源稿写入后读回，下载校验 SHA256。旧交付留作历史快照，新旧版本与原扫描版本分别记录，不移动原 tag。
