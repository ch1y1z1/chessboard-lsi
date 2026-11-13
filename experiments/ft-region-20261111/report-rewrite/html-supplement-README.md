# 简版 HTML 补充

日期：2026-11-13。用户要求在已有实验报告末尾附上简版 HTML。无新数值实验。

`overview.md` 是 answer-me-with-html 的扩展 Markdown 源稿。它只使用已核验的 report-rewrite-04 数据及图表；主真值窗宽数值来自 configuration-summary.csv，FT/LM 数据来自最终报告。页面包含六个面板、两个原生数据图和一张已有掩膜图。

制作命令：

```bash
node /home/chiyizi/.agents/skills/answer-me-with-html/scripts/am.mjs render \
  research/output/html-supplement/overview.md --no-open
```

CLI 版本为 0.5.7，实际 CLI 文件 SHA256、生成命令、源稿/HTML/图表 SHA256 另记在随附件保存的 versions.json。源稿图片路径为制作机器上的绝对路径；在其他机器复现时，改为该机器上的 `05_mask_erosion.png` 路径。该图由 report-rewrite-04 的 analyze.py 生成。

`html_supplement.py` 通过 ntn 追加三个报告块，并在日报「用户手记」前插入一条说明。运行前读取页面，保留原始块快照；运行后检查全部旧块、十张原生图片和用户手记保持不变。传输错误先读回，禁止盲目重试。不要运行 notion_sync.py 的整页替换入口。

```bash
python experiments/ft-region-20261111/report-rewrite/html_supplement.py \
  --output /absolute/path/to/research/output/html-supplement
```

输出目录需已有 `ft-region-overview.html` 与 `versions.json`。上传使用 workspace 实际文件限制和 single_part 模式。下载附件后校验 SHA256；签名链接、页面快照和传输回执仅保存在归档中，不提交 Git。

本次增量归档与原 delivery-rewrite 归档分别保留，原实验版本和标签不移动。
