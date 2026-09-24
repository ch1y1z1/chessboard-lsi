# 棋盘光栅二维剪切干涉：最小可复现实验

刘志祥博士论文《二维剪切干涉检测技术研究》（电子科技大学，2018）核心算法
的最小 Python 实现：45° 旋转棋盘光栅剪切干涉仪的前向光强模型，以及三条
波前反演路线。论文全文见 `references/二维剪切干涉检测技术研究_刘志祥.pdf`。

## 计算链条

```
干涉强度 I(x,y)
    ├─ 相移模式：N 步最小二乘解调   psi = atan2(-S, C)      （论文 2.3）
    ├─ 傅里叶模式：二维 FFT 提取 +f0 载频瓣 -> arg c(x,y)   （论文 2.4）
    └─ LM 直接反演：min ||I - I(c)||^2（不经解调/解包裹）
            ↓（前两条路线）
差分波前 dW_x = W(x+s,y)-W(x-s,y)，dW_y 同理
            ↓
差分 Zernike 最小二乘：c = (ΔZ^T ΔZ)^{-1} ΔZ^T ΔW           （式 2-28~2-31）
            ↓
波前 Zernike 系数（waves）
```

## 结构

```
lsi/
├── config.py       # Grid + SystemConfig（波长/NA/光栅周期 -> 剪切量 s、载频 f0）
├── grating.py      # 棋盘光栅衍射级次闭式振幅（表 2-3）
├── zernike.py      # Fringe/Wyant 序 Zernike 与双边差分基（式 2-25~2-27）
├── forward.py      # 前向模型：E = Σ A_ab exp(i[2πW + δ_ab + 载频])，I = |E|²
├── phaseshift.py   # N 步最小二乘相移解调 + 剪切区（式 2-20、图 2-8/2-9）
├── unwrap.py       # 掩膜内 Poisson 最小二乘解包裹
├── ftmode.py       # 单帧载频瓣解调（2.4 节）
├── reconstruct.py  # 差分 Zernike 最小二乘重构（式 2-28~2-31）
├── pipeline.py     # 端到端：解调 -> 解包裹 -> 重构
├── lm.py           # LM 直接光强反演：解析雅可比 + 增广最小二乘阻尼步
└── metrics.py      # PV/RMS/系数误差
experiment.py       # 三条路线端到端演示（输出到 output/）
test_lsi.py         # 端到端回归测试
```

## 前向模型

所有级次在探测器上相干叠加，光瞳重叠区自动出现，无需手工区域簿记：

```python
E(x,y) = Σ_ab A_ab · P(x+as, y+bs) · exp(i[2πW(x+as,y+bs) + δ_ab + 2πf0(ax+by)])
I(x,y) = |E|²
```

`test_lsi.py` 将该叠加模型与论文显式五光束公式 (2-14) 做逐点对照
（误差 ~1e-16）。

## 运行

```
uv sync                          # 安装依赖（numpy/scipy/matplotlib + pytest）
uv run pytest -q                 # 端到端回归测试
uv run python experiment.py      # 三条路线演示 + output/ 两张图
```

## 约定

- 光瞳归一化为单位圆，波前单位为 waves；
- Zernike 为 Fringe/Wyant 排序（Z1=平移、Z4=离焦、Z7=彗差）；
- 差分一律为双边 `W(x+s)-W(x-s)`；
- 理想棋盘光栅（50% 占空比，5 光束），标量模型，`0 < NA ≤ 1`；
- 演示波前 ≤ ~2 波长：更大像差会使调制度在剪切区内变号，
  解调路线失效（论文 2.3.2）。
