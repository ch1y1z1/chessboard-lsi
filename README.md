# 棋盘光栅二维剪切干涉：最小实现

刘志祥博士论文《二维剪切干涉检测技术研究》（电子科技大学，2018）核心算法
的最小 Python 实现：45° 旋转棋盘光栅剪切干涉仪的前向光强模型，以及三条
波前反演路线。

## 计算链条

```
干涉强度 I(x,y)
    ├─ 相移模式：N 步最小二乘解调   psi = atan2(-S, C)
    ├─ 傅里叶模式：二维 FFT 提取 ±f0 载频瓣 -> arg c(x,y)
    └─ LM 直接反演：min ||I - I(c)||^2（不经解调/解包裹）
            ↓（前两条路线）
差分波前 dW_x = W(x+s,y)-W(x-s,y)，dW_y 同理
            ↓
差分 Zernike 最小二乘：c = (ΔZ^T ΔZ)^{-1} ΔZ^T ΔW
            ↓
波前 Zernike 系数（waves）
```

## 结构

```
lsi/
├── config.py       # Grid + SystemConfig（波长/NA/光栅周期 -> 剪切量 s、载频 f0）
├── grating.py      # 棋盘光栅衍射级次闭式振幅（表 2-3），占空比误差模型
├── zernike.py      # Fringe/Wyant 序 Zernike 与双边差分基（式 2-25~2-27）
├── forward.py      # 前向模型：E = Σ A_ab exp(i[2πW + δ_ab + 载频])，I = |E|²
├── phaseshift.py   # N 步最小二乘相移解调 + 剪切区判定（式 2-20、3-1~3-5）
├── unwrap.py       # 掩膜内 Poisson 最小二乘解包裹
├── ftmode.py       # 单帧载频瓣解调（2.4 节）
├── reconstruct.py  # 差分 Zernike 最小二乘重构（式 2-28~2-31）
├── pipeline.py     # 端到端：解调 -> 解包裹 -> 重构
├── lm.py           # LM 直接光强反演：解析雅可比 + 增广最小二乘阻尼步
├── metrics.py      # PV/RMS/系数误差
└── plotting.py     # 绘图辅助
scripts/            # 六个实验脚本（输出到 output/）
tests/test_core.py  # 最小回归测试
```

## 前向模型

所有级次在探测器上相干叠加，光瞳重叠区自动出现，无需手工区域簿记：

```python
E(x,y) = Σ_ab A_ab · P(x+as, y+bs) · exp(i[2πW(x+as,y+bs) + δ_ab + 2πf0(ax+by)])
I(x,y) = |E|²
```

`paper_region_intensity` 逐字转写论文显式 4/5 光束公式 (2-12)~(2-16)，
与叠加模型逐点误差 ~1e-16（scripts/01 验证）。

## LM 反演

直接拟合光强残差 `r(c) = I_model(c) - I_meas`，解析雅可比

```
dE/dc_j = Σ_ab A_ab e^{iφ_ab} · i2π Z_j(x+as, y+bs)
dI/dc_j = 2 Re(E* · dE/dc_j)
```

阻尼步用增广最小二乘（而非显式法方程）求解，Nielsen 增益比更新 λ；
可选变量投影同时估计光强增益与背景。Z1（平移）对光强不可观，不拟合。

## 运行

```
uv sync                          # 安装依赖（numpy/scipy/matplotlib + pytest）
uv run pytest -q                 # 最小测试集
uv run python scripts/run_all.py # 测试 + 全部实验脚本
```

| 脚本 | 内容 |
|---|---|
| 01 | 衍射级次（表 2-3）、叠加模型 vs 论文区域公式、相移律 |
| 02 | 相移路线端到端：区域判定、噪声敏感性、半条纹偏移、大像差失效 |
| 03 | 傅里叶路线：+f0 瓣 = 双边差分、窗半径敏感性、式 (2-48) 载频 |
| 04 | LM：相移帧/单帧反演、未知对比度、噪声、6 波长大像差 |
| 05 | 第四章误差源：占空比（含表 4-1）、步长误差、剪切误差、噪声 |
| 06 | 非 Zernike 自由曲面：36 模投影下限 vs 三条路线 |

## 约定

- 光瞳归一化为单位圆，波前单位为 waves；
- Zernike 为 Fringe/Wyant 排序（Z1=平移、Z4=离焦、Z7=彗差）；
- 差分一律为双边 `W(x+s)-W(x-s)`；
- 标量模型，`0 < NA ≤ 1`，不含高 NA 瞳畸变与矢量效应。
