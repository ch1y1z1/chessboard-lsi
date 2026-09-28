# 棋盘光栅二维剪切干涉：最小可复现实验

刘志祥博士论文《二维剪切干涉检测技术研究》（电子科技大学，2018）核心算法
的最小 Python 实现：45° 旋转棋盘光栅剪切干涉仪的前向光强模型，以及三条
波前反演路线。论文全文见 `references/二维剪切干涉检测技术研究_刘志祥.pdf`。

## 计算链条

```
干涉强度 I(x,y)
    ├─ 相移模式：N 步闭式解调   psi = atan2(-S, C)          （论文 2.3）
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

三个模块按计算链条组织，每个文件对应读者脑中的一个阶段：

```
lsi/
├── model.py        # 前向物理：网格/系统参数 -> 光栅级次 -> Zernike 基 -> I=|E|²
├── invert.py       # 论文反演链：解调 -> 解包裹 -> 差分 Zernike 最小二乘
└── lm.py           # LM 直接光强反演：解析雅可比 + 增广最小二乘阻尼步
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

`experiment.py` 第 5 节还比较单帧无相移、无载频的 LM 反演：沿用演示真值，
拟合 Z2..Z13，使用全图像素。比较零初值与 10 个随机方向在
0.03、0.1、0.3 waves 波前 RMS 下的 30 个非零初值（随机种子 42）。
控制台报告每次光强残差、允许整体变号的最大系数误差及各组恢复次数；
`output/lm_unmodulated.csv` 保存完整初值与拟合系数。初始化不使用真值，
最佳运行按光强残差选择；真值仅用于仿真评价。

## 约定

- 光瞳归一化为单位圆，波前单位为 waves；
- Zernike 为 Fringe/Wyant 排序（Z1=平移、Z4=离焦、Z7=彗差）；
- 差分一律为双边 `W(x+s)-W(x-s)`；
- 理想棋盘光栅（50% 占空比，5 光束），标量模型，`0 < NA ≤ 1`；
- 演示波前 ≤ ~2 波长：更大像差会使调制度在剪切区内变号，
  解调路线失效（论文 2.3.2）。

## 单帧无调制：七种优化器比较

`benchmark_optimizers.py` 将 **GD、Momentum、Adam、BFGS、GN、LM、TRF**
用于同一个单帧原始光强目标：`F = mean((I_model - I_obs)**2) / 2`。
输入不包含相移/载频，直接估计 Fringe Zernike 系数（waves），不经过解调。

```bash
# 小规模冒烟检查
uv run python benchmark_optimizers.py --config configs/optimizers-smoke.json

# 当前演示真值，128×128，零初值 + 30 个非零初值，共 217 次求解
uv run python benchmark_optimizers.py --config configs/optimizers-demo.json

# 随机稠密波前：100 个真值 × 30 个初值 × 7 种算法，共 21,000 次求解
uv run python benchmark_optimizers.py --config configs/optimizers-random.json

# 含噪声先导实验（稀疏波前、无噪声/40/20 dB）
uv run python benchmark_optimizers.py --config configs/optimizers-noise-pilot.json
```

每次默认创建带时间戳的 `output/optimizers-*/`，也可指定 `--output 新目录`，
已有目录会拒绝覆盖。请串行运行性能实验，避免多进程竞争 CPU 干扰计时。
程序默认将常见 BLAS 线程环境变量设为 1；启动前已有设置会保留并写入元数据。

输出包括：

- `dataset.npz`：全部观测、真值、共用初值及模式顺序；
- `metadata.json`：完整配置、随机种子、代码版本/工作区状态、依赖版本和线程环境；
- `source/`：实际运行的源码和依赖锁文件快照，SHA-256 写入元数据；
- `runs.csv`：每次求解的系数、误差、停止原因、实际模型调用数和时间；
- `trajectories.jsonl.gz`：每次实际光强评价的参数、损失、梯度信息及时间，包含拒绝步；
- `summary.csv` / `multistart.csv`：非零初值汇总及按观测损失选出的最终解；
- `report.md`、`comparison.png`、`budget_curves.csv/png`：报告及累计时间下的恢复表现。

只有 `metadata.json` 的 `state` 为 `complete` 才表示整个实验及报告已完成；
中断运行保留已写出的结果，不应与完整初值池的统计混用。

### 参数与实验口径

JSON 里的 `options` 是共用预算和容差，`method_options` 可以覆盖指定方法的
算法参数。例如以下配置比较两种方法；学习率示例仅用于说明配置方式：

```json
{
  "methods": ["adam", "lm"],
  "options": {"max_forward": 500, "max_jacobian": 500, "max_seconds": 5},
  "method_options": {"adam": {"learning_rate": 0.001}}
}
```

算法超参数应在独立验证集上选择并冻结；提供的配置只是可复现基线，未做调优。
`truth_seed`、`initial_seed`、`noise_seed` 相互独立。将 `truth_kind` 设为
`dense` / `sparse` / `single` 可生成稠密、三项稀疏或逐模态真值，
`indices` 控制拟合项；`demo` 使用历史固定真值，要求包含 Z4…Z8。
`noise_snr_db` 中 `null` 表示无噪声，其他数值采用
`20 log10(max(I_clean) / sigma)`；高斯噪声不裁剪负像素。

主实验内部使用 `q_j = std_pupil(Z_j) * c_j`，输出仍为原始 Fringe 系数；
`scaling: "none"` 可用于尺度消融。`initial_coordinates: "coefficients"`
配合默认种子复现历史初值池；统计实验使用 `"rms"` 在模态 RMS 坐标生成方向。
每个方向最终按实际光瞳波前标准差归一化，不使用真值决定初值。

- 零点 `J(0)=0`，零初值单独列为退化诊断，不计入主要恢复率。
- 当前实振幅模型存在 `I(c)=I(-c)`。误差只允许一个共同符号；倾斜的
  `1/s` 周期等价指标另列，不隐藏原始系数越界。
- 严格恢复要求无噪声、符号对齐后最大系数误差 `<1e-6 waves` 且光强 RMS
  残差 `<1e-10`；工程恢复默认波前 RMS 误差 `<0.01 waves`。停止不等于恢复。
- GD/GN 使用 Armijo 回溯；GN 用 SVD 最小二乘；LM 复用现有增广阻尼求步和
  Nielsen 更新。Momentum/Adam 用全像素梯度，无权重衰减。
- 各方法返回已评价的最低损失点。多初值也仅按观测损失选解，真值只用于评价。
- `max_forward` / `max_jacobian` 计真实计算，包括拒绝步与补算导数时的前向计算。
  `n_gradient` 计适配层的梯度计算，不包括 SciPy 或线性求解器内部乘法。
- `max_iter` 适用于 TRF 以外的方法。为兼容 SciPy 1.10，TRF 不依赖新版回调，
  迭代数留空，以实际模型调用和时间限制控制。
- 时间限制在模型调用边界检查；首次评价总会执行，单次不可中断计算造成的超时
  如实计入。共享预计算、求解与含预计算的单次端到端时间分别记录。
- 累计时间图按固定初值池顺序串行累加，在所有方法/场景有运行覆盖的公共时间
  区间比较。固定 30 初值的总成功率本身不代表等时间预算排名。

共享 API：

```python
from lsi.problem import IntensityProblem
from lsi.optimizers import SolverOptions, solve

problem = IntensityProblem(forward, indices, raw_image)
result = solve(problem, initial_coeffs, "lm", SolverOptions(max_forward=500))
# result.coeffs / result.status / result.history
```

第一版已覆盖同模型仿真、随机模态、初值、尺度和高斯噪声实验。
独立验证集自动调参、Poisson 噪声、模型失配、实测标定及统计置信区间属于后续扩展。
完整设计见 [实验方案](docs/single-frame-optimization-plan.md)。
已完成的 217 次历史案例比较及评价口径说明见
[第一轮结果](docs/optimizer-first-results.md)。

## 相移/载频采集模式的优化器比较

`benchmark_modulated.py` 把同一组优化器接到论文另两种采集协议的原始
观测上：相移模式（N 步 x + N 步 y 的干涉图序列）或单帧载频图。
各帧附加相位（光栅相移 delta / 空间载频）作为已知调制进入前向模型，
仍直接拟合光强、不经解调；目标为全部帧像素上的均方残差。

```bash
uv run python benchmark_modulated.py \
  --config configs/modulated-phaseshift-pilot.json
uv run python benchmark_modulated.py \
  --config configs/modulated-carrier-pilot.json
```

共享 API：

```python
from lsi.problem import StackedIntensityProblem
problem = StackedIntensityProblem(forward, indices, frames, modulations)
# frames: [I_1..I_K]；modulations: 逐帧 (n_orders,) 相移或 (n_orders,n,n) 空间相位
result = solve(problem, initial_coeffs, "gn", SolverOptions(max_forward=500))
```

已知调制消除了单帧无调制的两个退化：`J(0)=0` 不再成立，零初值成为
可行初值；载频下成功解的符号均为 +1（`I(c)=I(-c)` 被观测数据打破）。
一次 `n_forward` 对应整个采集堆叠的一次模型评价，两模式间及与单帧
实验的耗时不可直接等比。小规模结果见
[调制模式第一轮结果](docs/modulated-first-results.md)。
