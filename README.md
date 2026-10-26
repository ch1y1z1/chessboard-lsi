# chessboard-lsi

BUV0725 / Retrieval0806 参考衍射模型下的二维剪切干涉波前反演。
一套物理模型，两种解算器，比较、选区和通光角范围三项主要研究。系统参数、场景真值和反演设置独立保存。

## 代码

```text
lsi/
  config.py     系统参数 SystemConfig、场景 Scene、拟合预算 FitConfig
  basis.py      RMS 归一化环形 Zernike，沿用参考的零基编号
  forward.py    三段传播、全部源强度叠加、解析强度导数
  fourier.py    边带窗、掩膜、解包裹、差分拟合、独立响应标定
  direct.py     原始光强残差上的 SciPy TRF 非线性最小二乘
  evaluate.py   固定光瞳上的波前误差
  run.py        compare / regions / apertures；另保留部件倾角 angles 对照
  io.py         配置、观测、结果表、轨迹和源码快照
  cli.py        简短命令入口

tools/
  check_reference.py   与用户提供的原参考脚本逐段核对
  make_report.py       只读已保存结果，生成比较图

configs/        可直接编辑的实验 JSON
```

`forward.py` 接收显式系数，不读取真值。`Scene` 只由观测生成和评价使用。
默认模型保留全部 64 源、1200×1200 相机、2400×2400 光栅、环形光瞳及周期方孔。
50% 孔边长占空比对应 25% 开口面积。可分离传播与逐段 CZT 是相同离散算子的两种计算方式；生产默认使用前者，后者供数值核对。

## 运行

```bash
uv sync --locked
uv run python -m lsi --help

# 本机只做四源小场景检查；此场景不能用于原采样精度结论。
uv run python -m lsi compare --profile test --output output/smoke
```

命令只选工作流、profile、配置和输出目录；科学参数集中写在 JSON。
`--profile` 可以覆盖 JSON 中的 profile，用于检查相同实验配置的小场景执行路径。
以下原采样任务仅在内存充足的机器上串行执行：

```bash
uv run python -m lsi compare --config configs/compare.json --allow-large-run --output output/compare
uv run python -m lsi compare --config configs/noise.json --allow-large-run --output output/noise
uv run python -m lsi regions --config configs/regions.json --allow-large-run --output output/regions
uv run python -m lsi apertures --config configs/apertures.json --allow-large-run --output output/apertures
uv run python -m lsi angles --config configs/angles.json --allow-large-run --output output/angles
```

配置的六个部分分别是：

| 字段 | 内容 |
|---|---|
| `profile` / `system` | 命名采样场景，以及物理参数覆盖值 |
| `scene` | 真实模式 `indices`、系数 `coefficients`、目标 `rms`、`snr_db` 和噪声 `seed` |
| `fit` | 待估计模式 `indices`、`starts`、`samples`、`max_nfev`、`screen_initials`、初始化 `seed` |
| `fourier` | 频谱半宽、中心、形状、掩膜、平滑及回归设置，详见 `FourierSettings` |
| `calibration_amplitude` | 正负单模式独立标定的幅度，默认 0.02 waves |
| `sweep` | 当前工作流的扫描列表，示例见 `configs/` |

系数单位为 waves；`scene.rms=null` 保留输入系数原始幅度。噪声先对干净强度作峰值归一化，再加入标准差 `10**(-snr_db/20)` 的 Gaussian 噪声。同一带噪图分别交给两种方法。

`regions` 默认冻结名义标定；把 `sweep.recalibrate` 改成 `true` 才逐设置重标定。
`apertures` 改变外 NA 和内外 NA 比例，比较不同通光角范围；每个 NA 工况重新生成观测和独立标定，
使用该工况自身完整环形光瞳评价，并把输入系数方向归一化到同一目标 RMS。
它比较的是相同模式方向、相同 RMS 的不同光瞳问题，并非固定物理波前被不同孔径裁切。
`angles` 是另保留的部件安装倾角研究，与通光角范围不同；分别记录已知倾角和忽略倾角的结果。
两个工作流中的 fixed/auto 都各自独立标定，属于完整流程比较。
NA 范围、噪声口径、采样边界和完整实验设计见 [通光角研究协议](docs/aperture-study.md)。
直接法联合拟合波前、正增益和常数背景，保存所有初值，包括未收敛结果。
傅里叶 waves 结果使用独立零像差与正负单模式标定；原始相位拟合系数仍可通过 `FourierExtractor.extract` 读取，单位为 a.u.。

超过四源小网格必须显式指定 `--allow-large-run`；同一用户一次只运行一个任务。
输出目录必须为空，中断保存已完成行和状态。默认逐点保存轻量结果，不为每个扫描点重复保存完整频谱/场数组。

## 数据与图

```text
manifest.json                  实际配置、随机种子、代码散列、耗时和状态
results.json / results.csv     每个工况的系数、误差和优化状态
*_observation.npz              原始强度、同一带噪观测、真值和固定光瞳
*_calibration.npz / .json       标定响应、独立参考、训练图散列和条件数
*_validation.npz               未参与标定的混合波前验证
*_direct_trials.json           所有初值及其迭代轨迹
*_sample_rows.npy              直接法实际使用的像素
source_snapshot/               本次运行的源码和锁文件
```

绘图单独安装可选依赖，不运行前向传播或优化：

```bash
uv run --group report python -m tools.make_report output/smoke
```

主 RMSE/PV 在固定物镜环形光瞳内去除活塞、双向倾斜和离焦；另报保留倾斜/离焦的 RMSE 和全部系数误差。`converged` 是优化停止标记，`recovered` 是声明阈值下的误差判断，两者独立。

## 数值验收

```bash
uv run pytest -q
uv run python -m tools.check_reference --profile test \
  --reference-directory /path/to/BUV0725 --output output/reference-check.json
```

四类小检查覆盖独立参考场/边带、解析导数、重构物理单位及工作流数据。还冻结了 PR16 同一四源场景的无噪声与三个含噪样本，保证重构前后的系数和 RMSE 在容差内一致。

原采样研究数据与图保存在 [最终报告](docs/results/followup/final-report.md)。这些是历史仿真结果，重构没有重新运行原采样扫描。原始大数组继续保存在远端，Git 内只保留摘要、必要黄金数组和图。
历史报告中的角度扫描为部件倾角；用户确认的第三目标是 `NA_in/NA_out` 通光角范围，正在通过 `apertures` 补做，不能用旧倾角结果替代。

模型匹配、无噪声算例的数值零不能当作仪器精度；参考倾斜是相位坡度近似，尚未计算严格旋转平面。
