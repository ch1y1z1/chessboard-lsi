# 参考衍射模型下的二维剪切干涉反演

以 `BUV0725` 的三段标量传播和 `Retrieval0806` 的傅里叶解算为基准，研究：

1. 同一张衍射干涉图上的傅里叶法与直接非线性最小二乘。
2. 频谱选窗和空间有效区对精度的影响。
3. 物面、光栅与相机相关平面倾斜时的解算变化。

模型使用参考的 6.7 nm 波长、环形光瞳、RMS 归一化环形 Zernike、周期方孔阵列和源点强度叠加。`reference` 配置保留全部 64 个源点及 1200×1200 / 2400×2400 采样。光栅衍射通过透射函数与 Fresnel/CZT 传播计算，不截取为五束光。

方孔阵列默认使用与逐段 CZT 等价的可分离算子，预计算光瞳频率到相机的传递矩阵，并直接计算选定像素处的场与导数。`--backend czt` 切回逐段计算。两者已通过复场/强度与导数核对；没有用代表源或场插值代替源点叠加。

## 运行

当前开发机内存有限，**不在这台机器上运行原采样研究或批量扫描**。
命令必须显式指定 `--profile` 或 `--config`，不会默认启动实验。
超过小测试场景的计算还必须带 `--allow-large-run`，包括 64 源粗采样。
默认每批一个源、一个计算 worker；同一用户的实验入口用进程锁串行运行。
这不是硬内存上限，原采样单任务此前已记录约 1.4 GiB 峰值。
中断会保存状态和已有结果；输出目录已有运行清单时拒绝覆盖。

```bash
uv sync --locked

# 本机只做小场景检查；四源测试场景不用于科研精度结论
uv run python experiment.py --profile test --starts 1 --max-nfev 20 --output output/smoke-new
```

以下命令留给内存充足的机器，每次只运行一个，使用新的输出目录。
原采样参数继续以参考为准，不会自动降采样。

```bash
# 原始参考采样
uv run python experiment.py --profile reference --allow-large-run --truth-rms .1 --output output/diffraction/compare-reference

# 64 源粗采样仅供开发检查，不能据此下精度结论
uv run python experiment.py --profile quick --allow-large-run --truth-rms .1 --output output/diffraction/compare-coarse

# 对照原参考的逐段复场与全部源点强度（需要本机参考目录）
uv run python -m lsi.study verify-reference --profile reference --allow-large-run --all-sources --output output/diffraction/parity

# 窗宽、二维中心偏移、形状、椭圆长宽比、空间掩膜扫描
uv run python -m lsi.study regions --profile reference --allow-large-run --truth-rms .1 --starts 1 --output output/diffraction/regions

# 两个转轴、三个部件，已知角度与忽略角度分别对照
uv run python -m lsi.study angles --profile reference --allow-large-run --truth-rms .1 --angles -.25 .25 --axes x y --starts 1 --output output/diffraction/angles
```

`uv run python -m lsi.study --help` 列出完整选项。`--config path.json` 可读取参考格式的物理配置；`--reference-directory` 可指定原脚本目录。原始 NPY 缺少足够的生成元信息，因此未按文件名猜测角度并用作定量真值。

## 两种方法与标定

`fourier_original` 保留参考的矩形窗、Gaussian 边缘、整数移频、掩膜外置零、skimage 解包裹、相位平滑和 Lasso。它输出相位拟合的 a.u. 系数，不能直接当作物镜光瞳上的 waves。

`fourier_calibrated` 是明确分列的标定版本：先用独立零像差参考的复边带消除基准相位，再解包裹，以线性差分基拟合并通过独立单项 ±像差样本建立局部响应。标定文件含条件数及未参与训练的混合像差验证。该线性换算不保证适用于大像差。扫描默认冻结名义窗口的标定；`--recalibrate-windows` 是单独的诊断选项。

直接法每步通过完整传播链计算光强及解析导数，并联合拟合正曝光比例与常量背景。默认比较零初值与固定种子的独立随机初值。`--screen-initials 32 --starts 3` 可先按观测光强残差筛选 0.03/0.1/0.3 waves RMS 的候选初值；筛选过程和全部优化轨迹均保存。优化停止不等于恢复真值。

默认未知量为参考的四个非零像差模式：零基下标 `7,25,11,15`，即 `(n,l)=(3,1),(5,-5),(5,-1),(3,3)`。`--modes all` 暴露除 piston 外的 35 项；该更高维问题需另行验证可辨识性和局部标定。若配置含未纳入未知量的非零模式，程序报错，避免把真值当作已知量泄漏给反演。

## 结果与比较口径

每次运行写出 `manifest.json`、`results.csv/json`、原始强度、归一化观测、标定与独立验证样本、选窗/掩膜、初值与逐步轨迹、采样像素和 PNG/SVG 图。清单包含完整物理配置、模式标签、随机种子、软件版本、代码散列、运行时间与进程峰值内存。

波前在固定物镜环形光瞳内评价。主 RMSE/PV 沿用参考去除 piston、双向倾斜和离焦的约定，另报保留倾斜/离焦的 RMSE 和所有拟合系数误差。整体变号不计为正确恢复。原版 a.u. 结果不参与 waves 误差表。噪声可用 `--snr-db` 与不同 `--seed` 重复，SNR 定义为干净观测峰值与 Gaussian 噪声标准差之比的 20 log10。

`reference` 与 `retrieval` 分别保留两份参考不同的轴向偏移/error_mode；每张图只有一份生成配置，反演不另行混用参数。`quick` 保留全部 64 源及相同的物面、光栅、相机物理范围，但采样变粗，不能代替原网格的数值收敛验证。`test` 是独立的四源测试场景，不是 64 源的近似。

参考倾斜实现是输入场的相位坡度修正，不是严格旋转平面传播。单像素点源的物面倾斜只改变整体相位，因此应作为零效应对照，不能据此推论真实扩展光源无倾斜影响。

## 代码

- `lsi/config.py` / `annular.py`：参考配置与环形基。
- `lsi/propagation.py`：源场、4f、光栅与相机传播；缓存批量 CZT。
- `lsi/fourier.py`：原版流程、选区及独立标定。
- `lsi/direct.py` / `metrics.py`：直接拟合、共同评价。
- `lsi/study.py`：比较与两类参数扫描。
- `lsi/execution.py`：显式运行开关、实验进程互斥与中断处理。
- `lsi/reference_check.py`：读取用户提供的原脚本，独立核对数值。
- `tests/data/`：由原脚本生成的小场景复场、强度及边带基准，测试不依赖本机绝对路径。

[研究计划](docs/research-plan.md)描述实验设计；[实施与实跑记录](docs/implementation-report.md)记录当前验证范围和限制。旧的五光束演示保存在 `experiments/legacy_five_beam.py`，`lsi/model.py`、`invert.py`、`lm.py` 与 `test_lsi.py` 仅作该演示的回归基线。
