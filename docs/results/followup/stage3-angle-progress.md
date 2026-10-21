# 阶段 3 进度：有限角度对照

- 非零几何：8 个（camera/grating × x/y × ±0.25°），行数 48，失败 0。
- known：倾斜模型 + 对应几何 fixed/auto 独立标定；ignored：零角度模型 + 零角度 fixed/auto 标定；direct 均从零初值。
- 优化停止不等于恢复成功；ignored 几何保留大的非零误差。

| 部件 | 轴 | 角度 | 几何 | 方法 | 中心 | RMSE / waves |
|---|---|---:|---|---|---|---:|
| camera | x | -0.25 | ignored | direct | — | 0.0222254 |
| camera | x | -0.25 | ignored | fourier_calibrated | auto | 0.266306 |
| camera | x | -0.25 | ignored | fourier_calibrated | fixed | 0.266352 |
| camera | x | -0.25 | known | direct | — | 4.33229e-18 |
| camera | x | -0.25 | known | fourier_calibrated | auto | 0.00122481 |
| camera | x | -0.25 | known | fourier_calibrated | fixed | 0.00133319 |
| camera | x | 0.25 | ignored | direct | — | 0.0298212 |
| camera | x | 0.25 | ignored | fourier_calibrated | auto | 0.26308 |
| camera | x | 0.25 | ignored | fourier_calibrated | fixed | 0.263317 |
| camera | x | 0.25 | known | direct | — | 6.30659e-18 |
| camera | x | 0.25 | known | fourier_calibrated | auto | 0.000822876 |
| camera | x | 0.25 | known | fourier_calibrated | fixed | 0.00161449 |
| camera | y | -0.25 | ignored | direct | — | 0.107139 |
| camera | y | -0.25 | ignored | fourier_calibrated | auto | 0.179014 |
| camera | y | -0.25 | ignored | fourier_calibrated | fixed | 0.178978 |
| camera | y | -0.25 | known | direct | — | 1.01499e-17 |
| camera | y | -0.25 | known | fourier_calibrated | auto | 0.00155751 |
| camera | y | -0.25 | known | fourier_calibrated | fixed | 0.00212952 |
| camera | y | 0.25 | ignored | direct | — | 0.103204 |
| camera | y | 0.25 | ignored | fourier_calibrated | auto | 0.17304 |
| camera | y | 0.25 | ignored | fourier_calibrated | fixed | 0.172832 |
| camera | y | 0.25 | known | direct | — | 0 |
| camera | y | 0.25 | known | fourier_calibrated | auto | 0.000944369 |
| camera | y | 0.25 | known | fourier_calibrated | fixed | 0.00235651 |
| grating | x | -0.25 | ignored | direct | — | 0.0144902 |
| grating | x | -0.25 | ignored | fourier_calibrated | auto | 0.079311 |
| grating | x | -0.25 | ignored | fourier_calibrated | fixed | 0.0792216 |
| grating | x | -0.25 | known | direct | — | 9.25242e-18 |
| grating | x | -0.25 | known | fourier_calibrated | auto | 0.00119972 |
| grating | x | -0.25 | known | fourier_calibrated | fixed | 0.001208 |
| grating | x | 0.25 | ignored | direct | — | 0.00709862 |
| grating | x | 0.25 | ignored | fourier_calibrated | auto | 0.083718 |
| grating | x | 0.25 | ignored | fourier_calibrated | fixed | 0.0837789 |
| grating | x | 0.25 | known | direct | — | 1.0882e-17 |
| grating | x | 0.25 | known | fourier_calibrated | auto | 0.000712755 |
| grating | x | 0.25 | known | fourier_calibrated | fixed | 0.00141729 |
| grating | y | -0.25 | ignored | direct | — | 0.0253958 |
| grating | y | -0.25 | ignored | fourier_calibrated | auto | 0.0355229 |
| grating | y | -0.25 | ignored | fourier_calibrated | fixed | 0.0354891 |
| grating | y | -0.25 | known | direct | — | 1.09774e-17 |
| grating | y | -0.25 | known | fourier_calibrated | auto | 0.00212323 |
| grating | y | -0.25 | known | fourier_calibrated | fixed | 0.00250895 |
| grating | y | 0.25 | ignored | direct | — | 0.0192743 |
| grating | y | 0.25 | ignored | fourier_calibrated | auto | 0.042522 |
| grating | y | 0.25 | ignored | fourier_calibrated | fixed | 0.0426312 |
| grating | y | 0.25 | known | direct | — | 1.01499e-17 |
| grating | y | 0.25 | known | fourier_calibrated | auto | 0.00153706 |
| grating | y | 0.25 | known | fourier_calibrated | fixed | 0.00294355 |

图：`angle_known_ignored.png/svg`（log 纵轴，含零角度基线）。
