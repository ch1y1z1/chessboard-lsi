# 阶段 3b 进度：camera ±0.25° 扩展

- 行数：18（3 个新几何 × 6 行）；状态：complete；秒：1006；峰值：3346 MiB。

| 几何 | 方法 | 中心 | RMSE / waves |
|---|---|---|---:|
| camera_x_-0.25_ignored | direct | — | 0.0222254 |
| camera_x_-0.25_ignored_auto | fourier_calibrated | auto | 0.266306 |
| camera_x_-0.25_ignored_fixed | fourier_calibrated | fixed | 0.266352 |
| camera_x_-0.25_known | direct | — | 4.33229e-18 |
| camera_x_-0.25_known_auto | fourier_calibrated | auto | 0.00122481 |
| camera_x_-0.25_known_fixed | fourier_calibrated | fixed | 0.00133319 |
| camera_y_-0.25_ignored | direct | — | 0.107139 |
| camera_y_-0.25_ignored_auto | fourier_calibrated | auto | 0.179014 |
| camera_y_-0.25_ignored_fixed | fourier_calibrated | fixed | 0.178978 |
| camera_y_-0.25_known | direct | — | 1.01499e-17 |
| camera_y_-0.25_known_auto | fourier_calibrated | auto | 0.00155751 |
| camera_y_-0.25_known_fixed | fourier_calibrated | fixed | 0.00212952 |
| camera_y_0.25_ignored | direct | — | 0.103204 |
| camera_y_0.25_ignored_auto | fourier_calibrated | auto | 0.17304 |
| camera_y_0.25_ignored_fixed | fourier_calibrated | fixed | 0.172832 |
| camera_y_0.25_known | direct | — | 0 |
| camera_y_0.25_known_auto | fourier_calibrated | auto | 0.000944369 |
| camera_y_0.25_known_fixed | fourier_calibrated | fixed | 0.00235651 |

- known 使用倾斜模型与对应几何 fixed/auto 独立标定；ignored 使用零角度模型与零角度标定；direct 均零初值。
- ignored direct 多次停在非零局部解；优化停止不等于恢复成功。
