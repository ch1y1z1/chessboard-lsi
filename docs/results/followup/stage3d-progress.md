# 阶段 3d 进度：camera x ±0.5° 代表扩展

- 行数：12；状态：complete；墙钟：1006s；峰值：3300 MiB。

| 几何 | 方法 | 中心 | RMSE / waves | converged |
|---|---|---|---:|---|
| camera_x_-0.5_known | direct | — | 1.0882e-17 | True |
| camera_x_-0.5_known_fixed | fourier_calibrated | fixed | 0.00144609 | — |
| camera_x_-0.5_known_auto | fourier_calibrated | auto | 0.00126195 | — |
| camera_x_-0.5_ignored | direct | — | 0.0419728 | False |
| camera_x_-0.5_ignored_fixed | fourier_calibrated | fixed | 0.533828 | — |
| camera_x_-0.5_ignored_auto | fourier_calibrated | auto | 0.533607 | — |
| camera_x_0.5_known | direct | — | 1.03363e-17 | True |
| camera_x_0.5_known_fixed | fourier_calibrated | fixed | 0.00148845 | — |
| camera_x_0.5_known_auto | fourier_calibrated | auto | 0.00133306 | — |
| camera_x_0.5_ignored | direct | — | 0.0463857 | True |
| camera_x_0.5_ignored_fixed | fourier_calibrated | fixed | 0.532757 | — |
| camera_x_0.5_ignored_auto | fourier_calibrated | auto | 0.532546 | — |

- known 使用倾斜模型和对应几何 fixed/auto 独立标定；ignored 使用零角度模型和零角度标定。
- ±0.5° 只做 camera x 代表扩展；不能外推到其他部件/轴。
- ignored direct 的 converged=True 不等于恢复成功。
