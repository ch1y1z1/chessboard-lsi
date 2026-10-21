# 阶段 2c 进度：形状/长宽比与异常诊断

- 本批时间：2026-10-21T14:43:27+00:00 → 2026-10-21T14:48:58+00:00（约 331 s）
- 形状设置：5 个（1 个复用 rect + 4 个新算），失败 0 个。
- 异常诊断：(-2,-3) 与 (-2,-2) 两点；stable exact reproduction; storage/nondeterminism is not the cause。
- 峰值 RSS：2574 MiB；worker/batch=1。

## 五个窗口（固定 rect 名义标定）
| 设置 | RMSE / waves | PV / waves | coverage | phase residual | weight_sum | ENBW | max DC weight |
|---|---:|---:|---:|---:|---:|---:|---:|
| rect (reused) | 0.000932681 | 0.00757318 | 0.483143 | 0.0592384 | 1600 | 1794.78 | 0 |
| shape_circle | 0.000885721 | 0.00720557 | 0.483143 | 0.0587014 | 1605 | 1782.07 | 0 |
| shape_ellipse_0.5 | 0.000907415 | 0.00640808 | 0.483143 | 0.0602281 | 1589 | 1782.99 | 0 |
| shape_ellipse_2 | 0.00105883 | 0.00875893 | 0.483143 | 0.0594423 | 1589 | 1782.99 | 0 |
| shape_gaussian | 0.000502375 | 0.00437889 | 0.483143 | 0.0587465 | 1600 | 3200 | 0.0223411 |

## 面积/权重定义
- rect：40×40 方窗，连续面积 1600 bins²。
- circle：半径 `20*2/sqrt(pi)=22.568`，连续面积 1600 bins²。
- ellipse 0.5/2：`hx=20/sqrt(aspect)`、`hy=20*sqrt(aspect)`，连续面积 1600 bins²。
- gaussian：`sx=sy=20*sqrt(2/pi)`，连续积分权重 1600 bins²；无限支撑，实际 ENBW 和 DC 泄漏须看离散指标。
- 所有形状仍固定 rect 名义标定；形状变化与标定适配混在一起，不能声称每个形状各自最佳精度。

## 异常诊断
- 复现：stable exact reproduction; storage/nondeterminism is not the cause；最大复现差 `0`。
- `(-2,-3)` 相对邻点中位数低约 `0.000812536` waves。
- 解释：phase fields are similar after common processing; the isolated low value remains unexplained despite stable reproduction。
- 源阶段 2b 未保存 raw 系数，诊断只比较已有字段；精确复现只排除已检查的存储/汇总/非确定性问题，不能排除 unwrap/边界/平滑的联合作用。status=ok 不等于物理正确。

## 执行异常
- 主扫描已落盘后，原合并脚本在异常诊断中比较阶段 2b 未保存的 `raw_coeffs` 字段而报错；主扫描结果未受影响。
- 异常诊断改用独立脚本在新目录 `stage2c_anomaly_diag_v3` 重跑，未重做形状扫描；失败的空/部分诊断目录已清理。
- 诊断只比较已有字段，raw 系数不可比，已在 JSON 中注明。

## 限制
- 名义“半宽 20”在不同形状下不是同一截止频率；形状、面积、权重、支撑与 DC 泄漏共同影响结果。
- gaussian 的 ENBW 约 3200 bins（rect 约 1795），最大 DC 点窗口权重约 0.0223（只是窗口在 DC 频点的权重，不是总零频泄漏能量）；RMSE 更低不能单独归因于形状更优。
- 仅 5 个设置、单一测试图、固定 rect 标定；不能据此选择生产形状。

## 下一批建议
空间掩膜有限方案：固定 rect/半宽 20/名义标定、中心 0，先做 erosion 0/2/4/8（4 点），再做低幅剔除 0/.1/.3（3 点）；预计 7 个解算约 1–2 分钟，加启动/汇总约 3–5 分钟。
