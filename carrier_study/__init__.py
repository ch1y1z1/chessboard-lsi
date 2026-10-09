"""单帧载频反演方法评估（docs/carrier-research-plan.md 的实施代码）。

模块按研究计划的分层组织：

    config     研究常量：系统参数、网格、拟合项、种子
    zern       通用 Zernike（含 Z36 以外的截断尾项）与 RMS 归一化
    nominal    反演端名义模型：光强、雅可比、J^T v、Hessian（q 坐标）
    generator  生成端：超采样像素积分、级次/剪切/载频失配、照明、尾项、噪声
    fourier    傅里叶解调链（与 lsi.invert 逐位一致）及其精确线性化
    problem    统一最小二乘目标、成本计量、预算与轨迹记录
    solvers    求解器套件（GN/LM/TRF/BFGS/L-BFGS/CG/GD/BB/Nesterov/Adam）
    metrics    FIM、CRB、白化误差、波前误差
    truths     真值池（dev/test，四种谱形）

参数坐标统一用 q：q_j = c_j * rms(Z_j)，即每项对波前 RMS 的贡献（waves）。
Zernike 在单位圆上正交，故 ||q|| 就是拟合模式部分的波前 RMS。
"""

import os as _os

# 单线程 BLAS：成本标定与并行实验都要求每个进程只占一个核
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")
