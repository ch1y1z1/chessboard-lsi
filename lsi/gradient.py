"""单帧光强模型的 J^T r 直接求值：不构造 M×p 雅可比。

供一阶/拟牛顿类优化器使用的轻量导数接口。数学上与
``lm._frame_and_jacobian`` 的显式雅可比路径一致，但只物化每光束的
复振幅 e_k（长度 M），把雅可比-向量运算降为逐光束的矩阵-向量乘：

    E = sum_k e_k,  e_k = inside_k * A_k * exp(i phi_k)
    phi_k = 2 pi (c @ Z_k) + modulation_k
    I = |E|^2,  r = I - target

    dI/dc_j = 2 Re( conj(E) * sum_k 2 pi i e_k Z_{k,j} )
    (J^T r)_j = sum_k 2 Re( 2 pi i * sum_m Z_k[j,m] * r_m * conj(E_m) * e_{k,m} )
            = -4 pi * sum_k ( Z_k @ Im(r * conj(E) * e_k) )_j

最后一步把复数 matvec 拆成实数运算：jtr 只依赖 Im(acc)，逐光束取
``w_im = r * (E.real * e.imag - E.imag * e.real)`` 后做实 dgemv。
若直接 ``Z @ (复数 w)``，NumPy 会把 (p, M) 的 Z 隐式转成 complex128
交给 zgemv —— 产生 p*M*16 字节的隐藏临时副本，恰好是本接口要避免的
雅可比量级分配（实测见 analysis/carrier_gradient_check.py）。

返回值与调用约定
----------------
``frame_intensity_jtr(cache, coeffs, target, modulation=None)``
返回 ``(intensity, jtr_coeff)``：

* ``intensity`` —— (n_rows,) float64，与显式路径相同的模型光强；
* ``jtr_coeff`` —— (n_terms,) float64，原始 Zernike 系数坐标下的
  J^T (I - target)，**不按像素数 M 归一化，也不含 q=scale*c 链式因子**。
  归一化与坐标变换由上层问题对象（T1）统一完成。

``cache`` 为 ``ForwardModel.zernike_samples(indices, rows)`` 的产物：
每项 ``(k, amp, inside, Z)``，``inside`` 为 (n_rows,) 布尔光瞳掩膜，
``Z`` 为 (n_terms, n_rows) 该级移位坐标上的 Zernike 基。
``modulation`` 支持 ``None``、每级标量相位 ``(n_orders,)`` 或
已按 rows 采样的空间相位 ``(n_orders, n_rows)``（如载频 ramp），
与 ``_frame_and_jacobian`` 接受的逐帧调制形式一致。

资源开销（n_orders = K，像素 M = n_rows，参数 p = n_terms）
------------------------------------------------------------
算法两遍扫描：第一遍逐光束算 e_k 并累加 E（物化 e_k 供第二遍复用，
避免重复的相位矩阵乘与 exp 求值），第二遍逐光束累加
``acc += Z_k @ (r * conj(E) * e_k)``，最后 ``jtr = -4 pi * Im(acc)``。

* 持久数组：E 16M 字节、I/r 各 8M、e_k 列表共 16K*M 字节、acc/jtr
  各 p 向量；逐光束临时为 phase/exp 链与第二遍的 w_im 实向量链。
  峰值估算约 (64~88 + 16K) * M + O(p) 字节（M=65512、K=5 时
  tracemalloc 实测峰值 9.5 MB，见分析脚本）；相比显式路径省去
  p*M 复数 dE 与
  M*p 实数 J，对 p=35、M=65536 即省去约 55 MB 的雅可比物化与读写。
* 浮点工作量：相位求值 K 次 (p*M) 矩阵乘（与显式路径的 dE 外积同阶），
  第二遍 K 次 (p*M) 实 matvec；总量级与显式 J + J^T r 相当。
  本接口的收益是内存与访存流量，不承诺更低的 flop 计数或更快耗时；
  任何速度结论须以独占条件下的实测为准。

不支持的用法：多帧堆叠请逐帧调用后按问题层的归一化约定求和；
本函数不读取真值、不做残差加权，mask/坏点只能经 cache 的 rows 固定。
"""
from __future__ import annotations

import numpy as np


def _checked_inputs(cache, coeffs, target, modulation):
    """形状/有限性检查；返回 (n_rows, n_terms, modulation)。"""
    if not cache:
        raise ValueError("cache 必须非空")
    n_rows = cache[0][2].size
    for entry in cache:
        if len(entry) != 4:
            raise ValueError("cache 每项必须是 (k, amp, inside, Z)")
        _, _, inside, Z = entry
        inside = np.asarray(inside)
        Z = np.asarray(Z)
        if inside.shape != (n_rows,) or inside.dtype != bool:
            raise ValueError("cache 的 inside 掩膜形状必须一致且为布尔型")
        if Z.ndim != 2 or Z.shape[1] != n_rows:
            raise ValueError("cache 的 Z 必须是 (n_terms, n_rows)")
    coeffs = np.asarray(coeffs, dtype=float)
    n_terms = cache[0][3].shape[0]
    if coeffs.shape != (n_terms,) or not np.isfinite(coeffs).all():
        raise ValueError("coeffs 必须有限且长度等于 Z 的项数")
    target = np.asarray(target, dtype=float)
    if target.shape != (n_rows,) or not np.isfinite(target).all():
        raise ValueError("target 必须是有限的 (n_rows,) 光强向量")
    if modulation is None:
        return n_rows, n_terms, None
    modulation = np.asarray(modulation, dtype=float)
    if modulation.shape not in ((len(cache),), (len(cache), n_rows)):
        raise ValueError(
            "modulation 必须是 (n_orders,) 标量相位或 (n_orders, n_rows) 空间相位")
    if not np.isfinite(modulation).all():
        raise ValueError("modulation 必须有限")
    return n_rows, n_terms, modulation


def frame_intensity_jtr(cache, coeffs, target, modulation=None):
    """一帧的模型光强与 J^T (I - target)（原始系数坐标，未归一化）。

    参数与返回语义见模块 docstring。与 ``lm._frame_and_jacobian``
    的显式雅可比路径数学等价，差异仅限浮点求和顺序。
    """
    n_rows, _, modulation = _checked_inputs(cache, coeffs, target, modulation)
    coeffs = np.asarray(coeffs, dtype=float)
    target = np.asarray(target, dtype=float)

    fields = []
    E = np.zeros(n_rows, dtype=complex)
    for k, amp, inside, Z in cache:
        phase = 2.0 * np.pi * (coeffs @ Z)
        if modulation is not None:
            phase = phase + modulation[k]
        e = np.where(inside, amp * np.exp(1j * phase), 0.0)
        fields.append(e)
        E += e
    intensity = np.abs(E) ** 2

    residual = intensity - target
    e_real, e_imag = E.real, E.imag
    # jtr 只用 Im(Z_k @ w_k)，w_k.imag = r * Im(conj(E) * e_k)，
    # 全部实数运算；Z @ 复向量会迫使 Z 复制为 complex128（p*M*16 字节）。
    acc = np.zeros(coeffs.size)
    for e, (_, _, _, Z) in zip(fields, cache):
        w_imag = residual * (e_real * e.imag - e_imag * e.real)
        acc += Z @ w_imag
    return intensity, -4.0 * np.pi * acc
