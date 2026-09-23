"""lsi -- 45° 棋盘光栅剪切干涉：前向模型 + 三条波前反演路线。

* 论文路线:
  ``I --(相移 / 傅里叶变换)--> 差分波前 --(差分 Zernike 最小二乘)--> W``
* 直接非线性路线:
  ``I --(非线性前向模型上的 Levenberg-Marquardt)--> W``

模块组织（按计算链条）：

    config       网格与物理参数（lambda/NA/p -> s, f0）
    zernike      Fringe 序 Zernike 与差分基
    grating      棋盘光栅衍射级次（表 2-3）
    forward      级次叠加前向模型 I = |E|^2
    phaseshift   N 步相移解调 + 剪切区域
    unwrap       掩膜内 Poisson 解包裹
    ftmode       单帧载频瓣解调
    reconstruct  差分 Zernike 最小二乘
    pipeline     两条解调路线 + 重构的端到端封装
    lm           LM 直接光强反演
    metrics      PV / RMS / 系数误差
"""
