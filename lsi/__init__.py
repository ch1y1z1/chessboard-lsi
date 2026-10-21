"""Reference scalar diffraction and lateral-shearing wavefront reconstruction.

New workflow: config / annular / propagation / fourier / direct / study.
Run ``python -m lsi.study --help``. Physical defaults follow BUV0725; the raw
Fourier route follows Retrieval0806. The original five-beam modules below
remain a separate numerical demonstration and regression baseline.

* 论文路线:
  ``I --(相移 / 傅里叶变换)--> 差分波前 --(差分 Zernike 最小二乘)--> W``
* 直接非线性路线:
  ``I --(非线性前向模型上的 Levenberg-Marquardt)--> W``

模块组织（按计算链条）：

    model    网格与物理参数、棋盘光栅级次、Zernike 基、级次叠加前向模型
    invert   相移解调、载频瓣解调、Poisson 解包裹、差分 Zernike 重构
    lm       LM 直接光强反演（不经解调/解包裹）
"""
