"""傅里叶变换（单帧空间载频）解调（论文 2.4）。

离焦引入空间载频 f0（f0 = m/2s，m 为泰伯级次），使 ±1 级干涉项在频谱
上移到 (±f0, 0) 与 (0, ±f0)。处理流程（论文 2.4.3）：

    I --二维 FFT--> 频谱 --提取 +f0 瓣--> 移回基带 --IFFT--> c(x,y)
      --arg--> 缠绕差分相位 --解包裹--> dW

这里用"逐像素去载频 + 带限低通"实现移频：c = IFFT( FFT(I e^{-i 2 pi f0 u})
* W )，相位参考保持在每个像素上 —— 被测瓣相对 f0 的位移（真实的线性
差分相位，如离焦）会保留在 arg c 中。

+f0 瓣的物理内容：两个对称一级都在时，瓣内含 E_+ E_0* 与 E_0 E_-* 两个
拍频，对对称棋盘系数合成为

    C_{+f0} = 2 A_0 A_1 cos(Gamma) * exp{ i pi [W(x+s,y) - W(x-s,y)] }

即论文的双边差分（式 2-42），条件是 cos(Gamma) 不变号（2.4.2 的调制度
约束）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Grid

__all__ = ["LobeResult", "spectrum", "demodulate_lobe"]


def spectrum(image: np.ndarray) -> np.ndarray:
    """实图像的移位二维频谱。"""
    return np.fft.fftshift(np.fft.fft2(np.asarray(image, dtype=float)))


@dataclass
class LobeResult:
    phase: np.ndarray          # 去除 phase_offset 后的缠绕相位 (rad)
    amplitude: np.ndarray      # |c(x, y)|
    field: np.ndarray          # 复数 c(x, y)
    peak_freq: tuple[float, float]  # 实测瓣峰频率 (fx, fy)，单位 cyc/归一化坐标


def _find_carrier_peak(
    mag: np.ndarray, grid: Grid, expected: tuple[float, float], search_px: int = 8
) -> tuple[int, int]:
    """在期望位置 ±search_px 窗口内取频谱幅值最大点，返回 (row, col)。"""
    n = mag.shape[0]
    L = 2.0 * grid.extent
    iy = int(round(expected[1] * L)) + n // 2
    ix = int(round(expected[0] * L)) + n // 2
    window = np.zeros_like(mag)
    window[max(0, iy - search_px) : iy + search_px + 1,
           max(0, ix - search_px) : ix + search_px + 1] = 1.0
    return np.unravel_index(int(np.argmax(mag * window)), mag.shape)


def demodulate_lobe(
    image: np.ndarray,
    grid: Grid,
    direction: str,
    f0: float,
    *,
    window_radius: float | None = None,
    phase_offset: float = 0.0,
) -> LobeResult:
    """解调单帧干涉图某方向的 +f0 载频瓣。

    ``direction="x"`` 取 (+f0, 0) 瓣，``"y"`` 取 (0, +f0) 瓣。
    ``window_radius`` 为低通滤波半径（频谱像素），缺省 4L（L 为网格宽度）。
    ``phase_offset`` 为光栅模型给出的常数相位（理想棋盘 x 方向为 pi），
    在取辐角前扣除，避免相位正好压在 ±pi 分支切线上。
    """
    I = np.asarray(image, dtype=float)
    x, y = grid.coords()
    n, L = grid.n, 2.0 * grid.extent
    ramp = x if direction == "x" else y

    # 实测瓣位置（仅作诊断；解调参考用名义 f0，瓣位移本身是信号）
    spec = spectrum(I)
    i0, j0 = _find_carrier_peak(
        np.abs(spec), grid,
        (f0, 0.0) if direction == "x" else (0.0, f0),
    )
    peak_freq = ((j0 - n // 2) / L, (i0 - n // 2) / L)

    # 逐像素去载频 -> 基带含 0 级谱与 +f0 瓣内容 -> Butterworth 低通
    baseband = np.fft.fftshift(np.fft.fft2(I * np.exp(-2j * np.pi * f0 * ramp)))
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    rr = np.hypot(xx - n // 2, yy - n // 2)
    rad = 4.0 * L if window_radius is None else float(window_radius)
    win = 1.0 / (1.0 + (rr / rad) ** 4)
    c = np.fft.ifft2(np.fft.ifftshift(baseband * win))
    if phase_offset:
        c = c * np.exp(-1j * phase_offset)

    return LobeResult(
        phase=np.angle(c),
        amplitude=np.abs(c),
        field=c,
        peak_freq=peak_freq,
    )
