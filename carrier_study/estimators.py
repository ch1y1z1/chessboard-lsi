"""实验用估计器：观测构造、多起点 MLE 参考解、傅里叶链、扩展名义模型。

扩展模型（E4 的可选扩展，供"直接拟合能否吸收失配"使用）：

    IntegratedModel   名义模型在 S x S 子像素上计算后取平均（像素积分建模）
    AmplitudeModel    在 q 之外再估计 5 个级次振幅的对数缩放（nuisance）
二者都实现与 NominalModel 相同的 forward / jacobian / jtv / hessian_terms 接口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse

from lsi.model import Grid, SystemConfig

from .fourier import FourierChain, FourierParams
from .generator import Generator, Noise, Scene, add_noise
from .nominal import NominalModel
from .problem import LSProblem
from .solvers import run


# --------------------------------------------------------------------------- #
# 观测
# --------------------------------------------------------------------------- #
@dataclass
class Obs:
    clean: np.ndarray        # n x n 干净光强
    y: np.ndarray            # n x n 观测
    var_true: np.ndarray     # n x n 真实方差
    var_est: np.ndarray      # n x n 由观测估计的方差（Poisson 用于加权）
    noise: Noise


def observe(clean: np.ndarray, noise: Noise, rng: np.random.Generator) -> Obs:
    y, vt, ve = add_noise(clean, noise, rng)
    return Obs(clean, y, vt, ve, noise)


def poisson_scale(obs: Obs) -> float:
    """光电子 / 光强单位（相机增益视为已标定）。"""
    return obs.noise.level / float(obs.clean.max())


def model_weights(model, q: np.ndarray, obs: Obs) -> np.ndarray:
    """Poisson + 读出噪声：w = 1 / var(I_model(q))，归一化到均值 1。"""
    scale = poisson_scale(obs)
    I = np.clip(model.forward(q)[0], 0.0, None)
    w = scale**2 / (I * scale + obs.noise.read_e**2)
    return w / w.mean()


def irls(model, y_pix: np.ndarray, obs: Obs, q0: np.ndarray, n_outer: int = 3, max_iter: int = 200) -> np.ndarray:
    """迭代重加权最小二乘：权重取当前模型光强处的方差（近似 Poisson MLE）。

    不用观测计数估计方差：低计数像素会被赋予过大权重（Neyman χ² 偏差）。
    """
    q = np.asarray(q0, float)
    for _ in range(n_outer):
        P = LSProblem(model, y_pix, model_weights(model, q, obs))
        q = run("gn", P, q, max_iter=max_iter)["q"]
    return q


def fim_for(model, q: np.ndarray, obs: Obs) -> np.ndarray:
    """真值处 FIM（用真实方差）。"""
    _, st = model.forward(q)
    J = model.jacobian(st)
    var = model.take(obs.var_true)
    return (J / var[:, None]).T @ J


# --------------------------------------------------------------------------- #
# 多起点 MLE（参考解）
# --------------------------------------------------------------------------- #
def mle(model, y_pix: np.ndarray, w, inits: dict[str, np.ndarray], method: str = "gn",
        max_iter: int = 200) -> dict:
    starts = {}
    for name, q0 in inits.items():
        P = LSProblem(model, y_pix, w)
        res = run(method, P, q0, **({"max_iter": max_iter} if method == "gn" else {}))
        starts[name] = (float(res["loss"]), res["q"], res["status"])
    best = min(starts, key=lambda k: starts[k][0])
    lb, qb, _ = starts[best]
    same = [k for k, (l, q, _) in starts.items() if np.linalg.norm(q - qb) < 1e-6 * (1 + np.linalg.norm(qb))]
    distinct = []
    for k, (l, q, _) in starts.items():
        if all(np.linalg.norm(q - d) >= 1e-6 * (1 + np.linalg.norm(d)) for d in distinct):
            distinct.append(q)
    return {"q": qb, "loss": lb, "best_start": best, "n_agree": len(same),
            "n_starts": len(starts), "n_distinct": len(distinct),
            "start_losses": {k: v[0] for k, v in starts.items()},
            "start_q": {k: v[1] for k, v in starts.items()}}


def random_inits(p: int, rms_levels: Sequence[float], rng: np.random.Generator) -> dict[str, np.ndarray]:
    out = {}
    for i, r in enumerate(rms_levels):
        u = rng.normal(size=p)
        out[f"rand{i}_{r:g}"] = u / np.linalg.norm(u) * r
    return out


# --------------------------------------------------------------------------- #
# 扩展模型
# --------------------------------------------------------------------------- #
class IntegratedModel:
    """名义模型在 S x S 子像素上计算后盒式平均（建模像素积分）。"""

    def __init__(self, cfg: SystemConfig, indices: Sequence[int], S: int = 2) -> None:
        g = cfg.grid
        fine_cfg = SystemConfig(wavelength_nm=cfg.wavelength_nm, na=cfg.na, period_um=cfg.period_um,
                                grid=Grid(n=g.n * S, extent=g.extent))
        coarse = NominalModel(cfg, indices)
        n, nf = g.n, g.n * S
        # 粗像素集里每个像素的全部子像素
        rows_c, cols_c = np.divmod(coarse.pix, n)
        sub = []
        owner = []
        for i in range(S):
            for j in range(S):
                sub.append((rows_c * S + i) * nf + (cols_c * S + j))
                owner.append(np.arange(coarse.pix.size))
        sub = np.concatenate(sub)
        owner = np.concatenate(owner)
        order = np.argsort(sub)
        fine_pix = sub[order]
        self.fine = NominalModel(fine_cfg, indices, f0=cfg.carrier_f0, pixels=fine_pix)
        self.A = sparse.csr_matrix(
            (np.full(fine_pix.size, 1.0 / S**2), (owner[order], np.arange(fine_pix.size))),
            shape=(coarse.pix.size, fine_pix.size))
        self.pix, self.M, self.p = coarse.pix, coarse.M, coarse.p
        self.indices = coarse.indices
        self.cfg = cfg

    def forward(self, q):
        I, st = self.fine.forward(q)
        return self.A @ I, st

    def jacobian(self, state):
        return self.A @ self.fine.jacobian(state)

    def jtv(self, state, u):
        return self.fine.jtv(state, self.A.T @ u)

    def hessian_terms(self, state, u):
        return self.fine.hessian_terms(state, self.A.T @ u)

    def take(self, image):
        return np.asarray(image).ravel()[self.pix]


class AmplitudeModel:
    """参数 theta = (q, log a_k)：在 q 之外估计各级次振幅的对数缩放。"""

    def __init__(self, base: NominalModel) -> None:
        self.base = base
        self.K = len(base.blocks)
        self.p = base.p + self.K
        self.pq = base.p
        self.M, self.pix = base.M, base.pix

    def forward(self, theta):
        theta = np.asarray(theta, float)
        q, la = theta[: self.pq], theta[self.pq:]
        E = np.zeros(self.M, dtype=complex)
        es = []
        for blk, l in zip(self.base.blocks, la):
            e = blk.amp * np.exp(l) * np.exp(1j * (2 * np.pi * (blk.Zt @ q) + blk.mod))
            E[blk.idx] += e
            es.append(e)
        return E.real**2 + E.imag**2, (E, es)

    def jacobian(self, state):
        E, es = state
        Jq = self.base.jacobian(state)
        Ja = np.zeros((self.M, self.K))
        for k, (blk, e) in enumerate(zip(self.base.blocks, es)):
            Ja[blk.idx, k] = 2.0 * np.real(np.conj(E[blk.idx]) * e)
        return np.hstack([Jq, Ja])

    def jtv(self, state, u):
        return self.jacobian(state).T @ u

    def take(self, image):
        return self.base.take(image)


def clean_image(cfg, indices, q, scene: Scene, tail=None) -> np.ndarray:
    return Generator(cfg, indices, scene).clean(q, tail)


def fourier_q(chain: FourierChain, image: np.ndarray) -> np.ndarray:
    try:
        q, _ = chain.estimate(image)
        if not np.all(np.isfinite(q)):
            raise FloatingPointError
        return q
    except Exception:  # noqa: BLE001  掩膜退化等：记为失败，由调用方处理
        return np.full(len(chain.indices), np.nan)


def chain_for(cfg, indices, params: FourierParams | None) -> FourierChain:
    return FourierChain(cfg, indices, params or FourierParams())
