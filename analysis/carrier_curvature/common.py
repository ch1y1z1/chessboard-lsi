"""T5 公共件：旧数据加载、场景定义、统一口径问题构造。

只读调用既有 lsi 模块；不修改公共源码与旧数据。
旧载频基准（output/medium-carrier）的 dataset.npz/runs.csv/trajectories 为输入。
真值仅用于离线评价与局部机制诊断，不参与任何求解/选解/停止。
"""
from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path

import numpy as np

# 单线程，便于复核；不修改系统全局环境（仅本进程）。
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

REPO = Path(__file__).resolve().parents[2]
OLD_RUN = Path(
    "/home/user/.lody/repos/github---ch1y1z1---chessboard-lsi/worktrees/"
    "cd15d640-c386-4307-a7c4-5966543b462f/output/medium-carrier"
)
ARTIFACTS = Path(
    "/home/user/research-artifacts/chessboard-lsi/carrier-research-20261009/T5"
)

# 与旧基准 metadata.json 一致的几何/物理配置（重建 ForwardModel 用）。
INDICES = list(range(2, 14))
GRID_N = 128
EXTENT = 1.10
PERIOD_UM = 30.0  # 傅里叶/载频模式

# Part 1：三实现 LM 比较的 6 个代表场景 (case_id, initial_id)。
PART1_SCENARIOS = [
    (0, 0),    # truth_rms 0.01，零初值
    (0, 6),    # truth_rms 0.01，init 0.1，旧 LM 在此梯度容差停但严格失败
    (5, 2),    # truth_rms 0.03
    (10, 8),   # truth_rms 0.1
    (14, 4),   # truth_rms 0.3
    (17, 9),   # truth_rms 1.0，init 0.3
]

# Part 2：诊断场景（case_id, initial_id, method, 采样点角色）。
# 角色 in {"init","mid","final"}；mid 取轨迹中位评价点。
PART2_SCENARIOS = [
    # A. 梯度容差停、严格失败但工程成功（阈值边缘停机质量）
    (0, 6, "lm", ["final"]),
    (0, 7, "gn", ["final"]),
    (4, 5, "lm", ["final"]),
    (7, 0, "trf", ["final"]),
    (17, 4, "lm", ["final"]),
    (17, 1, "lm", ["final"]),
    # B. momentum 预算耗尽停滞（含负号对齐与正号）
    (0, 8, "momentum", ["init", "mid", "final"]),
    (1, 5, "momentum", ["mid", "final"]),
    (1, 2, "momentum", ["final"]),
    (0, 7, "momentum", ["final"]),
    # C. 零初值对照
    (0, 0, "lm", ["mid", "final"]),
    (0, 0, "momentum", ["final"]),
    # D. 幅度对照（truth_rms 1.0/0.1）
    (16, 1, "lm", ["final"]),
    (16, 1, "momentum", ["final"]),
    (10, 8, "lm", ["final"]),
    (8, 2, "gn", ["final"]),
    # E. 其他一阶法预算端点
    (0, 8, "gd", ["final"]),
    (0, 8, "adam", ["final"]),
    # F. 拟牛顿端点
    (16, 1, "bfgs", ["final"]),
    # G. trf/gn 严格通过端点对照
    (16, 1, "gn", ["final"]),
]


def load_dataset():
    """读取旧基准 dataset.npz（只读）。"""
    return np.load(OLD_RUN / "dataset.npz")


def load_runs():
    with (OLD_RUN / "runs.csv").open() as stream:
        return list(csv.DictReader(stream))


def find_run(rows, case_id, initial_id, method):
    for row in rows:
        if (int(row["case_id"]) == case_id
                and int(row["initial_id"]) == initial_id
                and row["method"] == method):
            return row
    raise KeyError((case_id, initial_id, method))


def fitted_coeffs(row):
    """runs.csv 行 -> incumbent 系数向量（waves，按 INDICES 序）。"""
    return np.array([float(row[f"fitted_Z{j}"]) for j in INDICES])


def load_trajectory(case_id, initial_id, method):
    """按 (case, init, method) 取轨迹记录；文件中同键只出现一次。"""
    with gzip.open(OLD_RUN / "trajectories.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line)
            if (record["case_id"] == case_id
                    and record["initial_id"] == initial_id
                    and record["method"] == method):
                return record
    raise KeyError((case_id, initial_id, method))


def make_forward():
    from lsi.model import ForwardModel, Grid, SystemConfig

    return ForwardModel(SystemConfig(
        period_um=PERIOD_UM, grid=Grid(n=GRID_N, extent=EXTENT)))


def make_problem(fm, frame, modulation, scaling="rms"):
    """统一诊断口径下的单帧载频问题（全像素、归一化残差）。"""
    from lsi.problem import StackedIntensityProblem

    return StackedIntensityProblem(
        fm, INDICES, [frame], [modulation], scaling=scaling)


def carrier_problem(fm, dataset, case_id, scaling="rms"):
    return make_problem(
        fm, dataset["frames"][case_id, 0], dataset["modulations"][case_id, 0],
        scaling=scaling)


def trajectory_points(record, roles, final_q):
    """从轨迹 history 选代表点。q 均为 q 坐标。

    init=第一条评价，mid=按评价序号中位（去重后），final=incumbent。
    """
    points = {}
    qs = [np.asarray(ev["q"]) for ev in record["history"]]
    if "init" in roles:
        points["init"] = qs[0]
    if "mid" in roles:
        # 取接受迭代意义上的中位：按唯一 q 去重后的中位评价点
        uniq, seen = [], set()
        for q in qs:
            key = q.tobytes()
            if key not in seen:
                seen.add(key)
                uniq.append(q)
        points["mid"] = uniq[len(uniq) // 2]
    if "final" in roles:
        points["final"] = np.asarray(final_q, dtype=float)
    return points
