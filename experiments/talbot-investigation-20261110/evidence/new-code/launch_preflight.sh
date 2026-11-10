#!/usr/bin/env bash
set -euo pipefail
cd /data/yk/shear-reference-lm/talbot-rerun-20261109
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD/code"
exec /data/yk/conda/envs/shear-reference-lm-py312/bin/python code/preflight.py
