#!/usr/bin/env bash
#SBATCH --job-name=wangzihang_astra
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1024G
#SBATCH --time=12:00:00
set -euo pipefail
ASTRA_ROOT=/share-2/code/zihangwang/astra_policy
ASTRA_SESSION=${1:?Pass the prepared absolute batch directory}
cd "$ASTRA_ROOT"
export PYTHONPATH="$ASTRA_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export EVAL_RUN_NAME=astra_liberai_test
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec /share-2/home/vla/miniforge3/envs/fastwam-dpnew-qwen/bin/python -m astra.parallel --session "$ASTRA_SESSION"
