#!/usr/bin/env bash
#SBATCH --job-name=wangzihang_astra
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
ASTRA_ROOT=/share-2/code/zihangwang/astra_policy
ASTRA_SESSION=${1:?Pass the prepared absolute session directory}
cd "$ASTRA_ROOT"
export PYTHONPATH="$ASTRA_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export EVAL_RUN_NAME=astra_liberai_test
exec /share-2/home/vla/miniforge3/envs/fastwam-dpnew-qwen/bin/python -m astra.run --session "$ASTRA_SESSION"
