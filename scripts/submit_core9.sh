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
ASTRA_SESSION=${1:?Pass the prepared absolute Core9 batch directory}
cd "$ASTRA_SESSION/source_snapshot"
export PYTHONPATH="$ASTRA_SESSION/source_snapshot"
export PYTHONDONTWRITEBYTECODE=1
export EVAL_RUN_NAME=astra_liberai_test
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PATH=/share-2/home/vla/.nvm/versions/node/v24.15.0/bin:$PATH
POLICY_PYTHON=/share-2/home/vla/miniforge3/envs/fastwam-dpnew-qwen/bin/python
"$POLICY_PYTHON" "$ASTRA_SESSION/source_snapshot/scripts/launch_core9_supervisors.py" \
    --session "$ASTRA_SESSION" &
SUPERVISOR_MANAGER_PID=$!
"$POLICY_PYTHON" -m astra.core9_parallel --session "$ASTRA_SESSION" &
POOL_PID=$!
set +e
wait "$POOL_PID"
POOL_EXIT=$?
wait "$SUPERVISOR_MANAGER_PID"
SUPERVISOR_MANAGER_EXIT=$?
set -e
if (( POOL_EXIT != 0 || SUPERVISOR_MANAGER_EXIT != 0 )); then
    exit 1
fi
