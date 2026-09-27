#!/usr/bin/env bash
set -euo pipefail
ASTRA_ROOT=/share-2/code/zihangwang/astra_policy
ASTRA_SESSION=${1:?Pass the prepared absolute recovery session directory}
POLICY_ENV=/share-2/home/vla/miniforge3/envs/fastwam-dpnew-qwen
CODEX_HOME=${CODEX_HOME:-/share-2/home/vla/.codex-wangzihang}
if [[ ${ASTRA_GPU_ALLOCATION_CONFIRMED:-} != 1 ]]; then
    echo 'Set ASTRA_GPU_ALLOCATION_CONFIRMED=1 only after this GPU is assigned to the experiment.' >&2
    exit 2
fi
if [[ -z ${ASTRA_ASSIGNED_GPUS:-} || ${CUDA_VISIBLE_DEVICES:-} != "$ASTRA_ASSIGNED_GPUS" ]]; then
    echo 'ASTRA_ASSIGNED_GPUS must exactly match CUDA_VISIBLE_DEVICES.' >&2
    exit 2
fi
export ASTRA_JOB_ID=${ASTRA_JOB_ID:-bash_${HOSTNAME}_$(date +%s)_$$}
export ASTRA_ROOT ASTRA_SESSION CODEX_HOME
export PYTHONPATH="$ASTRA_SESSION/source_snapshot"
export PYTHONDONTWRITEBYTECODE=1
export EVAL_RUN_NAME=astra_liberai_test
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PATH="/share-2/home/vla/.nvm/versions/node/v24.15.0/bin:$POLICY_ENV/bin:$PATH"

if [[ ${ASTRA_INFRA_WATCHDOG_ENABLED:-1} == 1 ]]; then
    "$POLICY_ENV/bin/python" "$ASTRA_ROOT/scripts/restart_infrastructure.py" \
        --session "$ASTRA_SESSION" > "$ASTRA_SESSION/infrastructure_watchdog.stdout.log" 2>&1 &
    INFRA_WATCHDOG_PID=$!
    echo "$INFRA_WATCHDOG_PID" > "$ASTRA_SESSION/infrastructure_watchdog.pid"
fi

"$POLICY_ENV/bin/python" "$ASTRA_SESSION/source_snapshot/scripts/launch_core9_supervisors.py" \
    --session "$ASTRA_SESSION" > "$ASTRA_SESSION/external_supervisor_manager.log" 2>&1 &
SUPERVISOR_MANAGER_PID=$!
"$POLICY_ENV/bin/python" -m astra.core9_parallel --session "$ASTRA_SESSION" \
    > "$ASTRA_SESSION/bash_pool.log" 2>&1 &
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
