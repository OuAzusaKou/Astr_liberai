"""Runtime identity and GPU allocation checks for scheduled and direct runs."""
import os


def runtime_job_id():
    job_id = os.environ.get('SLURM_JOB_ID') or os.environ.get('ASTRA_JOB_ID')
    if not job_id:
        raise RuntimeError('A Slurm job or explicit ASTRA_JOB_ID is required')
    return job_id


def allocated_gpu_ids(require_core9=False):
    raw = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    values = [part.strip() for part in raw.split(',') if part.strip()]
    if (not values or any(not value.isdigit() for value in values)
            or len(set(values)) != len(values)):
        raise RuntimeError('CUDA_VISIBLE_DEVICES must list unique numeric GPU IDs')
    gpu_ids = [int(value) for value in values]
    if any(gpu_id < 0 or gpu_id > 7 for gpu_id in gpu_ids):
        raise RuntimeError('GPU IDs must be in the local 0-7 range')

    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
        if int(os.environ.get('SLURM_JOB_NUM_NODES', '1')) > 2:
            raise RuntimeError('User limits the evaluation to two nodes')
        if os.environ.get('SLURM_JOB_NAME') != 'wangzihang_astra':
            raise RuntimeError('Slurm evaluation jobs must use the wangzihang_astra task name')
        if require_core9 and (len(gpu_ids) != 8 or set(gpu_ids) != set(range(8))
                              or int(os.environ.get('SLURM_JOB_NUM_NODES', '0')) != 1):
            raise RuntimeError('Core9 requires one full eight-GPU wangzihang_astra allocation')
        return gpu_ids

    assigned = [part.strip() for part in os.environ.get('ASTRA_ASSIGNED_GPUS', '').split(',') if part.strip()]
    if (os.environ.get('ASTRA_GPU_ALLOCATION_CONFIRMED') != '1'
            or assigned != values or len(gpu_ids) > 8):
        raise RuntimeError('Direct bash runs require explicit matching ASTRA_ASSIGNED_GPUS and CUDA_VISIBLE_DEVICES')
    return gpu_ids
