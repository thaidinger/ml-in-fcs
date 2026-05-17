#!/usr/bin/env bash
#SBATCH --job-name=fts-sp500-downstream
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/fts-diffusion-sp500-hpc/logs/%x_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/fts-diffusion-sp500-hpc/logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/$USER/fts-diffusion-sp500-hpc}"
RESULTS_DIR="${RESULTS_DIR:-/iopsstor/scratch/cscs/$USER/fts-diffusion-sp500-hpc/results/sp500_hpc_full}"
CONTAINER_ENV="${CONTAINER_ENV:-/capstor/store/cscs/ethz/lsaie-ss26/environment/ngc.toml}"

mkdir -p "${REPO_ROOT}/logs" "${RESULTS_DIR}/raw"
cd "${REPO_ROOT}"

echo "[sbatch-master] running on $(hostname)"
echo "[sbatch-master] SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "[sbatch-master] SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID:-N/A}"
echo "[sbatch-master] SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "[sbatch-master] SLURM_NODELIST: ${SLURM_NODELIST:-N/A}"
echo "[sbatch-master] SLURM_NNODES: ${SLURM_NNODES:-N/A}"
echo "[sbatch-master] SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "[sbatch-master] REPO_ROOT: ${REPO_ROOT}"
echo "[sbatch-master] RESULTS_DIR: ${RESULTS_DIR}"
echo "[sbatch-master] CONTAINER_ENV: ${CONTAINER_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLBACKEND=Agg

echo "[sbatch-master] execute command on compute node"

CMD='
set -euo pipefail
cd "'"${REPO_ROOT}"'"
source "'"${REPO_ROOT}"'/.venv/bin/activate"
echo "[srun] rank=$SLURM_PROCID host=$(hostname) noderank=$SLURM_NODEID localrank=$SLURM_LOCALID array_task=${SLURM_ARRAY_TASK_ID:-N/A}"
echo "[srun] python=$(which python)"
python - <<'"'"'PY'"'"'
import torch
print(f"[srun] torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_devices={torch.cuda.device_count()}", flush=True)
if torch.cuda.is_available():
    print(f"[srun] gpu0={torch.cuda.get_device_name(0)}", flush=True)
PY
python hpc-implementation/scripts/run_downstream_task.py \
  --repo-root "'"${REPO_ROOT}"'" \
  --output-dir "'"${RESULTS_DIR}"'" \
  --run-id "${SLURM_ARRAY_TASK_ID}" \
  --experiment both \
  --datatype prices \
  --epochs 150 \
  --hidden-dim 64 \
  --window-size 64 \
  --ahead 1 \
  --loss mae \
  --tatr-augmentations 100 \
  --tatr-protocol author_independent \
  --tmtr-proportions 10 \
  --aug-length 252 \
  --mix-length 1260 \
  --torch-threads "'"${SLURM_CPUS_PER_TASK:-8}"'" \
  --epoch-log-interval 25
'

srun --environment="${CONTAINER_ENV}" bash -c "${CMD}"

echo "[sbatch-master] task finished"
