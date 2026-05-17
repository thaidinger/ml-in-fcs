#!/usr/bin/env bash
#SBATCH --job-name=fts-sp500-downstream
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/fts-diffusion-sp500-hpc/logs/%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/fts-diffusion-sp500-hpc/logs/%x_%j.err

set -euo pipefail

DEFAULT_REPO1="/iopsstor/scratch/cscs/$USER/fts-diffusion/fts-diffusion-sp500-hpc"
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO1}}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/hpc-implementation/results/sp500_hpc_full}"
CONTAINER_ENV="${CONTAINER_ENV:-/capstor/store/cscs/ethz/lsaie-ss26/environment/ngc.toml}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
RUN_COUNT="${RUN_COUNT:-100}"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "[sbatch-master] ERROR: could not find a virtualenv at ${VENV_PATH}"
  echo "[sbatch-master] Set VENV_PATH to the correct venv or create ${REPO_ROOT}/.venv"
  exit 1
fi

PYTHON_BIN="${VENV_PATH}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    echo "[sbatch-master] WARNING: ${VENV_PATH}/bin/python is unavailable; falling back to ${PYTHON_BIN}"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
    echo "[sbatch-master] WARNING: ${VENV_PATH}/bin/python is unavailable; falling back to ${PYTHON_BIN}"
  else
    echo "[sbatch-master] ERROR: no usable Python interpreter found"
    exit 1
  fi
fi

mkdir -p "${REPO_ROOT}/hpc-implementation/logs" "${RESULTS_DIR}/raw"
cd "${REPO_ROOT}"

JOB_LOG_DIR="${REPO_ROOT}/hpc-implementation/logs/${SLURM_JOB_NAME:-fts-sp500-downstream}_${SLURM_JOB_ID:-unknown}"
mkdir -p "${JOB_LOG_DIR}"

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
echo "[sbatch-master] VENV_PATH: ${VENV_PATH}"
echo "[sbatch-master] PYTHON_BIN: ${PYTHON_BIN}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLBACKEND=Agg
export REPO_ROOT RESULTS_DIR CONTAINER_ENV VENV_PATH RUN_COUNT JOB_LOG_DIR PYTHON_BIN

echo "[sbatch-master] execute command on compute node"

exec > >(tee -a "${JOB_LOG_DIR}/master.out") 2> >(tee -a "${JOB_LOG_DIR}/master.err" >&2)

srun --ntasks=1 --nodes=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --environment="${CONTAINER_ENV}" \
  --output="${JOB_LOG_DIR}/${SLURM_JOB_ID}.out" \
  --error="${JOB_LOG_DIR}/${SLURM_JOB_ID}.err" \
  bash <<'EOF'
set -euo pipefail
cd "${REPO_ROOT}"
if [[ -x "${VENV_PATH}/bin/activate" && -x "${VENV_PATH}/bin/python" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
echo "[srun] rank=$SLURM_PROCID host=$(hostname) noderank=$SLURM_NODEID localrank=$SLURM_LOCALID array_task=${SLURM_ARRAY_TASK_ID:-N/A}"
python_exe="${PYTHON_BIN}"
echo "[srun] python=${python_exe}"
"${PYTHON_BIN}" - <<'PY'
import torch
print(f"[srun] torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_devices={torch.cuda.device_count()}", flush=True)
if torch.cuda.is_available():
    print(f"[srun] gpu0={torch.cuda.get_device_name(0)}", flush=True)
PY
for run_id in $(seq 0 $((${RUN_COUNT} - 1))); do
  echo "[run] starting run_id=${run_id}"
  "${PYTHON_BIN}" hpc-implementation/scripts/run_downstream_task.py \
    --repo-root "${REPO_ROOT}" \
    --output-dir "${RESULTS_DIR}" \
    --run-id "${run_id}" \
    --experiment both \
    --datatype prices \
    --epochs 100 \
    --hidden-dim 32 \
    --window-size 64 \
    --ahead 1 \
    --loss mae \
    --tatr-augmentations 100 \
    --tatr-protocol author_independent \
    --tmtr-proportions 10 \
    --aug-length 252 \
    --mix-length 1260 \
    --torch-threads "${SLURM_CPUS_PER_TASK:-8}" \
    --epoch-log-interval 25 2>&1 | tee "${JOB_LOG_DIR}/run_${run_id}.log"
  echo "[run] finished run_id=${run_id}"
done
EOF

echo "[sbatch-master] task finished"
