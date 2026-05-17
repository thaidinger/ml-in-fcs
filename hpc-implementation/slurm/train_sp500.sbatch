#!/usr/bin/env bash
#SBATCH --job-name=fts-sp500-train
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/fts-diffusion-sp500-hpc/logs/%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/fts-diffusion-sp500-hpc/logs/%x_%j.err

set -euo pipefail

DEFAULT_REPO1="/iopsstor/scratch/cscs/$USER/fts-diffusion/fts-diffusion-sp500-hpc"
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO1}}"
CONTAINER_ENV="${CONTAINER_ENV:-/capstor/store/cscs/ethz/lsaie-ss26/environment/ngc.toml}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "[train-sbatch] ERROR: virtualenv not found at ${VENV_PATH}; create it per README"
  exit 1
fi

mkdir -p "${REPO_ROOT}/hpc-implementation/logs"
cd "${REPO_ROOT}"

echo "[train-sbatch] starting training on $(hostname)"
echo "[train-sbatch] REPO_ROOT=${REPO_ROOT} VENV_PATH=${VENV_PATH} CONTAINER_ENV=${CONTAINER_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLBACKEND=Agg
export REPO_ROOT VENV_PATH

srun --ntasks=1 --nodes=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" --environment="${CONTAINER_ENV}" \
  bash -lc '
set -euo pipefail
cd "${REPO_ROOT}"
PYTHON_BIN="${VENV_PATH}/bin/python"
if [[ -x "${VENV_PATH}/bin/activate" && -x "${VENV_PATH}/bin/python" ]]; then
  source "${VENV_PATH}/bin/activate"
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "[train-sbatch] ERROR: no usable Python interpreter found inside container"
    exit 1
  fi
  echo "[train-sbatch] WARNING: ${VENV_PATH}/bin/python is unavailable; falling back to ${PYTHON_BIN}"
fi
echo "[train-sbatch] python=${PYTHON_BIN}"
"${PYTHON_BIN}" -V
"${PYTHON_BIN}" hpc-implementation/scripts/train_sp500.py --repo-root "${REPO_ROOT}" --store-model "$@"
' _ "$@"

echo "[train-sbatch] finished"
