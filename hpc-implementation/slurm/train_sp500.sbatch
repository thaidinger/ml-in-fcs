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

# Auto-setup venv if missing, broken, or missing required packages
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  echo "[train-setup] venv at ${VENV_PATH} missing or broken; creating it"
  if [[ -d "${VENV_PATH}" ]]; then
    echo "[train-setup] removing stale venv"
    rm -rf "${VENV_PATH}"
  fi
  python -m venv --system-site-packages "${VENV_PATH}"
fi

VENV_PYTHON="${VENV_PATH}/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "[train-sbatch] ERROR: failed to create a usable python at ${VENV_PYTHON}"
  exit 1
fi

if ! "${VENV_PYTHON}" - <<'PY'
import importlib.util
import sys
missing = [name for name in ("yfinance",) if importlib.util.find_spec(name) is None]
sys.exit(0 if not missing else 1)
PY
then
  echo "[train-setup] yfinance missing from venv; installing hpc-implementation requirements"
  "${VENV_PYTHON}" -m pip install --upgrade pip
  "${VENV_PYTHON}" -m pip install -r hpc-implementation/requirements_clariden.txt
fi

if ! "${VENV_PYTHON}" - <<'PY'
import importlib.util
import sys
missing = [name for name in ("yfinance",) if importlib.util.find_spec(name) is None]
sys.exit(0 if not missing else 1)
PY
then
  echo "[train-sbatch] ERROR: yfinance is still unavailable in ${VENV_PYTHON} after installation"
  exit 1
fi

source "${VENV_PATH}/bin/activate"
PYTHON_BIN="${VENV_PYTHON}"
echo "[train-sbatch] python=${PYTHON_BIN}"
"${PYTHON_BIN}" -V
"${PYTHON_BIN}" hpc-implementation/scripts/train_sp500.py --repo-root "${REPO_ROOT}" --store-model "$@"
' _ "$@"

echo "[train-sbatch] finished"
