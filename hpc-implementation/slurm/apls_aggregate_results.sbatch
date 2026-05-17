#!/usr/bin/env bash
#SBATCH --job-name=fts-sp500-aggregate
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/fts-diffusion/fts-diffusion-sp500-hpc/hpc-implementation/logs/%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/fts-diffusion/fts-diffusion-sp500-hpc/hpc-implementation/logs/%x_%j.err

set -euo pipefail

DEFAULT_REPO1="/iopsstor/scratch/cscs/$USER/fts-diffusion/fts-diffusion-sp500-hpc"
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO1}}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/hpc-implementation/results/sp500_hpc_full}"
CONTAINER_ENV="${CONTAINER_ENV:-/capstor/store/cscs/ethz/lsaie-ss26/environment/ngc.toml}"

cd "${REPO_ROOT}"
mkdir -p logs "${RESULTS_DIR}"

CMD='
set -euo pipefail
cd "'"${REPO_ROOT}"'"
source "'"${REPO_ROOT}"'/.venv/bin/activate"
python hpc-implementation/scripts/aggregate_downstream_results.py \
  --results-dir "'"${RESULTS_DIR}"'"
'

srun --environment="${CONTAINER_ENV}" bash -c "${CMD}"
