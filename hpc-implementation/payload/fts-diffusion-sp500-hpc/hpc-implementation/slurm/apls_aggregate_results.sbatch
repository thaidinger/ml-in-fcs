#!/usr/bin/env bash
#SBATCH --job-name=fts-sp500-aggregate
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/fts-diffusion-sp500-hpc/logs/%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/fts-diffusion-sp500-hpc/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/$USER/fts-diffusion-sp500-hpc}"
RESULTS_DIR="${RESULTS_DIR:-/iopsstor/scratch/cscs/$USER/fts-diffusion-sp500-hpc/results/sp500_hpc_full}"
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
