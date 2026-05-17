#!/usr/bin/env bash
set -euo pipefail

# Build a small cluster payload with the reference code and only SP500 artifacts.
# Run from the repository root:
#   bash hpc-implementation/scripts/prepare_alps_payload.sh

PAYLOAD_DIR="${1:-hpc-implementation/payload/fts-diffusion-sp500-hpc}"
TARBALL="${2:-hpc-implementation/payload/fts-diffusion-sp500-hpc.tar.gz}"

rm -rf "${PAYLOAD_DIR}"
mkdir -p "${PAYLOAD_DIR}/fts-diffusion-ref" \
         "${PAYLOAD_DIR}/hpc-implementation" \
         "${PAYLOAD_DIR}/fts-diffusion-ref/trained_models" \
         "${PAYLOAD_DIR}/fts-diffusion-ref/res" \
         "${PAYLOAD_DIR}/fts-diffusion-ref/data"

rsync -a \
  --exclude '__pycache__/' \
  --exclude '.ipynb_checkpoints/' \
  --exclude 'trained_models/' \
  --exclude 'res/' \
  --exclude 'figs/' \
  --exclude 'data/' \
  fts-diffusion-ref/ "${PAYLOAD_DIR}/fts-diffusion-ref/"

rsync -a --exclude '__pycache__/' hpc-implementation/scripts hpc-implementation/slurm hpc-implementation/config "${PAYLOAD_DIR}/hpc-implementation/"
cp hpc-implementation/README.md "${PAYLOAD_DIR}/hpc-implementation/"
cp requirements.txt pyproject.toml "${PAYLOAD_DIR}/"
cp hpc-implementation/requirements_clariden.txt "${PAYLOAD_DIR}/hpc-implementation/"
cp fts-diffusion-ref/requirements_ref.txt "${PAYLOAD_DIR}/fts-diffusion-ref/"
cp fts-diffusion-ref/data/sp500_timeseries.csv "${PAYLOAD_DIR}/fts-diffusion-ref/data/"
cp fts-diffusion-ref/res/sisc_sp500_k14_l10-21_dba_kmpp_* "${PAYLOAD_DIR}/fts-diffusion-ref/res/"
cp fts-diffusion-ref/trained_models/pem_sp500_k14_e196_h32_lr4e-04_pw0.05_lw0.01_mw0.94.pth.pth "${PAYLOAD_DIR}/fts-diffusion-ref/trained_models/"
cp fts-diffusion-ref/trained_models/pem_sp500_k14_e196_h32_lr4e-04_pw0.05_lw0.01_mw0.94.pth.pt "${PAYLOAD_DIR}/fts-diffusion-ref/trained_models/"
cp fts-diffusion-ref/trained_models/pgm-2_c48-80_sp500_k14_n30_lr4e-04_dw0.01_pw1_sw0.01.pth.pth "${PAYLOAD_DIR}/fts-diffusion-ref/trained_models/"
cp fts-diffusion-ref/trained_models/pgm-2_c48-80_sp500_k14_n30_lr4e-04_dw0.01_pw1_sw0.01.pth.pt "${PAYLOAD_DIR}/fts-diffusion-ref/trained_models/"

mkdir -p "$(dirname "${TARBALL}")"
tar -czf "${TARBALL}" -C "$(dirname "${PAYLOAD_DIR}")" "$(basename "${PAYLOAD_DIR}")"

echo "Payload directory: ${PAYLOAD_DIR}"
echo "Tarball: ${TARBALL}"
echo "Included checkpoints:"
find "${PAYLOAD_DIR}/fts-diffusion-ref/trained_models" -maxdepth 1 -type f -print | sort
