# HPC Implementation for SP500 TMTR/TATR on Clariden/Alps

This folder contains a small, readable setup for running the downstream SP500
TMTR/TATR experiments on Clariden, the Swiss AI partition of CSCS Alps.

The settings are based on your local Large-Scale AI course files:

- login route: `ssh clariden` via `ela.cscs.ch`
- account: `lsaie-ss26`
- production partition: `normal`
- short smoke-test partition: `debug`
- fast job storage: `/iopsstor/scratch/cscs/$USER`
- long-term storage: `/capstor/scratch/cscs/$USER`
- course container: `/capstor/store/cscs/ethz/lsaie-ss26/environment/ngc.toml`

Compute jobs write to `/iopsstor`, not `/capstor`, following the course notes.

## What Is Included

- `scripts/run_downstream_task.py`: one downstream run id (invoked per run).
- `scripts/aggregate_downstream_results.py`: combines task CSVs and creates plots.
- `scripts/prepare_alps_payload.sh`: builds a clean upload bundle.
-- `slurm/apls_downstream_array.sbatch`: single-job runner that loops multiple downstream runs (configurable via `RUN_COUNT`).
- `slurm/apls_aggregate_results.sbatch`: post-processing/plot job.
- `config/sp500_hpc_settings.env`: readable record of the chosen settings.
- `requirements_clariden.txt`: extra Python packages installed into a project venv.
- `results/`: local placeholder for generated HPC results.

## Chosen HPC Settings

Compared with the small local tests, the cluster job uses:

  - Multiple independent runs executed sequentially inside one Slurm job (set `RUN_COUNT`, default 100). You can submit with `sbatch --export=RUN_COUNT=100 slurm/apls_downstream_array.sbatch` to override.
- 150 LSTM epochs per fit
- hidden size 64
- TATR augmentation grid `0..100` yearly blocks
- TMTR synthetic proportions `0,10,...,100`
- SP500 prices, one-day ahead, window size 64, MAE loss

The default TATR protocol is `author_independent`, matching the released
reference-code behavior where each added 252-day synthetic block is generated
from the same initial reference segment.  You can switch to
`continuous_chunked` in the Slurm file if you want the protocol-diagnostic
version where a long synthetic trajectory is split into chunks.

## Files to Take to the Cluster

Run this locally from the repo root:

```bash
bash hpc-implementation/scripts/prepare_alps_payload.sh
```

This creates:

```text
hpc-implementation/payload/fts-diffusion-sp500-hpc/
hpc-implementation/payload/fts-diffusion-sp500-hpc.tar.gz
```

The payload includes the reference code and only these SP500 model files:

```text
fts-diffusion-ref/trained_models/pem_sp500_k14_e196_h32_lr4e-04_pw0.05_lw0.01_mw0.94.pth.pth
fts-diffusion-ref/trained_models/pem_sp500_k14_e196_h32_lr4e-04_pw0.05_lw0.01_mw0.94.pth.pt
fts-diffusion-ref/trained_models/pgm-2_c48-80_sp500_k14_n30_lr4e-04_dw0.01_pw1_sw0.01.pth.pth
fts-diffusion-ref/trained_models/pgm-2_c48-80_sp500_k14_n30_lr4e-04_dw0.01_pw1_sw0.01.pth.pt
```

It also includes the SP500 time series and SP500 SISC artifacts, because the
reference loaders need those to build downstream data and sampling inputs.

## Recommended Cluster Layout

Use these locations:

```text
/iopsstor/scratch/cscs/$USER/fts-diffusion-sp500-hpc/          code, venv, logs
/iopsstor/scratch/cscs/$USER/fts-diffusion-sp500-hpc/results/  generated CSVs and plots
/capstor/scratch/cscs/$USER/fts-diffusion-sp500-hpc-archive/   optional final archive
```

Keep active results on `/iopsstor`, because 100 array tasks produce many logs
and intermediate CSVs.  Copy the final result folder to `/capstor` only after
the jobs are done.

## Upload and Unpack

From your laptop:

```bash
scp hpc-implementation/payload/fts-diffusion-sp500-hpc.tar.gz <USER>@<ALPS_LOGIN>:~
```

On the cluster:

```bash
mkdir -p /iopsstor/scratch/cscs/$USER
tar -xzf ~/fts-diffusion-sp500-hpc.tar.gz -C /iopsstor/scratch/cscs/$USER
cd /iopsstor/scratch/cscs/$USER/fts-diffusion-sp500-hpc
```

## Environment Setup on the Cluster

Use the course NGC container for PyTorch/CUDA and create a small venv inside the
project for the additional reference-code packages.  Run this once from an
interactive debug job:

```bash
srun --account=lsaie-ss26 \
  --partition=debug \
  --container-writable \
  --environment=/capstor/store/cscs/ethz/lsaie-ss26/environment/ngc.toml \
  --pty bash

cd /iopsstor/scratch/cscs/$USER/fts-diffusion/fts-diffusion-sp500-hpc/
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r hpc-implementation/requirements_clariden.txt
```

Do not run `pip install -r fts-diffusion-ref/requirements_ref.txt` on Clariden.
That file pins old CPU-era package versions and can fight the course container.
Also do not install or downgrade `torch`; use the container's PyTorch build.

Quick sanity check:

```bash
source .venv/bin/activate
python hpc-implementation/scripts/run_downstream_task.py \
  --repo-root "$PWD" \
  --output-dir hpc-implementation/results/smoke \
  --run-id 0 \
  --experiment tatr \
  --epochs 1 \
  --tatr-augmentations 1 \
  --hidden-dim 16


srun --account=lsaie-ss26 \
  --partition=debug \
  --time=00:05:00 \
  --container-writable \
  --environment=/capstor/store/cscs/ethz/lsaie-ss26/environment/ngc.toml \
  --pty bash -lc '
    cd /iopsstor/scratch/cscs/course_00241/fts-diffusion/fts-diffusion-sp500-hpc/
    source /iopsstor/scratch/cscs/course_00241/fts-diffusion/fts-diffusion-sp500-hpc/.venv/bin/activate
    python hpc-implementation/scripts/run_downstream_task.py \
      --repo-root "$PWD" \
      --output-dir hpc-implementation/results/smoke \
      --run-id 0 \
      --experiment tatr \
      --epochs 1 \
      --tatr-augmentations 1 \
      --hidden-dim 16
  '

```

Expected output:

```text
hpc-implementation/results/smoke/raw/tatr_run_000.csv
```

## Submit the Full Downstream Jobs

The Slurm scripts are already set to `--account=lsaie-ss26` and
`--partition=normal`.  Adjust only if your course allocation changes.

Important: Slurm resolves `#SBATCH --output` and `#SBATCH --error` before the
script starts. That means the target directory must already exist when you
submit the job. If you move the log location, create the folder first or use a
small submit wrapper that creates it before calling `sbatch`.

Then submit (default `RUN_COUNT=100`):

```bash
sbatch slurm/apls_downstream_array.sbatch
# or override the number of runs:
sbatch --export=RUN_COUNT=50 slurm/apls_downstream_array.sbatch
```

Monitor:

```bash
squeue -u "$USER"
ls -lh /iopsstor/scratch/cscs/$USER/fts-diffusion-sp500-hpc/results/sp500_hpc_full/raw
# view the per-job master srun output or individual run logs:
tail -f /iopsstor/scratch/cscs/$USER/fts-diffusion-sp500-hpc/logs/fts-sp500-downstream_<JOBID>.out
tail -f /iopsstor/scratch/cscs/$USER/fts-diffusion-sp500-hpc/logs/fts-sp500-downstream_<JOBID>/run_0.log
```

Each completed run writes (inside the job's results `raw` folder):

```text
$RESULTS_DIR/raw/tatr_run_000.csv
$RESULTS_DIR/raw/tmtr_run_000.csv
$RESULTS_DIR/raw/task_000_metadata.json
```

The scripts are resumable at the file level.  If a task finished its CSV, a
rerun skips it.

The `.out` logs now follow the same style as the Large-Scale AI assignments:

```text
[sbatch-master] running on ...
[sbatch-master] SLURM_JOB_ID: ...
[srun] rank=0 host=... noderank=... localrank=... array_task=...
[srun] torch=... cuda_available=True cuda_devices=...
[2026-..] [tatr-prices run=0 aug=4] epoch=25/150 loss=...
[2026-..] [tmtr-prices run=0 prop=30] epoch=100/150 loss=...
```

By default, each downstream LSTM fit logs epoch `1`, every `25` epochs, and the
final epoch.  Change `--epoch-log-interval 25` in
`slurm/apls_downstream_array.sbatch` if you want denser or quieter logs.

## Aggregate and Get Plots

After enough array jobs finish:

```bash
sbatch slurm/apls_aggregate_results.sbatch
```

The final files are:

```text
$RESULTS_DIR/tatr_sp500_hpc.png
$RESULTS_DIR/tatr_sp500_hpc.pdf
$RESULTS_DIR/tmtr_sp500_hpc.png
$RESULTS_DIR/tmtr_sp500_hpc.pdf
$RESULTS_DIR/tatr_summary.csv
$RESULTS_DIR/tmtr_summary.csv
$RESULTS_DIR/tatr_matrix.csv
$RESULTS_DIR/tmtr_matrix.csv
$RESULTS_DIR/summary.json
```

Download results:

```bash
rsync -av clariden:/iopsstor/scratch/cscs/$USER/fts-diffusion-sp500-hpc/results/sp500_hpc_full/ \
  hpc-implementation/results/sp500_hpc_full/
```

Optional archive on Clariden after the run:

```bash
mkdir -p /capstor/scratch/cscs/$USER/fts-diffusion-sp500-hpc-archive
rsync -av /iopsstor/scratch/cscs/$USER/fts-diffusion-sp500-hpc/results/sp500_hpc_full/ \
  /capstor/scratch/cscs/$USER/fts-diffusion-sp500-hpc-archive/sp500_hpc_full/
```

## Scaling Notes

The current setup parallelizes across runs, which is the cleanest scaling axis
for TMTR/TATR because each downstream LSTM fit is independent.  One GPU per
array task is enough; requesting more GPUs per task will not help unless the
downstream predictor code is rewritten for distributed training.

To use more of the platform, increase the array size for more runs, or submit
separate arrays for:

- `--ahead 5`
- `--datatype returns`
- `--tatr-protocol continuous_chunked`
- future assets once their PEM/PGM checkpoints and SISC files are added
