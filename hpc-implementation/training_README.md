cd /iopsstor/scratch/cscs/$USER/fts-diffusion/fts-diffusion-sp500-hpc
source .venv/bin/activate
python hpc-implementation/scripts/train_sp500.py --repo-root "$PWD" --store-model --debugTraining SP500 PGM/PEM on Alps (H100)
===================================

This folder contains helper scripts and an Slurm template to train the PGM
(pattern generation) and PEM (pattern evolution) models for S&P500 on the Alps
cluster. The implementation uses the reference training code in
`fts-diffusion-ref` and preserves the repository layout expected by the
reference scripts.

Files added:
- `scripts/train_sp500.py`: Python wrapper that downloads the S&P500 series
  (if missing) and calls the reference `train_ftsdiffusion` entrypoint.
- `slurm/train_sp500.sbatch`: Slurm job template (1 node, 1 GPU, 32 CPUs,
  48h). Adjust `--cpus-per-task`, `--time`, and GPU request for your queue.

Usage
-----

Create the venv and install clariden requirements as described in
`hpc-implementation/README.md`, then submit the training job:

```bash
source .venv/bin/activate
sbatch slurm/train_sp500.sbatch
# or run interactively for debugging
srun --partition=debug --gpus-per-node=1 --cpus-per-task=8 --time=02:00:00 --pty bash -lc '
  source .venv/bin/activate; python hpc-implementation/scripts/train_sp500.py --repo-root "$PWD" --store-model --debug'
```

Notes and recommendations
-------------------------
- The wrapper uses the authors' default training parameters in
  `fts-diffusion-ref/models/model_params.py` (PGM: n_epochs=30, PEM: n_epochs=60).
- H100 GPUs will significantly reduce wall time; request 1 GPU per job.
- Start with `--debug` for a 1-epoch smoke test to verify environment and runtime.
- The training writes checkpoints to `fts-diffusion-ref/trained_models/`.
