# FTS-Diffusion

Paper-faithful implementation of **FTS-Diffusion** from:

- Hongbin Huang, Minghua Chen, Xiao Qiao. *Generative Learning for Financial Time Series with Irregular and Scale-Invariant Patterns*. ICLR 2024.
- OpenReview: <https://openreview.net/forum?id=CdjnzWsQax>
- ICLR proceedings page: <https://proceedings.iclr.cc/paper_files/paper/2024/hash/f90fc76b199fe6b0ec2a51aaf72c3277-Abstract-Conference.html>

I could not locate an official public repository for the paper from the paper pages or common curated lists, so this repo implements the method directly from the paper and appendix.

## What Is Implemented

The repository follows the paper's three-module decomposition:

1. **Pattern recognition** via SISC:
   - K-means-style initialization over candidate subsequences
   - greedy variable-length segmentation
   - DTW-based matching in normalized space
   - iterative centroid updates

2. **Pattern generation**:
   - scaling autoencoder with 2-layer GRU/LSTM option
   - pattern-conditioned DDPM-style temporal convolutional network
   - joint training of reconstruction and diffusion loss

3. **Pattern evolution**:
   - Markov-style state transition network over `(pattern, alpha, beta)`
   - classification head for next pattern
   - regression heads for duration and magnitude scales

4. **End-to-end workflow**:
   - data loading and return transformation
   - chronological train/test split
   - artifact export for segmented subsequences and learned patterns
   - checkpointed training
   - standalone synthetic sampling script

## Repo Layout

```text
configs/default.yaml               Default experiment config
scripts/train_fts_diffusion.py     Training entrypoint
scripts/sample_fts_diffusion.py    Sampling entrypoint
src/fts_diffusion/config.py        Structured experiment config
src/fts_diffusion/data/            Data loading and datasets
src/fts_diffusion/models/sisc.py   SISC pattern recognition
src/fts_diffusion/models/autoencoder.py
src/fts_diffusion/models/diffusion.py
src/fts_diffusion/models/evolution.py
src/fts_diffusion/training/pipeline.py
```

## Environment

Use a Python version supported by PyTorch. The project is pinned for `Python >=3.10,<3.13`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Expected Data Format

The paper trains on **univariate daily return series** derived from asset close prices. This repo supports:

- `input_type: close`
  - reads a raw price column and converts it to simple returns
- `input_type: log_return`
  - reads a raw price column and converts it to log returns
- `input_type: return`
  - assumes the provided column is already a return series
- `input_type: value`
  - consumes the provided numeric column as-is

Minimum CSV example:

```csv
date,close
2005-01-03,101.52
2005-01-04,102.11
2005-01-05,100.88
```

If you already have returns:

```csv
date,return
2005-01-04,0.00581
2005-01-05,-0.01205
```

## Quick Start

1. Put your file at `data/asset.csv` or edit `configs/default.yaml`.
2. Check the config values for:
   - `data.value_column`
   - `data.input_type`
   - `pattern.num_patterns`
   - `runtime.output_dir`
3. Train:

```bash
python scripts/train_fts_diffusion.py --config configs/default.yaml
```

4. Sample a synthetic series from the trained run:

```bash
python scripts/sample_fts_diffusion.py \
  --run-dir outputs/default \
  --length 2048 \
  --output outputs/default/samples/generated.csv
```

You can also use the installed console scripts:

```bash
fts-train --config configs/default.yaml
fts-sample --run-dir outputs/default --length 2048 --output outputs/default/samples/generated.csv
```

## Training Artifacts

Each run writes:

- `resolved_config.yaml`
- `series_stats.json`
- `patterns.npy`
- `sisc_result.json`
- `segment_bank.json`
- `generator_history.json`
- `evolution_history.json`
- `checkpoints/generator_final.pt`
- `checkpoints/evolution_final.pt`

## Paper Details Reflected In Config

The appendix specifies the following defaults used in the paper and mirrored in the config:

- segment length range: `10` to `21`
- diffusion steps: `100`
- diffusion network: `6` residual TCN blocks
- scaling AE: `2` recurrent layers
- generator optimizer: Adam with learning rate `5e-4`, batch size `32`
- evolution optimizer: Adam with learning rate `4e-4`, trained for `1000` epochs

## Replication Notes

Some implementation details are omitted or underspecified in the paper. This code keeps those choices explicit and localized:

- **SISC centroid update**
  - The paper gives the segmentation and assignment logic but not a DTW barycenter update rule.
  - This repo updates centroids by resampling normalized cluster members to the common centroid length and averaging them.

- **Scaling autoencoder stretch/compress operator**
  - The paper says the AE can be implemented with 2-layer LSTMs or GRUs but does not specify the exact resizing operator.
  - This repo uses linear time-axis resampling around the recurrent encoder/decoder to realize the variable-length to fixed-length mapping.

- **Diffusion equation ambiguity**
  - The main text reuses `beta` both for magnitude and diffusion variance.
  - The implementation follows the standard DDPM equations given in Appendix C.1 and treats segment magnitude as a separate transition state for generation.

- **Return-series assumption**
  - The original experiments are on asset returns, not raw prices.
  - The default pipeline therefore converts close prices to returns before training.

These choices are deliberate, documented, and easy to swap if you later obtain author code or further details.

## Recommended First Real Run

Once your course dataset is ready:

1. Start with one asset and confirm the full pipeline runs.
2. Inspect `patterns.npy` and `sisc_result.json` to verify segment lengths and cluster usage are sensible.
3. Sample a few synthetic sequences and compare their marginal distribution and visual patterns to the observed returns.
4. Only then tune:
   - `pattern.num_patterns`
   - `pattern.max_iters`
   - `generation.epochs`
   - `sampling.temperature`

## Limitations

- The repo does not include the paper's downstream forecasting evaluation yet.
- No baseline implementations are included.
- I could not run a full training smoke test in this environment because the workspace currently lacks PyTorch and related dependencies.

