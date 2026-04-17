from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from fts_diffusion.config import ExperimentConfig, dump_experiment_config, load_experiment_config
from fts_diffusion.data.datasets import SegmentDataset, TransitionDataset
from fts_diffusion.data.loading import LoadedSeries, load_financial_series
from fts_diffusion.models.autoencoder import ScalingAutoencoder
from fts_diffusion.models.diffusion import ConditionalDiffusionModel
from fts_diffusion.models.evolution import PatternEvolutionNetwork
from fts_diffusion.models.sisc import SISC, SISCResult
from fts_diffusion.utils.io import ensure_dir, load_json, save_json
from fts_diffusion.utils.random import seed_everything


class PatternGenerator(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        fixed_length = int(config.generation.fixed_length or config.pattern.centroid_length or config.pattern.max_length)
        self.fixed_length = fixed_length
        self.autoencoder = ScalingAutoencoder(
            fixed_length=fixed_length,
            hidden_dim=config.generation.hidden_dim,
            rnn_layers=config.generation.rnn_layers,
            rnn_type=config.generation.rnn_type,
        )
        self.diffusion = ConditionalDiffusionModel(
            fixed_length=fixed_length,
            channels=config.generation.tcn_channels,
            residual_blocks=config.generation.residual_blocks,
            kernel_size=config.generation.kernel_size,
            diffusion_steps=config.generation.diffusion_steps,
        )

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        fixed, reconstruction = self.autoencoder(batch["normalized"], batch["length"], batch["beta"])
        reconstruction_loss = self._masked_mse(reconstruction, batch["raw"], batch["length"])

        timesteps = torch.randint(
            low=0,
            high=self.diffusion.diffusion_steps,
            size=(fixed.shape[0],),
            device=fixed.device,
        )
        noise = torch.randn_like(fixed)
        noisy = self.diffusion.q_sample(fixed, timesteps, noise)
        noise_prediction = self.diffusion(noisy, timesteps, batch["pattern"])
        diffusion_loss = F.mse_loss(noise_prediction, noise)
        total = reconstruction_loss + diffusion_loss
        metrics = {
            "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
            "diffusion_loss": float(diffusion_loss.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
        }
        return total, metrics

    def sample_segment(self, pattern: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        fixed = self.diffusion.sample(pattern=pattern, device=device)
        lengths = torch.clamp((alpha * self.fixed_length).round().long(), min=1)
        decoded = self.autoencoder.decode(fixed, lengths, beta)
        outputs = []
        for index, length in enumerate(lengths):
            outputs.append(decoded[index, : int(length.item())])
        return outputs[0] if len(outputs) == 1 else torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True)

    @staticmethod
    def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        horizon = prediction.shape[1]
        mask = torch.arange(horizon, device=prediction.device).unsqueeze(0) < lengths.unsqueeze(1)
        return ((prediction - target) ** 2).masked_select(mask).mean()


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _save_sisc_artifacts(run_dir: Path, sisc_result: SISCResult, series: np.ndarray) -> None:
    patterns_path = run_dir / "patterns.npy"
    np.save(patterns_path, sisc_result.patterns)
    save_json(sisc_result.to_dict(), run_dir / "sisc_result.json")
    segment_bank = []
    for segment in sisc_result.segments:
        raw = series[segment.start : segment.end].astype(np.float32).tolist()
        segment_bank.append(
            {
                "raw": raw,
                "cluster_id": segment.cluster_id,
                "alpha": segment.alpha,
                "beta": segment.beta,
            }
        )
    save_json(segment_bank, run_dir / "segment_bank.json")


def _train_pattern_generator(
    config: ExperimentConfig,
    run_dir: Path,
    dataset: SegmentDataset,
    device: torch.device,
) -> PatternGenerator:
    model = PatternGenerator(config).to(device)
    loader = DataLoader(
        dataset,
        batch_size=config.generation.batch_size,
        shuffle=True,
        num_workers=config.runtime.num_workers,
        collate_fn=SegmentDataset.collate_fn,
    )
    optimizer = Adam(
        model.parameters(),
        lr=config.generation.learning_rate,
        weight_decay=config.generation.weight_decay,
    )
    history: list[dict[str, float]] = []

    for epoch in range(1, config.generation.epochs + 1):
        model.train()
        epoch_metrics = {"reconstruction_loss": 0.0, "diffusion_loss": 0.0, "total_loss": 0.0}
        progress = tqdm(loader, desc=f"generator epoch {epoch}", leave=False)
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = model.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.generation.grad_clip_norm)
            optimizer.step()
            for key, value in metrics.items():
                epoch_metrics[key] += value
            progress.set_postfix(loss=metrics["total_loss"])

        for key in epoch_metrics:
            epoch_metrics[key] /= max(len(loader), 1)
        history.append({"epoch": epoch, **epoch_metrics})

        if epoch % config.runtime.checkpoint_every == 0 or epoch == config.generation.epochs:
            torch.save(
                {"model_state": model.state_dict(), "config": asdict(config.generation)},
                run_dir / "checkpoints" / f"generator_epoch_{epoch}.pt",
            )

    torch.save(
        {"model_state": model.state_dict(), "config": asdict(config.generation)},
        run_dir / "checkpoints" / "generator_final.pt",
    )
    save_json(history, run_dir / "generator_history.json")
    return model


def _train_evolution_model(
    config: ExperimentConfig,
    run_dir: Path,
    dataset: TransitionDataset,
    device: torch.device,
) -> PatternEvolutionNetwork:
    model = PatternEvolutionNetwork(
        num_patterns=config.pattern.num_patterns,
        hidden_dim=config.evolution.hidden_dim,
        num_layers=config.evolution.num_layers,
        dropout=config.evolution.dropout,
    ).to(device)
    loader = DataLoader(
        dataset,
        batch_size=config.evolution.batch_size,
        shuffle=True,
        num_workers=config.runtime.num_workers,
    )
    optimizer = Adam(
        model.parameters(),
        lr=config.evolution.learning_rate,
        weight_decay=config.evolution.weight_decay,
    )
    history: list[dict[str, float]] = []

    for epoch in range(1, config.evolution.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(loader, desc=f"evolution epoch {epoch}", leave=False)
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits, alpha_prediction, beta_prediction = model(
                batch["current_pattern"], batch["current_alpha"], batch["current_beta"]
            )
            loss = (
                F.cross_entropy(logits, batch["next_pattern"])
                + F.mse_loss(alpha_prediction, batch["next_alpha"])
                + F.mse_loss(beta_prediction, batch["next_beta"])
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.evolution.grad_clip_norm)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            progress.set_postfix(loss=float(loss.detach().cpu()))

        epoch_loss /= max(len(loader), 1)
        history.append({"epoch": epoch, "loss": epoch_loss})
        if epoch % config.runtime.checkpoint_every == 0 or epoch == config.evolution.epochs:
            torch.save(
                {"model_state": model.state_dict(), "config": asdict(config.evolution)},
                run_dir / "checkpoints" / f"evolution_epoch_{epoch}.pt",
            )

    torch.save(
        {"model_state": model.state_dict(), "config": asdict(config.evolution)},
        run_dir / "checkpoints" / "evolution_final.pt",
    )
    save_json(history, run_dir / "evolution_history.json")
    return model


def train_from_config(config: ExperimentConfig) -> Path:
    seed_everything(config.runtime.seed)
    device = _resolve_device(config.runtime.device)
    run_dir = ensure_dir(config.runtime.output_dir)
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "samples")
    dump_experiment_config(config, run_dir / "resolved_config.yaml")

    loaded = load_financial_series(config.data)
    save_json(
        {
            "train_mean": loaded.train_mean,
            "train_std": loaded.train_std,
            "train_length": int(len(loaded.train_values)),
            "test_length": int(len(loaded.test_values)),
        },
        run_dir / "series_stats.json",
    )

    sisc = SISC(config.pattern, rng=np.random.default_rng(config.runtime.seed))
    sisc_result = sisc.fit(loaded.train_values)
    _save_sisc_artifacts(run_dir, sisc_result, loaded.train_values)

    segment_dataset = SegmentDataset(loaded.train_values, sisc_result)
    generator = _train_pattern_generator(config, run_dir, segment_dataset, device)

    transition_dataset = TransitionDataset(sisc_result)
    if len(transition_dataset) == 0:
        raise ValueError("Transition dataset is empty. Increase the series length or decrease segment length.")
    _train_evolution_model(config, run_dir, transition_dataset, device)

    torch.save({"train_values": loaded.train_values}, run_dir / "train_series.pt")
    torch.save({"generator_state": generator.state_dict()}, run_dir / "checkpoints" / "generator_bundle.pt")
    return Path(run_dir)


def _load_bundle(run_dir: str | Path) -> tuple[ExperimentConfig, np.ndarray, list[dict], PatternGenerator, PatternEvolutionNetwork, torch.device]:
    run_dir = Path(run_dir)
    config = load_experiment_config(run_dir / "resolved_config.yaml")
    patterns = np.load(run_dir / "patterns.npy").astype(np.float32)
    segment_bank = load_json(run_dir / "segment_bank.json")
    device = _resolve_device(config.runtime.device)

    generator = PatternGenerator(config).to(device)
    generator_ckpt = torch.load(run_dir / "checkpoints" / "generator_final.pt", map_location=device)
    generator.load_state_dict(generator_ckpt["model_state"])
    generator.eval()

    evolution = PatternEvolutionNetwork(
        num_patterns=config.pattern.num_patterns,
        hidden_dim=config.evolution.hidden_dim,
        num_layers=config.evolution.num_layers,
        dropout=config.evolution.dropout,
    ).to(device)
    evolution_ckpt = torch.load(run_dir / "checkpoints" / "evolution_final.pt", map_location=device)
    evolution.load_state_dict(evolution_ckpt["model_state"])
    evolution.eval()
    return config, patterns, segment_bank, generator, evolution, device


@torch.no_grad()
def sample_from_run(run_dir: str | Path, terminal_length: int | None = None, output_path: str | Path | None = None) -> np.ndarray:
    config, patterns, segment_bank, generator, evolution, device = _load_bundle(run_dir)
    terminal_length = terminal_length or config.sampling.default_terminal_length
    rng = np.random.default_rng(config.runtime.seed)
    seed_item = segment_bank[int(rng.integers(0, len(segment_bank)))]

    generated: list[float] = list(seed_item["raw"])
    current_pattern = torch.tensor([seed_item["cluster_id"]], dtype=torch.long, device=device)
    current_alpha = torch.tensor([seed_item["alpha"]], dtype=torch.float32, device=device)
    current_beta = torch.tensor([seed_item["beta"]], dtype=torch.float32, device=device)

    while len(generated) < terminal_length:
        next_pattern, next_alpha, next_beta = evolution.sample_next_state(
            current_pattern,
            current_alpha,
            current_beta,
            temperature=config.sampling.temperature,
            alpha_noise=config.sampling.alpha_noise,
            beta_noise=config.sampling.beta_noise,
            min_beta=config.sampling.min_beta,
        )
        pattern_tensor = torch.tensor(patterns[next_pattern.cpu().numpy()], dtype=torch.float32, device=device)
        segment = generator.sample_segment(pattern_tensor, next_alpha, next_beta)
        generated.extend(segment.detach().cpu().tolist())
        current_pattern, current_alpha, current_beta = next_pattern, next_alpha, next_beta

    output = np.asarray(generated[:terminal_length], dtype=np.float32)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_path, output, delimiter=",")
    return output


def train_from_path(config_path: str | Path) -> Path:
    return train_from_config(load_experiment_config(config_path))

