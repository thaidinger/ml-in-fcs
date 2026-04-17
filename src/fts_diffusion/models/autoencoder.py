from __future__ import annotations

import torch
from torch import nn

from fts_diffusion.utils.interpolation import batch_resample_1d


def _build_rnn(rnn_type: str, input_size: int, hidden_dim: int, layers: int) -> nn.Module:
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        return nn.GRU(input_size, hidden_dim, num_layers=layers, batch_first=True)
    if rnn_type == "lstm":
        return nn.LSTM(input_size, hidden_dim, num_layers=layers, batch_first=True)
    raise ValueError(f"Unsupported rnn_type={rnn_type}")


class ScalingAutoencoder(nn.Module):
    """
    Learns the variable-length <-> fixed-length mapping described in the paper.

    The paper does not fully specify the mechanics of the stretch/compress operator, so this
    implementation uses linear time-axis resampling wrapped with the requested two-layer GRU/LSTM.
    """

    def __init__(self, fixed_length: int, hidden_dim: int, rnn_layers: int, rnn_type: str) -> None:
        super().__init__()
        self.fixed_length = fixed_length
        self.encoder_rnn = _build_rnn(rnn_type, input_size=1, hidden_dim=hidden_dim, layers=rnn_layers)
        self.encoder_projection = nn.Linear(hidden_dim, 1)

        self.decoder_rnn = _build_rnn(rnn_type, input_size=3, hidden_dim=hidden_dim, layers=rnn_layers)
        self.decoder_projection = nn.Linear(hidden_dim, 1)

    def encode(self, normalized_segments: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        resized = batch_resample_1d(
            [normalized_segments[index, : int(length.item())] for index, length in enumerate(lengths)],
            target_length=self.fixed_length,
        )
        encoded, _ = self.encoder_rnn(resized.unsqueeze(-1))
        fixed = self.encoder_projection(encoded).squeeze(-1)
        return fixed

    def decode(self, fixed_representation: torch.Tensor, lengths: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
        target_length = int(lengths.max().item())
        resized = batch_resample_1d(
            [fixed_representation[index] for index in range(fixed_representation.shape[0])],
            target_length=target_length,
        )
        time_grid = torch.linspace(0.0, 1.0, target_length, device=fixed_representation.device)
        time_grid = time_grid.unsqueeze(0).expand(fixed_representation.shape[0], -1)
        beta_feature = torch.log1p(betas).unsqueeze(1).expand(-1, target_length)
        decoder_input = torch.stack([resized, time_grid, beta_feature], dim=-1)
        decoded, _ = self.decoder_rnn(decoder_input)
        normalized = self.decoder_projection(decoded).squeeze(-1)
        return normalized * betas.unsqueeze(1)

    def forward(
        self, normalized_segments: torch.Tensor, lengths: torch.Tensor, betas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fixed = self.encode(normalized_segments, lengths)
        reconstruction = self.decode(fixed, lengths, betas)
        return fixed, reconstruction

