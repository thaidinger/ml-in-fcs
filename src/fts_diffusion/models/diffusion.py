from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device) * exponent)
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.embedding_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
        return embedding


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, time_dim: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.time_projection = nn.Linear(time_dim, channels)
        self.condition_projection = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.activation = nn.SiLU()
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)
        h = h + self.time_projection(time_emb).unsqueeze(-1)
        h = h + self.condition_projection(condition)
        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)
        return residual + h


class ConditionalDiffusionModel(nn.Module):
    def __init__(
        self,
        fixed_length: int,
        channels: int,
        residual_blocks: int,
        kernel_size: int,
        diffusion_steps: int,
    ) -> None:
        super().__init__()
        self.fixed_length = fixed_length
        self.diffusion_steps = diffusion_steps
        time_dim = channels * 2
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_projection = nn.Conv1d(1, channels, kernel_size=1)
        self.condition_projection = nn.Conv1d(1, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualTCNBlock(channels, kernel_size=kernel_size, time_dim=time_dim) for _ in range(residual_blocks)]
        )
        self.output_projection = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

        betas = torch.linspace(1e-4, 0.02, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(noisy.unsqueeze(1))
        condition = self.condition_projection(pattern.unsqueeze(1))
        time_emb = self.time_embedding(timesteps)
        for block in self.blocks:
            x = block(x, time_emb, condition)
        return self.output_projection(x).squeeze(1)

    def q_sample(self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        scale = self.sqrt_alphas_cumprod[timesteps].unsqueeze(1)
        noise_scale = self.sqrt_one_minus_alphas_cumprod[timesteps].unsqueeze(1)
        return scale * clean + noise_scale * noise

    @torch.no_grad()
    def sample(self, pattern: torch.Tensor, device: torch.device) -> torch.Tensor:
        batch_size = pattern.shape[0]
        current = torch.randn(batch_size, self.fixed_length, device=device)
        for timestep in reversed(range(self.diffusion_steps)):
            t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            noise_prediction = self.forward(current, t, pattern)
            alpha = self.alphas[t].unsqueeze(1)
            alpha_bar = self.alphas_cumprod[t].unsqueeze(1)
            beta = self.betas[t].unsqueeze(1)
            mean = (current - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * noise_prediction) / torch.sqrt(alpha)
            if timestep > 0:
                current = mean + torch.sqrt(beta) * torch.randn_like(current)
            else:
                current = mean
        return current

