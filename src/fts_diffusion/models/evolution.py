from __future__ import annotations

import torch
from torch import nn


class PatternEvolutionNetwork(nn.Module):
    def __init__(self, num_patterns: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.pattern_embedding = nn.Embedding(num_patterns, hidden_dim)
        layers = []
        input_dim = hidden_dim + 2
        for layer_index in range(num_layers):
            layers.append(nn.Linear(input_dim if layer_index == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)
        self.pattern_head = nn.Linear(hidden_dim, num_patterns)
        self.alpha_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(
        self, pattern_ids: torch.Tensor, alphas: torch.Tensor, betas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.pattern_embedding(pattern_ids)
        features = torch.cat([embedded, alphas.unsqueeze(1), torch.log1p(betas).unsqueeze(1)], dim=1)
        hidden = self.backbone(features)
        logits = self.pattern_head(hidden)
        alpha_prediction = self.alpha_head(hidden).squeeze(1)
        beta_prediction = self.beta_head(hidden).squeeze(1)
        return logits, alpha_prediction, beta_prediction

    @torch.no_grad()
    def sample_next_state(
        self,
        pattern_id: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        temperature: float,
        alpha_noise: float,
        beta_noise: float,
        min_beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, alpha_prediction, beta_prediction = self.forward(pattern_id, alpha, beta)
        probabilities = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        next_pattern = torch.multinomial(probabilities, num_samples=1).squeeze(1)
        next_alpha = torch.clamp(alpha_prediction + alpha_noise * torch.randn_like(alpha_prediction), min=0.25)
        next_beta = torch.clamp(beta_prediction + beta_noise * torch.randn_like(beta_prediction), min=min_beta)
        return next_pattern, next_alpha, next_beta
