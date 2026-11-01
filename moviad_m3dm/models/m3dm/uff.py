from __future__ import annotations

import torch
import torch.nn as nn


class Mlp(nn.Module):

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class UFF(nn.Module):
    def __init__(
        self,
        rgb_dim: int = 768,
        pt_dim: int = 1152,
        proj_dim: int = 256,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.rgb_dim = rgb_dim
        self.pt_dim = pt_dim

        self.xyz_norm = nn.LayerNorm(pt_dim)
        self.xyz_mlp = Mlp(pt_dim, int(pt_dim * mlp_ratio))
        self.rgb_norm = nn.LayerNorm(rgb_dim)
        self.rgb_mlp = Mlp(rgb_dim, int(rgb_dim * mlp_ratio))

        self.rgb_head = nn.Linear(rgb_dim, proj_dim)
        self.xyz_head = nn.Linear(pt_dim, proj_dim)

    def forward(self, rgb_feat: torch.Tensor, pt_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.shape[-2:] != pt_feat.shape[-2:]:
            raise ValueError(f"UFF  requires RGB and point features on the same grid")
        B, _, H, W = pt_feat.shape

        rgb = rgb_feat.flatten(2).transpose(1, 2)
        pt = pt_feat.flatten(2).transpose(1, 2)

        pt = self.xyz_mlp(self.xyz_norm(pt))
        rgb = self.rgb_mlp(self.rgb_norm(rgb))

        fused = torch.cat([pt, rgb], dim = -1)
        return fused.transpose(1, 2).reshape(B, self.pt_dim + self.rgb_dim, H, W)
