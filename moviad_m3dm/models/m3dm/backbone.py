from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from moviad.models.m3dm.point_transformers.point_transformer import interpolate_features


def build_vit(name: str, device: torch.device) -> nn.Module:
    model = timm.create_model(name, pretrained = True, num_classes = 0)
    print(f"backbone downloaded from HuggingFace")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_vit_features(vit: nn.Module, x: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    tokens = vit.forward_features(x)
    patch_tokens = tokens[:, 1:, :]
    N = patch_tokens.shape[1]
    H_p = int(N ** 0.5)
    return patch_tokens.permute(0, 2, 1).reshape(B, -1, H_p, H_p)


def extract_point_cloud_features(
    point_backbone: nn.Module,
    organized_pc: torch.Tensor,
    device: torch.device,
    point_grid_size: int = 56,
) -> torch.Tensor:
    B, _, H, W = organized_pc.shape
    pc = organized_pc.to(device)

    feat_maps: list = []

    for i in range(B):
        pc_i = pc[i : i + 1]
        flat_points = pc_i.reshape(1, 3, H * W).permute(0, 2, 1)
        valid_idx = (flat_points[0] != 0).all(-1).nonzero(as_tuple = True)[0]

        if valid_idx.shape[0] == 0:
            raise ValueError("point cloud without valid (nonzero) points")

        valid_points = flat_points[:, valid_idx, :].permute(0, 2, 1).contiguous()
        tokens, center, _, _ = point_backbone(valid_points)
        D_pt = tokens.shape[1]

        query_pts = flat_points[:, valid_idx, :]
        interp = interpolate_features(xyz = query_pts, center = center, features = tokens, k = 3).squeeze(0)

        dense_features = torch.zeros(D_pt, H * W, device = device)
        dense_features[:, valid_idx] = interp
        feat_maps.append(dense_features.reshape(D_pt, H, W))

    feat_map = torch.stack(feat_maps, dim = 0)

    smoothed = F.avg_pool2d(feat_map, kernel_size = 3, stride = 1)
    return F.adaptive_avg_pool2d(smoothed, (point_grid_size, point_grid_size))