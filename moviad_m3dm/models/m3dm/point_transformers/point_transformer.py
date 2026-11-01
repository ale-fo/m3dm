from __future__ import annotations

import torch
import torch.nn as nn

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype = torch.long, device = device)\
        .view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype = torch.long, device = device)
    distance = torch.full((B, N), 1e10, device = device)
    farthest = torch.zeros(B, dtype = torch.long, device = device)
    batch_idx = torch.arange(B, dtype = torch.long, device = device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim = -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim = -1)[1]
    return centroids

def knn_points(xyz: torch.Tensor, center: torch.Tensor, k: int) -> torch.Tensor:
    dists = square_distance(center, xyz)
    idx = dists.topk(k, dim = -1, largest = False).indices
    return idx

def fps(data: torch.Tensor, number: int):
    fps_idx = farthest_point_sample(data, number)
    fps_data = index_points(data, fps_idx)
    return fps_data, fps_idx

def interpolate_features(
    xyz: torch.Tensor, center: torch.Tensor, features: torch.Tensor, k: int = 3, chunk_size: int = 8192,
) -> torch.Tensor:
    B, N, _ = xyz.shape
    D = features.shape[1]
    features_t = features.permute(0, 2, 1)

    chunks = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        n = end - start

        dists = square_distance(xyz[:, start:end, :], center)
        dists, idx = dists.topk(k, dim = -1, largest = False)

        weights = 1.0 / (dists + 1e-8)
        weights = weights / weights.sum(dim = -1, keepdim = True)

        idx_exp = idx.unsqueeze(-1).expand(B, n, k, D)
        feat_exp = features_t.unsqueeze(1).expand(B, n, -1, D)
        neighbor_feats = feat_exp.gather(2, idx_exp)

        chunks.append((weights.unsqueeze(-1) * neighbor_feats).sum(dim = 2))

    interp = torch.cat(chunks, dim = 1)
    return interp.permute(0, 2, 1)

class Group(nn.Module):
    def __init__(self, num_group: int, group_size: int) -> None:
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz: torch.Tensor):
        B, N, _ = xyz.shape

        center, center_idx = fps(xyz.contiguous(), self.num_group)

        neighbor_idx = knn_points(xyz, center, self.group_size)

        idx_base = torch.arange(B, device = xyz.device).view(-1, 1, 1) * N
        idx_flat = (neighbor_idx + idx_base).view(-1)
        neighborhood = xyz.reshape(B * N, 3)[idx_flat]\
            .reshape(B, self.num_group, self.group_size, 3).contiguous()

        neighborhood = neighborhood - center.unsqueeze(2)

        return neighborhood, center, neighbor_idx, center_idx

class Encoder(nn.Module):
    def __init__(self, encoder_channel: int) -> None:
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)

        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim = 2, keepdim = True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim = 1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim = 2, keepdim = False)[0]

        return feature_global.reshape(bs, g, self.encoder_channel)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None,
                 act_layer = nn.GELU, drop = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features) # hidden layer
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features) # output layer
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, qk_scale = None,
                 attn_dropout = 0, proj_drop = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)\
            .permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 4., qkv_bias = False, qk_scale = None,
                 drop = 0.0, attn_dropout = 0.0, drop_path = 0.0,
                 act_layer = nn.GELU, norm_layer = nn.LayerNorm) -> None:
        super().__init__()
        from timm.models.layers import DropPath
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.attn = Attention(dim, num_heads = num_heads, qkv_bias = qkv_bias,
                                   qk_scale = qk_scale, attn_dropout = attn_dropout, proj_drop = drop)
        self.mlp = Mlp(in_features = dim, hidden_features = int(dim * mlp_ratio), act_layer = act_layer, drop = drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim = 768, depth = 4, num_heads = 12, mlp_ratio = 4.0,
                 qkv_bias = False, qk_scale = None, drop_rate = 0.0,
                 attn_dropout_rate = 0.0, drop_path_rate = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if isinstance(drop_path_rate, list):
                block_drop_path = drop_path_rate[i]
            else:
                block_drop_path = drop_path_rate
            self.blocks.append(
                Block(dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio,
                      qkv_bias = qkv_bias, qk_scale = qk_scale, drop = drop_rate,
                      attn_dropout = attn_dropout_rate, drop_path = block_drop_path))

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list

class PointTransformer(nn.Module):
    def __init__(self, group_size: int = 128, num_group: int = 1024,
                 encoder_dims: int = 384) -> None:
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.num_heads = 6
        self.group_size = group_size
        self.num_group = num_group
        self.encoder_dims = encoder_dims

        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        if self.encoder_dims != self.trans_dim:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
            self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim))

        dpr = []
        for x in torch.linspace(0, self.drop_path_rate, self.depth):
            dpr.append(x.item())
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim, depth = self.depth,
            drop_path_rate = dpr, num_heads = self.num_heads,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

    def forward(self, pts: torch.Tensor):
        B = pts.shape[0]
        pts = pts.transpose(-1, -2)

        neighborhood, center, neighbor_idx, center_idx = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)

        feature_list = []
        if self.encoder_dims != self.trans_dim:
            group_input_tokens = self.reduce_dim(group_input_tokens)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_pos = self.cls_pos.expand(B, -1, -1)
            pos = self.pos_embed(center)
            x = torch.cat((cls_tokens, group_input_tokens), dim = 1)
            pos = torch.cat((cls_pos, pos), dim = 1)
            for f in self.blocks(x, pos):
                feature_list.append(self.norm(f)[:, 1:].transpose(-1, -2).contiguous())
        else:
            pos = self.pos_embed(center)
            for f in self.blocks(group_input_tokens, pos):
                feature_list.append(self.norm(f).transpose(-1, -2).contiguous())

        x = torch.cat([feature_list[0], feature_list[1], feature_list[2]], dim = 1)
        return x, center, neighbor_idx, center_idx