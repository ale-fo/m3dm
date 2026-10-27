from __future__ import annotations
from typing import Optional

import torch
from .point_transformer import PointTransformer


NO_BACKBONE = 0
POINT_MAE = 1
POINT_BERT = 2

POINT_BACKBONES = {NO_BACKBONE: "None", POINT_MAE: "Point_MAE", POINT_BERT: "Point_Bert"}


def strip_module_prefix(key: str) -> str:
    return key.replace("module.", "")


def point_mae_state(state: dict) -> dict:
    backbone_state = {}
    for k, v in state.items():
        k = strip_module_prefix(k)
        if k.startswith("MAE_encoder."):
            k = k[len("MAE_encoder."):]
        backbone_state[k] = v
    return backbone_state


def point_bert_state(state: dict) -> dict:
    backbone_state = {}
    for k, v in state.items():
        k = strip_module_prefix(k)
        if k.startswith("transformer_q.") and not k.startswith("transformer_q.cls_head"):
            backbone_state[k[len("transformer_q."):]] = v
    return backbone_state


def load_point_backbone(backbone: int, checkpoint_path: Optional[str], device: torch.device,
                        group_size: int = 128, num_group: int = 1024) -> Optional[PointTransformer]:
    if backbone == NO_BACKBONE:
        return None
    name = POINT_BACKBONES[backbone]

    if backbone == POINT_MAE:
        model = PointTransformer(group_size = group_size, num_group = num_group, encoder_dims = 384)
    else:
        model = PointTransformer(group_size = group_size, num_group = num_group, encoder_dims = 256)

    checkpoint = torch.load(checkpoint_path, map_location = "cpu")
    state = checkpoint["base_model"]
    if backbone == POINT_MAE:
        backbone_state = point_mae_state(state)
    else:
        backbone_state = point_bert_state(state)

    result = model.load_state_dict(backbone_state, strict = False)
    n_matched = len(backbone_state) - len(result.unexpected_keys)
    if n_matched:
        print(f"{name} checkpoint loaded from {checkpoint_path}")
    else:
        print(f"WARNING: no weights matched in {checkpoint_path}")

    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model
