from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs
from moviad.models.patchcore.kcenter_greedy import CoresetExtractor

from moviad.models.m3dm.anomaly_map import AnomalyMapGenerator
from moviad.models.m3dm.backbone import build_vit, extract_vit_features, extract_point_cloud_features
from moviad.models.m3dm.uff import UFF
from moviad.models.m3dm.dlf import (
    DeepLatentFusion,
    nn_scores,
    reweighted_image_score,
)


class M3DM(VADModel):

    def __init__(
        self,
        device,
        input_size = (224, 224),
        rgb_backbone_name = "vit_base_patch8_224.dino",
        point_backbone = None,
        mode = "rgb",
        uff_checkpoint = None,
        rgb_dim = 768,
        pt_dim = 1152,
        f_coreset: float = 0.1,
        ocsvm_nu = 0.5,
        ocsvm_maxiter = 1000,
        max_train_samples: int = 400,
        point_grid_size = 56,
        rgb_image_weight = 0.1,
        rgb_pixel_weight = 0.1,
        xyz_image_weight = 1.0,
        xyz_pixel_weight = 1.0,
        fusion_image_weight = 1.0,
        fusion_pixel_weight = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if mode not in ("rgb", "3d", "fusion"):
            raise ValueError("mode must be 'rgb', '3d', or 'fusion'")
        if mode != "rgb" and point_backbone is None:
            raise ValueError("point_backbone is required for '3d' and 'fusion' modes")
        if mode == "fusion" and uff_checkpoint is None:
            raise ValueError("'fusion' mode needs uff checkpoint")

        self.device = device
        self.input_size = input_size
        self.mode = mode
        self.f_coreset = f_coreset
        self.num_neighbors = 3
        self.max_train_samples = max_train_samples
        self.point_grid_size = point_grid_size
        self._point_grid_resize = torch.nn.AdaptiveAvgPool2d((point_grid_size, point_grid_size))
        self.modality_weights = {
            "rgb": (rgb_image_weight, rgb_pixel_weight),
            "pt": (xyz_image_weight, xyz_pixel_weight),
            "fs": (fusion_image_weight, fusion_pixel_weight),
        }

        self.rgb_backbone = build_vit(rgb_backbone_name, device)
        self.point_backbone = point_backbone

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.uff = None
        if uff_checkpoint is not None:
            self.uff = UFF(rgb_dim = rgb_dim, pt_dim = pt_dim, proj_dim = 256)
            if not os.path.isfile(uff_checkpoint):
                raise FileNotFoundError(f"Could not read the UFF checkpoint at {uff_checkpoint}")
            state = torch.load(uff_checkpoint, map_location = device, weights_only = False)
            if isinstance(state, dict) and "model" in state and "xyz_mlp" not in state:
                state = state["model"]
            uff_state = self.uff.state_dict()
            mismatched = []
            for k, v in uff_state.items():
                if k not in state or state[k].shape != v.shape:
                    mismatched.append(k)
            for k in state.keys():
                if k not in uff_state:
                    mismatched.append(k)
            if len(mismatched) > 0:
                raise ValueError(f"UFF checkpoint '{uff_checkpoint}' loaded but its parameters do not "
                                 f"match the UFF architecture (mismatched keys: {mismatched[:5]}).")
            self.uff.load_state_dict(state)
            self.uff = self.uff.to(device).eval()
            for p in self.uff.parameters():
                p.requires_grad_(False)

        self.register_buffer("memory_bank_rgb", torch.empty(0))
        self.register_buffer("memory_bank_pt", torch.empty(0))
        self.register_buffer("memory_bank_fs", torch.empty(0))

        for name in ("rgb", "pt", "fs"):
            self.register_buffer(f"_{name}_mean", torch.zeros(1))
            self.register_buffer(f"_{name}_std", torch.ones(1))

        self.dlf: Optional[DeepLatentFusion] = None
        if mode == "fusion":
            self.dlf = DeepLatentFusion(ocsvm_nu = ocsvm_nu, ocsvm_maxiter = ocsvm_maxiter)

        self.anomaly_map_generator = AnomalyMapGenerator(radius = 4)

        self.parts_rgb: list = []
        self.parts_pt: list = []
        self.parts_fs: list = []

        self.s_lib: list = []
        self.s_map_lib: list = []

    @staticmethod
    def reference_training_args() -> TrainingArgs:
        return TrainingArgs(batch_size = 1, epochs = 1, shuffle = False, num_workers = 1)

    @staticmethod
    def spatial_to_rows(feat: torch.Tensor) -> torch.Tensor:
        B,D,H,W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B * H * W, D)

    @staticmethod
    def count_rows(parts: list) -> int:
        n = 0
        for p in parts:
            n += p.shape[0]
        return n

    @staticmethod
    def pixel_maps_to_rows(pixel_maps: list) -> torch.Tensor:
        B = pixel_maps[0].shape[0]
        flat_maps = []
        for p in pixel_maps:
            flat_maps.append(p.reshape(B, 1, -1))
        px = torch.cat(flat_maps, dim = 1)
        return px.permute(0, 2, 1).reshape(-1, len(pixel_maps))

    def normalization_stats(self, modality: str) -> tuple[torch.Tensor, torch.Tensor]:
        if modality == "rgb":
            return self._rgb_mean, self._rgb_std
        elif modality == "pt":
            return self._pt_mean, self._pt_std
        else:
            return self._fs_mean, self._fs_std
        
    @torch.no_grad()
    def extract_rgb_features(self, rgb: torch.Tensor) -> torch.Tensor:
        self.rgb_backbone.eval()
        return extract_vit_features(self.rgb_backbone, rgb.to(self.device))

    @torch.no_grad()
    def extract_pt_features(self, organized_pc: torch.Tensor) -> torch.Tensor:
        self.point_backbone.train()
        return extract_point_cloud_features(self.point_backbone, organized_pc, self.device, self.point_grid_size)

    def resize_to_point_grid(self, feat: torch.Tensor) -> torch.Tensor:
        return self._point_grid_resize(feat)

    @torch.no_grad()
    def extract_fs_features(self, rgb_feat: torch.Tensor, pt_feat: torch.Tensor) -> torch.Tensor:
        self.uff.eval()
        fs_feat, _ = self.uff(self.resize_to_point_grid(rgb_feat), pt_feat)
        return fs_feat

    def forward(self, batch: torch.Tensor):
        batch = batch.to(self.device)
        rgb, organized_pc = batch[:, :3], batch[:, 3:6]

        rgb_feat = self.extract_rgb_features(rgb)
        pt_feat = None
        if self.mode in ("3d", "fusion"):
            pt_feat = self.extract_pt_features(organized_pc)
        fs_feat = None
        if self.mode == "fusion":
            fs_feat = self.extract_fs_features(rgb_feat, pt_feat)

        if self.training:
            if self.mode == "rgb":
                return self.spatial_to_rows(rgb_feat)
            elif self.mode == "3d":
                return self.spatial_to_rows(pt_feat)
            else:
                return (self.spatial_to_rows(rgb_feat), self.spatial_to_rows(pt_feat), self.spatial_to_rows(fs_feat))

        return self.score(rgb_feat, pt_feat, fs_feat)

    def bank_score(self, feat: torch.Tensor, bank: torch.Tensor, modality: str):
        mean, std = self.normalization_stats(modality)
        feat_n = (feat - mean) / std

        B, _, H_p, W_p = feat.shape
        flat = self.spatial_to_rows(feat_n)
        patch_scores, locations = nn_scores(flat, bank)
        patch_scores_grid = patch_scores.reshape(B, 1, H_p, W_p)
        pixel_map = self.anomaly_map_generator(patch_scores_grid, self.input_size)

        img_score = reweighted_image_score(
            patch_scores.reshape(B, H_p * W_p),
            locations.reshape(B, H_p * W_p),
            flat, bank, num_neighbors = self.num_neighbors,
            eps = 0.0 if self.mode == "fusion" else 1e-5,
        )

        if self.mode == "fusion":
            image_weight, pixel_weight = self.modality_weights[modality]
            img_score = img_score * image_weight
            pixel_map = pixel_map * pixel_weight

        return img_score, pixel_map

    def score(self, rgb_feat, pt_feat, fs_feat) -> tuple[torch.Tensor, torch.Tensor]:
        B = rgb_feat.shape[0]
        img_scores_list, pixel_maps_list = [], []

        if self.mode == "rgb":
            candidates = [("rgb", rgb_feat, self.memory_bank_rgb)]
        elif self.mode == "3d":
            candidates = [("pt", pt_feat, self.memory_bank_pt)]
        else:
            candidates = [("pt", pt_feat, self.memory_bank_pt),("rgb", rgb_feat, self.memory_bank_rgb),("fs", fs_feat, self.memory_bank_fs)]

        missing = []
        for modality, feat, bank in candidates:
            if bank.numel() == 0:
                missing.append(modality)
        if len(missing) > 0:
            raise RuntimeError(f"Empty memory bank for {missing}")

        for modality, feat, bank in candidates:
            img_s, px_map = self.bank_score(feat, bank, modality)
            img_scores_list.append(img_s)
            pixel_maps_list.append(px_map)

        if self.mode != "fusion":
            return pixel_maps_list[0], img_scores_list[0]

        if not self.dlf.is_fitted():
            raise RuntimeError("DLF is not fitted")

        H_out, W_out = self.input_size
        s = torch.stack(img_scores_list, dim = -1)
        px = self.pixel_maps_to_rows(pixel_maps_list)

        px_score, img_score = self.dlf.score(s, px, self.device)
        px_map = px_score.reshape(B, 1, H_out, W_out)
        return px_map, img_score

    @torch.no_grad()
    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs = None):
        emb = self(batch)
        if self.mode == "rgb":
            self.parts_rgb.append(emb.cpu())
        elif self.mode == "3d":
            self.parts_pt.append(emb.cpu())
        else:
            rgb_part, pt_part, fs_part = emb
            self.parts_rgb.append(rgb_part.cpu())
            self.parts_pt.append(pt_part.cpu())
            self.parts_fs.append(fs_part.cpu())

    def train_epoch(self, epoch, train_dataloader, training_args: TrainingArgs):
        self.parts_rgb, self.parts_pt, self.parts_fs = [], [], []
        self.train()
        n_seen = 0
        for batch in tqdm(train_dataloader, desc = "Embedding extraction", mininterval = 5.0):
            self.train_step(batch, training_args)
            n_seen += batch.shape[0]
            if n_seen > self.max_train_samples:
                break

        targets = self.coreset_targets(self.parts_rgb, self.parts_pt)
        print(f"Coreset target sizes: {targets} (f_coreset={self.f_coreset})")

        self.build_banks(self.parts_rgb, self.parts_pt, self.parts_fs, targets)

        if self.mode == "fusion":
            self.collect_dlf_scores(train_dataloader)
            self.dlf.fit(self.s_lib, self.s_map_lib)
            self.s_lib, self.s_map_lib = [], []

        self.eval()

    @torch.no_grad()
    def collect_dlf_scores(self, train_dataloader):
        self.s_lib = []
        self.s_map_lib = []
        self.eval()
        n_seen = 0

        for batch in tqdm(train_dataloader, desc = "DLF score collection", mininterval = 5.0):
            batch = batch.to(self.device)
            rgb, organized_pc = batch[:, :3], batch[:, 3:6]

            rgb_feat = self.extract_rgb_features(rgb)
            pt_feat = self.extract_pt_features(organized_pc)
            fs_feat = self.extract_fs_features(rgb_feat, pt_feat)

            img_scores, pixel_maps = [], []
            for feat, bank, modality in ((pt_feat, self.memory_bank_pt, "pt"),(rgb_feat, self.memory_bank_rgb, "rgb"),(fs_feat, self.memory_bank_fs, "fs")):
                s, m = self.bank_score(feat, bank, modality)
                img_scores.append(s)
                pixel_maps.append(m)

            n_seen += rgb_feat.shape[0]
            self.s_lib.append(torch.stack(img_scores, dim = -1).cpu().numpy())
            self.s_map_lib.append(self.pixel_maps_to_rows(pixel_maps).cpu().numpy())

            if n_seen > self.max_train_samples:
                break

    def set_normalization_stats(self, lib_rgb: Optional[torch.Tensor], lib_pt: Optional[torch.Tensor]) -> None:
        if self.mode == "rgb":
            self._rgb_mean = lib_rgb.mean().reshape(1)
            self._rgb_std = lib_rgb.std().reshape(1)
        elif self.mode == "fusion":
            shared_mean = lib_pt.mean().reshape(1)
            shared_std = lib_rgb.std().reshape(1)
            self._rgb_mean, self._rgb_std = shared_mean, shared_std
            self._pt_mean, self._pt_std = shared_mean, shared_std
            self._fs_mean, self._fs_std = shared_mean, shared_std

    def append_zero_anchor_pt(self) -> None:
        bank = self.memory_bank_pt
        bank = bank[torch.nonzero(torch.all(bank != 0, dim = 1))[:, 0]]
        self.memory_bank_pt = torch.cat(
            (bank, torch.zeros(1, bank.shape[1], dtype = bank.dtype, device = bank.device)), 0)

    def coreset_targets(self, parts_rgb: list, parts_pt: list) -> dict:
        if self.mode == "rgb":
            n_ref = self.count_rows(parts_rgb)
        else:
            n_ref = self.count_rows(parts_pt)
        first = max(1, int(self.f_coreset * n_ref))
        if self.mode != "fusion":
            return {"rgb": first, "pt": first, "fs": first}
        second = max(1, int(self.f_coreset * first))
        return {"pt": first, "rgb": second, "fs": second}

    def consume_parts(self, parts: list) -> torch.Tensor:
        n = self.count_rows(parts)
        out = torch.empty((n, parts[0].shape[1]), dtype = torch.float32, device = self.device)
        start = 0
        while parts:
            part = parts.pop(0)
            out[start:start + part.shape[0]] = part.to(self.device)
            start += part.shape[0]
        return out

    def normalized_coreset(self, lib: torch.Tensor, modality: str, n: int) -> torch.Tensor:
        mean, std = self.normalization_stats(modality)
        lib.sub_(mean).div_(std)
        if self.f_coreset >= 1:
            return lib
        return CoresetExtractor(False, self.device, k = max(n - 1, 0)).extract_coreset(lib)

    def build_banks(self, parts_rgb: list, parts_pt: list, parts_fs: list, targets: dict) -> None:
        lib_rgb = self.consume_parts(parts_rgb) if parts_rgb else None
        lib_pt = self.consume_parts(parts_pt) if parts_pt else None
        self.set_normalization_stats(lib_rgb, lib_pt)

        print("Coreset Extraction:")
        if lib_pt is not None:
            self.memory_bank_pt = self.normalized_coreset(lib_pt, "pt", targets["pt"])
        del lib_pt
        torch.cuda.empty_cache()
        if lib_rgb is not None:
            self.memory_bank_rgb = self.normalized_coreset(lib_rgb, "rgb", targets["rgb"])
        del lib_rgb
        torch.cuda.empty_cache()
        if parts_fs:
            self.memory_bank_fs = self.normalized_coreset(self.consume_parts(parts_fs), "fs", targets["fs"])
        torch.cuda.empty_cache()

        if self.mode == "fusion":
            self.append_zero_anchor_pt()

    def save(self, save_path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "dlf": self.dlf if self.dlf is not None and self.dlf.is_fitted() else None,
        }, save_path)