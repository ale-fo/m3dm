from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import ImageFilter


class AnomalyMapGenerator(nn.Module):
    def __init__(self, radius: int = 4) -> None:
        super().__init__()
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius = radius)

    def blur(self, img: torch.Tensor) -> torch.Tensor:
        map_max = img.max()
        return self.load(self.unload(img / map_max).filter(self.blur_kernel)) * map_max

    def forward(self, patch_scores: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        anomaly_map = F.interpolate(patch_scores, size = (image_size[0], image_size[1]), mode = "bilinear", align_corners = False)
        device = anomaly_map.device
        blurred = [self.blur(m.cpu()) for m in anomaly_map]
        return torch.stack(blurred, dim = 0).to(device)