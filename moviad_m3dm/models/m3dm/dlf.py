from __future__ import annotations

import numpy as np
import torch
from sklearn import linear_model


def nn_scores(embedding: torch.Tensor, memory_bank: torch.Tensor):
    return torch.cdist(embedding, memory_bank).min(dim = 1)


def reweighted_image_score(
    patch_scores: torch.Tensor,
    locations: torch.Tensor,
    embedding: torch.Tensor,
    memory_bank: torch.Tensor,
    num_neighbors: int = 3,
    eps: float = 0.0,
) -> torch.Tensor:
    batch_size, num_patches = patch_scores.shape
    idx = torch.arange(batch_size, device = patch_scores.device)

    max_patches = torch.argmax(patch_scores, dim = 1)
    s_star = patch_scores[idx, max_patches]
    m_test = embedding.reshape(batch_size, num_patches, -1)[idx, max_patches]

    nn_index = locations[idx, max_patches]
    m_star = memory_bank[nn_index, :]

    k = min(num_neighbors, memory_bank.shape[0])
    support_distances = torch.cdist(m_star, memory_bank)
    _, support_samples = support_distances.topk(k = k, largest = False, dim = 1)
    neighbors = memory_bank[support_samples[:, 1:]]

    m_star_knn = torch.linalg.norm(m_test.unsqueeze(1) - neighbors, dim = 2)
    D = torch.sqrt(torch.tensor(embedding.shape[-1], dtype = torch.float32, device = patch_scores.device))
    w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D), dim = 1) + eps))

    return w * s_star


class DeepLatentFusion:
    def __init__(self, ocsvm_nu: float = 0.5, ocsvm_maxiter: int = 1000, random_state: int = 42):
        self.detect_fuser = linear_model.SGDOneClassSVM(random_state = random_state, nu = ocsvm_nu, max_iter = ocsvm_maxiter)
        self.seg_fuser = linear_model.SGDOneClassSVM(random_state = random_state, nu = ocsvm_nu, max_iter = ocsvm_maxiter)
        self.fitted = False

    def is_fitted(self) -> bool:
        return self.fitted

    def fit(self, s_lib: list, s_map_lib: list):
        s_lib = np.concatenate(s_lib, axis = 0)
        s_map_lib = np.concatenate(s_map_lib, axis = 0)

        for name, arr in [("s_lib", s_lib), ("s_map_lib", s_map_lib)]:
            n_nan, n_inf = np.isnan(arr).sum(), np.isinf(arr).sum()
            col_std = arr.std(axis = 0)
            if n_nan or n_inf:
                print(f"WARNING: {name} contains {n_nan} NaN values ​​and {n_inf} Inf values ​​"
                      f"out of a total of {arr.size} values, please check the raw scores at the source.")
            if (col_std < 1e-8).any():
                print(f"WARNING: {name} has nearly constant columns "
                      f"(std = {col_std}), that bank may not be able to distinguish anything")

        print(f"fitting detect_fuser to {s_lib.shape}...")
        self.detect_fuser.fit(s_lib)
        print(f"fitting seg_fuser to {s_map_lib.shape}...")
        self.seg_fuser.fit(s_map_lib)
        self.fitted = True
        print("DLF fitting completed.")

    def score(
        self,
        img_scores: torch.Tensor,
        patch_scores: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = img_scores.cpu().numpy()
        px = patch_scores.cpu().numpy()

        img_score = torch.tensor(self.detect_fuser.score_samples(s), dtype = torch.float64, device = device)
        px_score = torch.tensor(self.seg_fuser.score_samples(px), dtype = torch.float64, device = device)

        return px_score, img_score