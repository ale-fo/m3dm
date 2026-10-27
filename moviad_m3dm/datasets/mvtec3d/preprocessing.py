import argparse
import math
import os
from pathlib import Path

import numpy as np
import tifffile as tiff
from PIL import Image
import open3d as o3d

def organized_pc_to_unorganized_pc(organized_pc: np.ndarray) -> np.ndarray:
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])

def get_edges_of_pc(organized_pc: np.ndarray) -> np.ndarray:
    edges = organized_pc[0:10, :, :].reshape(-1, organized_pc.shape[2])
    edges = np.concatenate([edges, organized_pc[-10:, :, :].reshape(-1, organized_pc.shape[2])], axis = 0)
    edges = np.concatenate([edges, organized_pc[:, 0:10, :].reshape(-1, organized_pc.shape[2])], axis = 0)
    edges = np.concatenate([edges, organized_pc[:, -10:, :].reshape(-1, organized_pc.shape[2])], axis = 0)
    return edges[np.nonzero(np.all(edges != 0, axis = 1))[0], :]

def get_plane_eq(unorganized_pc: np.ndarray, ransac_n_pts: int = 50):
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc))
    plane_model, _ = o3d_pc.segment_plane(distance_threshold = 0.004, ransac_n = ransac_n_pts, num_iterations = 1000)
    return plane_model

def remove_plane(organized_pc_clean: np.ndarray, organized_rgb: np.ndarray, distance_threshold: float = 0.005):
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc_clean)
    unorganized_rgb = organized_pc_to_unorganized_pc(organized_rgb)
    clean_planeless_unorganized_pc = unorganized_pc.copy()
    planeless_unorganized_rgb = unorganized_rgb.copy()

    plane_model = get_plane_eq(get_edges_of_pc(organized_pc_clean))
    distances = np.abs(np.dot(np.array(plane_model),np.hstack((clean_planeless_unorganized_pc, np.ones((clean_planeless_unorganized_pc.shape[0], 1)))).T))
    plane_indices = np.argwhere(distances < distance_threshold)

    planeless_unorganized_rgb[plane_indices] = 0
    clean_planeless_unorganized_pc[plane_indices] = 0
    return (clean_planeless_unorganized_pc.reshape(organized_pc_clean.shape), planeless_unorganized_rgb.reshape(organized_rgb.shape))


def connected_components_cleaning(organized_pc: np.ndarray, organized_rgb: np.ndarray):
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc)
    unorganized_rgb = organized_pc_to_unorganized_pc(organized_rgb)

    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis = 1))[0]
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc[nonzero_indices, :]))
    labels = np.array(o3d_pc.cluster_dbscan(eps = 0.006, min_points = 30, print_progress = False))

    unique_cluster_ids, cluster_size = np.unique(labels, return_counts = True)
    largest_cluster_id = unique_cluster_ids[np.argmax(cluster_size)]
    outlier_indices = nonzero_indices[np.argwhere(labels != largest_cluster_id)]
    unorganized_pc[outlier_indices] = 0
    unorganized_rgb[outlier_indices] = 0
    return unorganized_pc.reshape(organized_pc.shape), unorganized_rgb.reshape(organized_rgb.shape)

def roundup_next_100(x: int) -> int:
    return int(math.ceil(x / 100.0)) * 100

def pad_cropped_pc(cropped_pc: np.ndarray, single_channel: bool = False) -> np.ndarray:
    orig_h, orig_w = cropped_pc.shape[0], cropped_pc.shape[1]
    large_side = max(roundup_next_100(orig_h), roundup_next_100(orig_w))

    a = (large_side - orig_h) // 2
    aa = large_side - a - orig_h
    b = (large_side - orig_w) // 2
    bb = large_side - b - orig_w
    if single_channel:
        return np.pad(cropped_pc, pad_width = ((a, aa), (b, bb)), mode = "constant")
    return np.pad(cropped_pc, pad_width = ((a, aa), (b, bb), (0, 0)), mode = "constant")


def preprocess_pc(tiff_path) -> None:
    organized_pc = tiff.imread(tiff_path)
    rgb_path = str(tiff_path).replace("xyz", "rgb").replace("tiff", "png")
    gt_path = str(tiff_path).replace("xyz", "gt").replace("tiff", "png")
    organized_rgb = np.array(Image.open(rgb_path))

    gt_exists = os.path.isfile(gt_path)
    organized_gt = np.array(Image.open(gt_path)) if gt_exists else None

    planeless_organized_pc, planeless_organized_rgb = remove_plane(organized_pc, organized_rgb)

    padded_planeless_organized_pc = pad_cropped_pc(planeless_organized_pc, single_channel = False)
    padded_planeless_organized_rgb = pad_cropped_pc(planeless_organized_rgb, single_channel = False)
    if gt_exists:
        padded_organized_gt = pad_cropped_pc(organized_gt, single_channel = True)

    organized_clustered_pc, organized_clustered_rgb = connected_components_cleaning(padded_planeless_organized_pc, 
                                                                                    padded_planeless_organized_rgb)

    tiff.imwrite(tiff_path, organized_clustered_pc)
    Image.fromarray(organized_clustered_rgb).save(rgb_path)
    if gt_exists:
        Image.fromarray(padded_organized_gt).save(gt_path)


def preprocess_dataset(dataset_path: str) -> int:
    processed = 0
    for path in Path(dataset_path).rglob("*.tiff"):
        preprocess_pc(path)
        processed += 1
        if processed % 50 == 0:
            print(f"Processed {processed} tiff files...")
    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Preprocess MVTec 3D-AD (in place)")
    parser.add_argument("dataset_path", type = str, help = "Root of MVTec 3D-AD. The preprocessed dataset overwrites the existing one.")
    args = parser.parse_args()
    print(f"Processed {preprocess_dataset(args.dataset_path)} tiff files.")
