from pathlib import Path
import pandas as pd
import tifffile
import torch
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode

from moviad.datasets.vad_dataset import VADDataset
from moviad.utilities.configurations import Split, LabelName
from moviad.datasets.dataset_arguments import DatasetArguments

IMG_EXTENSIONS = (".png", ".PNG")
PC_EXTENSIONS = (".tif", ".tiff")

CATEGORIES = (
    "bagel",
    "cable_gland",
    "carrot",
    "cookie",
    "dowel",
    "foam",
    "peach",
    "potato",
    "rope",
    "tire",
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class MVTec3DDataset(VADDataset):
    """MVTec3D AD dataset adapted for moviad"""
    def __init__(
        self,
        dataset_arguments: DatasetArguments,
        category: str,
        split: Split,
    ) -> None:
        super().__init__(
            arguments = dataset_arguments,
            category = category,
            split = split,
        )

        self.root_category = Path(self.dataset_arguments.dataset_path) / Path(self.category)
        self.samples: pd.DataFrame = None

        if not self.dataset_arguments.image_transform_list:
            raise ValueError("MVTec3DDataset requires an explicit list of image transformations")

        self.transform_mask = transforms.Compose(
            [
                transforms.Resize(self.dataset_arguments.gt_mask_size, interpolation = InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

        self.read_point_cloud = tifffile.imread

        self.transform_pc = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    self.dataset_arguments.img_size,
                    interpolation = InterpolationMode.NEAREST,
                ),
            ]
        )

        self.load_dataset()

    def is_loaded(self) -> bool:
        return self.samples is not None

    def load_dataset(self) -> None:
        if self.is_loaded():
            print("dataset already loaded")
            return

        split = self.split
        samples_list = []

        base_dir = self.root_category / split.value
        if not base_dir.exists():
            raise RuntimeError(f"split directory not found: {base_dir}")

        for defect_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
            defect_type = defect_dir.name
            is_normal = defect_type == "good"
            label_index = LabelName.NORMAL.value if is_normal else LabelName.ABNORMAL.value
            needs_mask = not is_normal and split != Split.TRAIN

            rgb_files = {p.stem: p for p in (defect_dir / "rgb").glob("*") if p.suffix in IMG_EXTENSIONS}
            pc_files = {p.stem: p for p in (defect_dir / "xyz").glob("*") if p.suffix.lower() in PC_EXTENSIONS}
            gt_files = (
                {p.stem: p for p in (defect_dir / "gt").glob("*") if p.suffix in IMG_EXTENSIONS}
                if needs_mask else {}
            )

            for stem in sorted(rgb_files):
                if stem not in pc_files:
                    raise RuntimeError(f"missing point cloud for image: {rgb_files[stem]}")

                mask_path = None
                if needs_mask:
                    if stem not in gt_files:
                        raise RuntimeError(f"missing mask for anomalous image: {rgb_files[stem]}")
                    mask_path = str(gt_files[stem])

                samples_list.append((
                    str(self.root_category),
                    split.value,
                    defect_type,
                    str(rgb_files[stem]),
                    str(pc_files[stem]),
                    mask_path,
                    label_index,
                ))

        if not samples_list:
            raise RuntimeError(f"found 0 samples in {self.root_category}")

        self.samples = pd.DataFrame(
            samples_list,
            columns = ["path", "split", "label", "image_path", "pc_path", "mask_path", "label_index"],
        )

        n_abnormal = int((self.samples["label_index"] == LabelName.ABNORMAL.value).sum())
        print(f"{self.category}/{split.value}: {len(self.samples)} samples loaded "
              f"({n_abnormal} abnormal, {len(self.samples) - n_abnormal} normal)")


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): index of the element to be returned
        Returns:
            sample (Tensor): tensor of shape (6, H, W), RGB on channels 0:3, XYZ on channels 3:6
            label (int): label of the image (if not train split)
            mask (Tensor): tensor of shape (1, H, W) with values in {0,1} (if not train split)
            path (str): path of the RGB image (if not train split)
        """
        sample_row = self.samples.iloc[index]
        image_path = sample_row.image_path

        rgb = self.transform_image(Image.open(image_path).convert("RGB"))
        organized_pc = self.transform_pc(self.read_point_cloud(sample_row.pc_path))

        sample = torch.cat([rgb, organized_pc], dim = 0)

        if self.split == Split.TRAIN:
            return sample

        label = int(sample_row.label_index)

        if pd.notna(sample_row.mask_path):
            mask = self.transform_mask(Image.open(sample_row.mask_path).convert("L"))
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, *self.dataset_arguments.gt_mask_size)

        return sample, label, mask, image_path

    def compute_contamination_ratio(self) -> float:
        """Calculate the ratio of abnormal vs total samples currently loaded."""
        if self.samples is None or len(self.samples) == 0:
            return 0.0
        abnormal_count = len(self.samples[self.samples["label_index"] == LabelName.ABNORMAL.value])
        return abnormal_count / len(self.samples)

    @staticmethod
    def get_categories() -> list:
        return list(CATEGORIES)