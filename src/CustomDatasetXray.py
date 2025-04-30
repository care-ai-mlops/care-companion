from pathlib import Path
from typing import Dict, Tuple
import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CustomDatasetXray:
    """Wraps torchvision.ImageFolder splits into a dict of DataLoaders."""

    def __init__(
                self,
                root: str | Path = "mnt/object/chest-data",
                img_size: int = 224,
                batch_size: int = 32,
                num_workers: int = 4,
                augment: bool = False,                 
            ) -> None:
        self.root = Path(root)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment

    def _build_tfms(self, train: bool):
        base = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ]
        if train and self.augment:
            base += [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        return transforms.Compose(base)

    def _make_dataset(self, split: str):
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split dir: {split_dir}")
        transforms = self._build_tfms(train=(split == "train"))
        return datasets.ImageFolder(split_dir, transform=transforms)

    def get_loaders(self) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
        """
        Returns
        -------
        loaders       dict: {'train': DLoader, 'val': …, …}
        class_to_idx  dict: {'NORMAL': 0, 'PNEUMONIA': 1, …}
        """
        splits = ["train", "val", "test", "canary_testing_data", "production_data"]
        datasets_dict = {s: self._make_dataset(s) for s in splits}

        class_to_idx = datasets_dict["train"].class_to_idx

        loaders = {
            split: DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=(split == "train"),
                num_workers=self.num_workers,
                pin_memory=True,
            )
            for split, ds in datasets_dict.items()
        }
        return loaders, class_to_idx