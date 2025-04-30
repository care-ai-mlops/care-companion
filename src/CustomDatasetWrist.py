from pathlib import Path
from typing import Dict, Tuple
import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class CustomDatasetWrist:
    """Wraps torchvision.ImageFolder splits into a dict of DataLoaders."""

    def __init__(self,
                root: str | Path = "data/data",
                img_size: int = 224,
                batch_size: int = 32,
                num_workers: int = 4,
                augment: bool = False,    
                pin_memory: bool | None = False,             
            ) -> None:
        self.root = Path(root)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.csv_path = self.root / "filtered_dataset.csv"
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

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

    def _make_dataset(self, split: str) -> datasets.ImageFolder:
        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split dir: {split_dir}")
        
        projection_dir = [d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith("projection")]
        if not projection_dir:
            raise FileNotFoundError(f"Missing projection dir: {split_dir}")

        datasets_list = [
            datasets.ImageFolder(root=proj_dir, transform=self._build_tfms(train=(split == "train")))
            for proj_dir in projection_dir
        ]
        if len(datasets_list) == 1:
            return datasets_list[0]

        return torch.utils.data.ConcatDataset(datasets_list)
    
    def get_loaders(self) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
        """
        Returns
        -------
        loaders       dict: {'train': DLoader, 'val': …, …}
        class_to_idx  dict: {'FRACTURE': 0, 'NOT_FRACTURE': 1}
        """
        splits = ["train", "val", "test", "canary_testing_data", "production_data"]
        datasets_dict = {s: self._make_dataset(s) for s in splits}

        sample_ds = next(ds for ds in datasets_dict.values() if len(ds) > 0)
        class_to_idx = sample_ds.datasets[0].class_to_idx if isinstance(sample_ds, torch.utils.data.ConcatDataset) else sample_ds.class_to_idx  # type: ignore[attr-defined]

        loaders = {
            split: DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=(split == "train"),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
            )
            for split, ds in datasets_dict.items()
        }
        return loaders, class_to_idx