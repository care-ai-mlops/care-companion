import torch
import torch.utils.data as data
import pandas as pd
import os
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as transforms
import ray 
import numpy as np 
import argparse 


IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

class CustomDatasetWrist(data.Dataset):
    """
    Custom Dataset for loading paired wrist X-ray images (Projection 1 and 2) per patient.
    This version skips patients who do not have both Projection 1 and Projection 2
    images available after filtering Projection 3.
    """

    def __init__(self, 
                 root_dir: str | Path, 
                 split: str,
                 img_size: int = 224, 
                 transform: bool | None = False,
                 csv_filename: str = "filtered_dataset_patient_split.csv"
                 ) -> None:
        """
        Args:
            root_dir (str or Path): Directory with the data split folders (train, val, test, etc.)
                                    and the filtered_dataset_patient_split.csv.
            split (str): The data split to use ('train', 'val', 'test', etc.).
            transform (bool): If it is set true, then transforms would be applied to images.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.metadata_path = self.root_dir / csv_filename

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.metadata_df = pd.read_csv(self.metadata_path)
    
        context = getattr(ray.train, 'get_context', lambda: None)()
        self.world_rank = context.get_world_rank() if context else 0
        if self.world_rank == 0:
            print(f"Loading Data from split: {self.split}")

        self.split_df = self.metadata_df[
            (self.metadata_df['split'] == self.split) &
            (self.metadata_df['projection'].isin([1, 2]))
        ].copy() 

        if self.split_df.empty:
            if self.world_rank == 0:
                print(f"Warning: No data found for split '{self.split}' with projections 1 and 2.")
            self.patient_ids = []
            self.grouped_patients = pd.core.groupby.generic.DataFrameGroupBy(pd.DataFrame())
            return

        grouped = self.split_df.groupby('patient_id')

        self.patient_ids = [
            patient_id for patient_id, patient_data in grouped
            if 1 in patient_data['projection'].values and 2 in patient_data['projection'].values
        ]

        self.grouped_patients = grouped.filter(
            lambda x: x['patient_id'].iloc[0] in self.patient_ids
        ).groupby('patient_id')

        if self.world_rank == 0:
            print(f"Split '{self.split}': Found {len(grouped)} patients before filtering for complete pairs.")
            print(f"Split '{self.split}': Keeping {len(self.patient_ids)} patients with both P1 and P2.")

        self.label_map = {'NOT_FRACTURE': 0, 'FRACTURE': 1}

    def __len__(self):
        """Returns the number of unique patients with both projections in this split."""
        return len(self.patient_ids)

    def _find_image_path(self, row: pd.Series) -> Optional[Path]:
        """Tries to find the image file for a given row across possible extensions."""
        split_dir = self.root_dir / row['split']
        proj_dir = split_dir / f"projection{int(row['projection'])}"
        label_dir = proj_dir / row['label']
        filestem = row['filestem']

        for ext in IMAGE_EXTENSIONS:
            img_path = label_dir / f"{filestem}{ext}"
            if img_path.exists():
                return img_path
        return None

    def _build_transforms(self, train: bool) -> transforms.Compose:
        """
        Builds the image transformation pipeline.
        Args:
            train (bool): If True, apply training transformations.
        Returns:
            transforms.Compose: The composed transformation pipeline.
        """
        base = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ]
        if train and self.transform:
            base += [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        return transforms.Compose(base)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loads and returns paired images and the label for a patient with both projections.

        Args:
            idx (int): Index of the patient to load (from the filtered list of patient_ids).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Tensor of the Projection 1 image.
                - Tensor of the Projection 2 image.
                - Tensor of the numerical label (0 or 1).
        """
        patient_id = self.patient_ids[idx]
        patient_data = self.grouped_patients.get_group(patient_id)

        img1_tensor = img2_tensor = None
        label = None 

        is_train = self.split == 'train'
        pipeline = self._build_transforms(is_train)

        for _, row in patient_data.iterrows():
            img_path = self._find_image_path(row)

            if not img_path:
                raise FileNotFoundError(f"Image file not found for patient {patient_id}, filestem {row['filestem']}, projection {row['projection']}.")

            img = Image.open(img_path).convert('L')
            tensor = pipeline(img) 
            if row['projection'] == 1:
                img1_tensor = tensor
            elif row['projection'] == 2:
                img2_tensor = tensor
            label = self.label_map.get(row['label'], -1) 

        assert img1_tensor is not None, f"Projection 1 image not loaded for patient {patient_id} (should not happen after filtering)"
        assert img2_tensor is not None, f"Projection 2 image not loaded for patient {patient_id} (should not happen after filtering)"
        assert label in [0, 1], f"Invalid or missing label ({label}) for patient {patient_id} (should not happen after filtering)"

        return img1_tensor, img2_tensor, torch.tensor(label, dtype=torch.long)


def create_ray_dataset_iterator(
        dataset: data.Dataset,
        batch_size: int, 
        train: bool = False,
        pin_memory: bool = False):
    """
    Creates a Ray Dataset from a PyTorch Dataset and returns a batch iterator.

    Args:
        dataset (data.Dataset): The PyTorch Dataset to convert.
        batch_size (int): The batch size for iteration.
        pin_memory (bool): If True, pin the data in memory before returning.
                           This is typically handled when moving data to GPU.

    Returns:
        ray.data.Dataset.iterator: An iterator over batches of data.
    """

    ray_dataset = ray.data.from_torch(dataset)


    ray_dataset = ray_dataset.random_shuffle()

    print(f"Creating Ray Data iterator with batch_size={batch_size}.")
    return ray_dataset.iter_batches(batch_size=batch_size)
