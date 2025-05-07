from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
import shutil, math, random, os, argparse
from pathlib import Path
from typing import List, Tuple

@dataclass
class Config:
    """Holds paths, column names, split ratios, RNG seed."""
    root: Path = Path("/mnt/object/wrist-data/")
    target_ratios: dict[str, float] = field(
        default_factory=lambda: {
            "train": 0.80,
            "val": 0.08,
            "test": 0.10,
            "canary_testing_data": 0.01,
            "production_data": 0.01,
        }
    )
    seed: int = 42
    meta_csv: Path = field(init=False)

    SPLIT_MAP: dict[str, str] = field(
        default_factory=lambda: {
            "train": "train",
            "val": "val",
            "test": "test",
            "canary_testing_data": "canary_testing_data",
            "production_data": "production_data",
        }
    )

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.meta_csv = self.root / "dataset.csv"

        if abs(sum(self.target_ratios.values()) - 1.0) > 1e-6:
            raise ValueError("`target_ratios` must sum to 1.0")

        random.seed(self.seed)

class Helper:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def save_csv(self, df: pd.DataFrame, filename: str) -> None:
        self.ensure_dir(self.cfg.root)
        df.to_csv(self.cfg.root / filename, index=False)
        print(f"Saved {filename} to {self.cfg.root / filename}")

    def safe_remove_dir(self, path: Path) -> None:
        if path.exists() and path.is_dir():
            try:
                shutil.rmtree(path)
                print(f"Removed directory: {path}")
            except OSError as e:
                print(f"Error removing directory {path}: {e}")
        else:
            print(f"Directory does not exist or is not a directory: {path}")


    def ensure_dir(self, path: Path) -> None:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    def sanity_checks(self, splits: List[str], df: pd.DataFrame) -> None:
        print("\nPerforming sanity checks...")
        total_files_in_dirs = 0
        for split in splits:
            for i in range(1, 3):
                fr_path = self.cfg.root / split / f"projection{i}" / "FRACTURE"
                nf_path = self.cfg.root / split / f"projection{i}" / "NOT_FRACTURE"
                fr_count = len(list(fr_path.glob("*")))
                nf_count = len(list(nf_path.glob("*")))
                print(f"Split: {split}, Projection: {i}, FRACTURE: {fr_count}, NOT_FRACTURE: {nf_count}")
                total_files_in_dirs += fr_count + nf_count

        df_filtered = df[df['projection'] != 3] 
        df_frac_count = df_filtered[df_filtered['label'] == 'FRACTURE'].shape[0]
        df_not_frac_count = df_filtered[df_filtered['label'] == 'NOT_FRACTURE'].shape[0]
        df_total_count = df_filtered.shape[0]

        print(f"\nTotal files in new directories: {total_files_in_dirs}")
        print(f"Total FRACTURE entries in filtered CSV: {df_frac_count}")
        print(f"Total NOT_FRACTURE entries in filtered CSV: {df_not_frac_count}")
        print(f"Total entries in filtered CSV (Projections 1 & 2): {df_total_count}")


        assert total_files_in_dirs == df_total_count, \
            f"Mismatch in total file count: {total_files_in_dirs} in directories vs {df_total_count} in filtered CSV"


class SplitAndCopy:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.helper = Helper(cfg)

    def compute_splits_by_patient(self, patient_ids: List[int], ratios: dict[str, float]) -> dict[str, List[int]]:
        """Splits a list of patient IDs into subsets based on target ratios."""
        random.shuffle(patient_ids)
        n = len(patient_ids)
        assignments = {}
        start = 0
        for split, ratio in ratios.items():
            if split == 'train': continue
            size = math.floor(n * ratio)
            assignments[split] = patient_ids[start: start + size]
            start += size
        assignments['train'] = patient_ids[start:]
        return assignments


    def _read_add_label_(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.meta_csv)
        df_filtered = df[df['projection'] != 3].copy()
        print(f"Original dataset size: {len(df)} rows")
        print(f"Filtered to {len(df_filtered)} rows with projection != 3")

        assert df_filtered["projection"].nunique() == 2, f"Filtered dataframe should have 2 unique projection values, found {df_filtered['projection'].nunique()}"

        df_filtered["label"] = df_filtered["fracture_visible"].apply(lambda x: "FRACTURE" if x == 1 else "NOT_FRACTURE")
        assert df_filtered["label"].nunique() == 2, f"label column should have 2 unique values, found {df_filtered['label'].nunique()}"

        print(f"Unique Labels in filtered data: {df_filtered['label'].unique()}")
        print(f"FRACTURE count in filtered data: {df_filtered[df_filtered['label'] == 'FRACTURE'].shape[0]}")
        print(f"NOT_FRACTURE count in filtered data: {df_filtered[df_filtered['label'] == 'NOT_FRACTURE'].shape[0]}")

        return df_filtered

    def split_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assigns a split (train, val, test, etc.) to each row based on patient ID."""
        patient_ids = df['patient_id'].unique().tolist()
        print(f"\nFound {len(patient_ids)} unique patients.")

        patient_assignments = self.compute_splits_by_patient(patient_ids, self.cfg.target_ratios)

        patient_to_split = {}
        print("\nPatient ID assignments to splits:")
        for split, ids in patient_assignments.items():
            print(f"  {split}: {len(ids)} patients")
            for id in ids:
                patient_to_split[id] = split

        df['split'] = df['patient_id'].apply(lambda x: patient_to_split[x])

        print("\nRow distribution across splits (should reflect patient distribution):")
        print(df['split'].value_counts())


        self.helper.save_csv(df, "filtered_dataset_patient_split.csv")
        return df

    def copy_files(self, df: pd.DataFrame) -> None:
        """Copies files to the new directory structure based on the assigned split."""
        print("\nStarting file copying...")
        for split_dir_name in self.cfg.SPLIT_MAP.values():
            split_dir_path = self.cfg.root / split_dir_name
            self.helper.safe_remove_dir(split_dir_path)

        original_fracture_dir = self.cfg.root / "FRACTURE"
        original_not_fracture_dir = self.cfg.root / "NOT_FRACTURE"

        if not original_fracture_dir.exists() and not original_not_fracture_dir.exists():
             print("Error: Original 'FRACTURE' or 'NOT_FRACTURE' directories not found in the root path.")
             print("Please ensure the original image folders are present for copying.")
             return 

        copied_count = 0
        for index, row in df.iterrows():
            split_dir_name = self.cfg.SPLIT_MAP[row["split"]]
            proj_dir_name = f"projection{int(row['projection'])}" 
            label_dir_name = row["label"]

            dest_dir = self.cfg.root / split_dir_name / proj_dir_name / label_dir_name
            self.helper.ensure_dir(dest_dir) 

            filestem = row["filestem"]
            candidate_sources = [
                original_fracture_dir / f"{filestem}.jpg", 
                original_fracture_dir / f"{filestem}.png", 
                original_not_fracture_dir / f"{filestem}.jpg",
                original_not_fracture_dir / f"{filestem}.png",
            ]

            src_file = None
            for candidate_src in candidate_sources:
                if candidate_src.exists():
                    src_file = candidate_src
                    break

            if src_file is None:
                print(f"Warning: Cannot locate source file for filestem {filestem}. Looked in {original_fracture_dir} and {original_not_fracture_dir}. Skipping.")
                continue 

            dst_file = dest_dir / src_file.name
            if not dst_file.exists():
                try:
                    shutil.copy2(src_file, dst_file)
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying file {src_file} to {dst_file}: {e}")

        print(f"Completed file copying. Total files copied: {copied_count}")
        print("Note: The original FRACTURE and NOT_FRACTURE folders were NOT removed.")


    def execute(self) -> None:
        print("Starting data splitting and copying process...")
        df = self._read_add_label_()
        df = self.split_data(df)
        self.copy_files(df)
        self.helper.sanity_checks(self.cfg.SPLIT_MAP.keys(), df)
        print("Data splitting and copying process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and copy wrist data by patient ID.")
    parser.add_argument("--root", type=str,
                        default="/mnt/object/wrist-data/", 
                        help="Root directory containing the wrist-data folder (with FRACTURE, NOT_FRACTURE, and dataset.csv).")
    args = parser.parse_args()

    cfg = Config(root=Path(args.root))
    splitter = SplitAndCopy(cfg)
    splitter.execute()