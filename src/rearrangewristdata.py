from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
import shutil, math, random, os, argparse
from pathlib import Path
from typing import List, Tuple

@dataclass
class Config:
    root: Path = Path("wrist-data/")
    target_ratios: dict[str, float] = field(
        default_factory=lambda: {
            "train": 0.75,
            "val": 0.08,
            "test": 0.10,
            "canary_testing_data": 0.04,
            "production_data": 0.03,
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
        if not os.path.exists(self.cfg.root):
            os.makedirs(self.cfg.root)
        df.to_csv(self.cfg.root / filename, index=False)
        print(f"Saved {filename} to {self.cfg.root / filename}")
    
    def safe_remove_dir(self, path: Path) -> None:
        if path.exists() and path.is_dir(): 
            shutil.rmtree(path)
            print(f"Removed directory: {path}")
        else: 
            print(f"Directory does not exist: {path}")
    
    def ensure_dir(self, path: Path) -> None:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    
    def sanity_checks(self, splits: List[str], df: pd.DataFrame) -> None:
        fr_val = nf_val = 0
        for split in splits:
            for i in range(1, 3):
                fr_path = self.cfg.root / split / f"projection{i}" / "FRACTURE"
                nf_path = self.cfg.root / split / f"projection{i}" / "NOT_FRACTURE"
                fr_val += len(list(fr_path.glob("*")))
                nf_val += len(list(nf_path.glob("*")))
                print(f"Split: {split}, Projection: {i}, FRACTURE: {fr_val}, NOT_FRACTURE: {nf_val}")
        df_frac = df[df['label'] == 'FRACTURE']['label'].count()
        df_not_frac = df[df['label'] == 'NOT_FRACTURE']['label'].count()
        assert df_frac == fr_val, f"Mismatch in FRACTURE count: {df_frac} != {fr_val}"
        assert df_not_frac == nf_val, f"Mismatch in NOT_FRACTURE count: {df_not_frac} != {nf_val}"
        print(f"Total FRACTURE: {fr_val}, Total NOT_FRACTURE: {nf_val} across all splits")


class SplitAndCopy:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.helper = Helper(cfg)
        self.TARGET_DIR = self.cfg.root

    def compute_splits(self, indices, ratios) -> dict[str, List[int]]:
        random.shuffle(indices)
        n = len(indices)
        assignments = {}
        start = 0
        for split, ratio in ratios.items():
            if split == 'train': continue
            size = math.floor(n * ratio)
            assignments[split] = indices[start: start + size]
            start += size
        assignments['train'] = indices[start:]
        return assignments


    def _read_add_label_(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.meta_csv)
        df = df[df['projection'] != 3].copy()
        print(f"Filtered to {len(df)} rows with projection != 3")

        assert df["projection"].nunique() == 2, "projection should have 2 unique values"

        df["label"] = df["fracture_visible"].apply(lambda x: "FRACTURE" if x == 1 else "NOT_FRACTURE")
        assert df["label"].nunique() == 2, "label should have 2 unique values"
        
        print(f"Unique Labels: {df['label'].unique()}")
        print(f"FRACTURED LABELS: {df[df['label'] == 'FRACTURE']['label'].count()}, NOT FRACTURED LABELS: {df[df['label'] == 'NOT_FRACTURE']['label'].count()}")
        return df

    def split_data(self, df: pd.DataFrame) -> pd.DataFrame:
        assignments = []
        for (label, _), group in df.groupby(["label", "projection"]):
            indices = group.index.tolist()
            split_assignments = self.compute_splits(indices, self.cfg.target_ratios)
            for split, indices in split_assignments.items():
                assignments.extend((split, i) for i in indices)

        split_df = pd.DataFrame(assignments, columns=["split", "index"]).set_index("index")
        df = df.join(split_df)
        self.helper.save_csv(df, "filtered_dataset.csv")
        return df
    
    def copy_files(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            split_dir = self.cfg.SPLIT_MAP[row["split"]]
            proj_dir = f"projection{row['projection']}"
            label_dir = row["label"]

            dest_dir = self.cfg.root / split_dir / proj_dir / label_dir
            self.helper.ensure_dir(dest_dir)

            filestem = row["filestem"]
            candidate_files = list((self.cfg.root / label_dir).glob(f"{filestem}.*"))
            if not candidate_files:
                raise FileNotFoundError(f"Cannot locate file starting with {filestem} in {label_dir}/")
            src = candidate_files[0]
            dst = dest_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
        print("Copied files to respective directories.")

        for dir in ["FRACTURE", "NOT_FRACTURE"]:
            path = self.cfg.root / dir
            if path.exists() and path.is_dir():
                self.helper.safe_remove_dir(path)
                print(f"Removed directory: {path}")
            else:
                print(f"Directory does not exist: {path}")

        self.helper.sanity_checks(self.cfg.SPLIT_MAP.keys(), df)
        print("Sanity checks passed.")

    def execute(self) -> None:
        df = self._read_add_label_()
        print(df.columns)
        df = self.split_data(df)
        self.copy_files(df)
        print("Data splitting and copying completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and copy wrist data.")
    parser.add_argument("--root", type=str, default="/Users/akashpeddaputha/Documents/Projects/MLOps-Trial/data/wrist-xray_check_1", help="Root directory for the dataset.")
    args = parser.parse_args()

    cfg = Config(root=Path(args.root))
    splitter = SplitAndCopy(cfg)
    splitter.execute()
        
    