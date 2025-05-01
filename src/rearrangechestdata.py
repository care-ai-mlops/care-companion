from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import argparse, random, shutil
import pandas as pd


@dataclass
class Config:
    """Holds paths, column names, split ratios, RNG seed."""
    root: Path = Path("data/data")
    class_col: str = "label"
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

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.meta_csv = self.root / "dataset_splits_with_metadata.csv"

        if abs(sum(self.target_ratios.values()) - 1.0) > 1e-6:
            raise ValueError("`target_ratios` must sum to 1.0")

        random.seed(self.seed)


class DatasetWalker:
    """Walks train/val/test tree and returns a dataframe of images on disk."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def walk(self) -> pd.DataFrame:
        rows: list[dict] = []
        for split in ["train", "val", "test"]:
            split_dir = self.cfg.root / split
            if not split_dir.exists():
                continue
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                label = class_dir.name
                for img in class_dir.iterdir():
                    if img.is_file():
                        rows.append(
                            {
                                "filepath": str(img.relative_to(self.cfg.root)),
                                "orig_split": split,
                                self.cfg.class_col: label,
                            }
                        )
        return pd.DataFrame(rows)


class Splitter:
    """Handles class-balanced repartitioning."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def repartition(self, group: pd.DataFrame) -> pd.DataFrame:
        n = len(group)
        counts = {k: int(r * n) for k, r in self.cfg.target_ratios.items()}
        counts["train"] += n - sum(counts.values())  # rounding remainder

        idx = list(group.index)
        random.shuffle(idx)

        cur, out = 0, []
        for split, c in counts.items():
            out.extend((i, split) for i in idx[cur : cur + c])
            cur += c
        return pd.DataFrame(out, columns=["idx", "split"])


class Rearranger:
    """Full pipeline: merge metadata, drop UNKNOWN, split, move, save."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.walker = DatasetWalker(cfg)
        self.splitter = Splitter(cfg)

    def _load_dataframe(self) -> pd.DataFrame:
        meta = pd.read_csv(self.cfg.meta_csv)

        paths_df = (
            self.walker.walk()
            .rename(columns={self.cfg.class_col: f"{self.cfg.class_col}_from_path"})
        )

        df = meta.join(paths_df)
        df[self.cfg.class_col] = df[f"{self.cfg.class_col}_from_path"]
        df = df.drop(columns=[f"{self.cfg.class_col}_from_path"])
        assert not df.columns.duplicated().any()
        return df

    def execute(self) -> None:
        df = self._load_dataframe()
        for split in ["train", "val", "test"]:
            unk_dir = self.cfg.root / split / "UNKNOWN"
            if unk_dir.exists():
                print(f"Removing folder {unk_dir} …")
                shutil.rmtree(unk_dir) 
        before = len(df)
        df = df[df[self.cfg.class_col] != "UNKNOWN"].reset_index(drop=True)
        print(f"Removed {before - len(df)} UNKNOWN samples")

        df = df.dropna(subset=["filepath"]).copy()
        df["filepath"] = df["filepath"].astype("string")

        split_map = (
            df.groupby(self.cfg.class_col, group_keys=False)
            .apply(self.splitter.repartition)
            .set_index("idx")
        )
        df["split"] = split_map["split"]

        for idx, row in df.iterrows():
            old = self.cfg.root / Path(row["filepath"])
            if not old.exists():             
                continue
            new = self.cfg.root / row["split"] / row[self.cfg.class_col] / old.name
            new.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(old, new)
            except shutil.SameFileError:
                pass
            else:
                df.at[idx, "filepath"] = str(new.relative_to(self.cfg.root))


        df.to_csv(self.cfg.meta_csv, index=False)
        print("\n  Dataset restructured & CSV updated.\n")
        print(df.groupby(["split", self.cfg.class_col]).size().unstack(fill_value=0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-partition X-ray dataset.")
    p.add_argument(
        "--root", type=Path, default=Path("data/data"), help="Dataset root directory"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(root=args.root)

    if not cfg.root.exists():
        raise FileNotFoundError(f"Dataset root {cfg.root} does not exist.")
    if not cfg.meta_csv.exists():
        raise FileNotFoundError(f"Metadata CSV {cfg.meta_csv} does not exist.")

    print(f"\nRe-arranging dataset under {cfg.root} …\n")
    Rearranger(cfg).execute()
