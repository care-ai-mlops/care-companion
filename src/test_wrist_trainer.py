from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import argparse, time, copy, os, subprocess, shutil
from typing import Optional, List, Dict, Any, Tuple
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset 
from torchvision import transforms, models
import timm

import mlflow
import mlflow.pytorch

import ray
import ray.train as train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig

from CustomDatasetWrist import CustomDatasetWrist

from contextlib import contextmanager
import warnings
warnings.filterwarnings("ignore")


class MultiViewPreTrainedModel(nn.Module):
    def __init__(self,
                 backbone_configs: List[Dict[str, Any]], 
                 num_classes: int = 2,
                 dropout: float = 0.5,
                 img_size: int = 224,
                 pretrained: bool = True,
                ):
        super().__init__()

        self.dropout = dropout
        self._backbone_configs = backbone_configs
        self.num_views = len(backbone_configs)
        if self.num_views != 2:
             raise ValueError(f"MultiViewPreTrainedModel with CustomDatasetWrist expects exactly 2 views, but got {self.num_views} in backbone_configs.")

        self.backbones = nn.ModuleList()
        self.feature_sizes = []

        ctx  = train.get_context()
        rank = ctx.get_world_rank() if ctx else 0


        for config in self._backbone_configs:
            backbone_name = config.get("name")
            input_channels = config.get("input_channels", 1)

            if not backbone_name:
                raise ValueError(f"Missing backbone name in config: {config}")

            try:
                backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
                if hasattr(backbone, 'conv1'):
                    self._modify_input_conv(backbone, 'conv1', input_channels, pretrained)
                elif hasattr(backbone, 'patch_embed') and hasattr(backbone.patch_embed, 'proj'): 
                     self._modify_input_conv(backbone.patch_embed, 'proj', input_channels, pretrained)
                elif rank == 0:
                    print(f"Warning: Could not automatically modify input channels for {backbone_name}. Manual adaptation may be needed.")

                self.backbones.append(backbone)

                dummy_input = torch.randn(1, input_channels, img_size, img_size)
                try:
                     dummy_input = dummy_input.to('cpu') 
                     with torch.no_grad():
                        feature_size = self.backbones[-1].to('cpu')(dummy_input).numel() 
                     self.feature_sizes.append(feature_size)
                except Exception as e:
                     raise RuntimeError(f"Error determining output feature size for {backbone_name}: {e}")
                finally:
                     if torch.cuda.is_available():
                         self.backbones[-1].to(torch.device("cuda"))

            except Exception as e:
                raise ValueError(f"Error setting up backbone {backbone_name}: {str(e)}")

        fused_feature_size = sum(self.feature_sizes)

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(fused_feature_size, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, num_classes)
        )

    def _modify_input_conv(self, module, conv_attr_name: str, new_channels: int, pretrained: bool):
        """Helper to modify input conv layers to accept different channel counts"""
        original_conv = getattr(module, conv_attr_name)
        if original_conv.in_channels != new_channels:
            print(f"Changing input channels from {original_conv.in_channels} to {new_channels}")
            new_conv = nn.Conv2d(
                new_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=(original_conv.bias is not None)
            )

            if pretrained and original_conv.in_channels == 3 and new_channels == 1:
                with torch.no_grad():
                    new_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

            setattr(module, conv_attr_name, new_conv)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor: 
        assert len(inputs) == self.num_views, f"Expected {self.num_views} inputs, got {len(inputs)}"
        assert all([input.shape[1] == self._backbone_configs[i]["input_channels"] for i, input in enumerate(inputs)]), \
            f"Input channels mismatch: {[(input.shape[1], self._backbone_configs[i]['input_channels']) for i, input in enumerate(inputs)]}"

        features = []
        for i, view in enumerate(inputs):
            view_features = self.backbones[i](view)
            features.append(view_features)

        combined_features = torch.cat(features, dim=1)
        output = self.classifier(combined_features)

        return output


@dataclass
class TrainerConfig:
    backbone_configs: List[Dict[str, Any]] = field(default_factory=list)
    num_classes: int = 2
    dropout: float = 0.5
    pretrained: bool = True
    img_size: int = 224

    data_root: Path = Path("/mnt/object/wrist-data")
    batch_size: int = 32
    lr: float = 3e-4
    fine_tune_lr: float = 3e-5
    weight_decay: float = 1e-4
    initial_epochs: int = 5
    total_epochs: int = 20

    num_workers: int = 1
    gpu_per_worker: float = 1

    save_root: Path = Path("checkpoints/")
    mlflow_experiment_name: str = "wrist-fracture-classifier"


class RayTrainWristTrainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ctx = train.get_context()
        self.rank = ctx.get_world_rank() if ctx else 0

        if self.rank == 0:
            mlflow.set_experiment(self.cfg.mlflow_experiment_name)

        self.loaders = {}
        for split in ["train", "val", "test"]:
            ds = CustomDatasetWrist(
                root_dir=self.cfg.data_root,
                split=split,
                img_size=self.cfg.img_size,
                transform=(split == "train"),
            )
            if len(ds) == 0:
                if self.rank == 0:
                    print(f"[WARN] Empty split: {split}")
                continue
            dl = DataLoader(
                ds,
                batch_size=self.cfg.batch_size,
                shuffle=(split == "train"),
                num_workers=0,
                pin_memory=False,
                drop_last=(split == "train"),
            )
            self.loaders[split] = train.torch.prepare_data_loader(dl)

        self.model = MultiViewPreTrainedModel(
            backbone_configs=self.cfg.backbone_configs,
            num_classes=self.cfg.num_classes,
            dropout=self.cfg.dropout,
            pretrained=self.cfg.pretrained,
            img_size=self.cfg.img_size,
        )
        self.model = train.torch.prepare_model(self.model).to(self.device)
        self.criterion = nn.CrossEntropyLoss()


    @staticmethod
    def _accuracy(out: torch.Tensor, y: torch.Tensor) -> float:
        return (out.argmax(1) == y).float().mean().item()

    @staticmethod
    def _recall(preds: torch.Tensor, target: torch.Tensor, k: int) -> float:
        rec = []
        for c in range(k):
            tp = ((preds == c) & (target == c)).sum()
            pos = (target == c).sum()
            rec.append((tp / pos).item() if pos > 0 else 0.0)
        return float(sum(rec)) / k

    def _run_epoch(self, loader, train_mode):
        self.model.train(train_mode)
        loss_sum = acc_sum = tot = 0
        for img1, img2, y in loader:
            img1, img2, y = img1.to(self.device), img2.to(self.device), y.to(self.device)
            if train_mode:
                self.opt.zero_grad()

            out = self.model((img1, img2))
            loss = self.criterion(out, y)

            if train_mode:
                loss.backward()
                self.opt.step()

            bs = y.size(0)
            loss_sum += loss.item() * bs
            acc_sum += self._accuracy(out, y) * bs
            tot += bs

        return loss_sum / tot, acc_sum / tot


    def train(self):
        if self.rank == 0:
            run = mlflow.start_run(run_name="wrist-fracture-train")
            mlflow.log_params(vars(self.cfg))
            try:
                gpu_info = subprocess.check_output("nvidia-smi", text=True)
                mlflow.log_text(gpu_info, "gpu-info.txt")
            except Exception:
                pass

        for n, p in self.model.named_parameters():
            p.requires_grad = n.startswith("classifier")
        self.opt = train.torch.prepare_optimizer(
            optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
        )
        self.sched = optim.lr_scheduler.StepLR(
            self.opt, step_size=max(self.cfg.initial_epochs // 2, 1), gamma=0.1
        )

        best = -1.0
        for epoch in range(1, self.cfg.initial_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._run_epoch(self.loaders["train"], True)
            val_loss, val_acc = self._run_epoch(self.loaders["val"], False)
            self.sched.step()

            if self.rank == 0:
                mlflow.log_metrics(
                    {
                        "train_loss": tr_loss,
                        "train_acc": tr_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "lr": self.opt.param_groups[0]["lr"],
                    },
                    step=epoch,
                )
            train.report(
                {"val_accuracy": val_acc},
                checkpoint=train.Checkpoint.from_dict(
                    {"model_state_dict": self.model.state_dict()}
                ),
            )
            if val_acc > best:
                best = val_acc

        for p in self.model.parameters():
            p.requires_grad = True
        self.opt = train.torch.prepare_optimizer(
            optim.AdamW(self.model.parameters(), lr=self.cfg.fine_tune_lr)
        )
        self.sched = optim.lr_scheduler.StepLR(
            self.opt,
            step_size=max((self.cfg.total_epochs - self.cfg.initial_epochs) // 2, 1),
            gamma=0.1,
        )

        with mlflow.start_run(run_name="fine-tune", nested=True) if self.rank == 0 else _nullcontext():
            for epoch in range(self.cfg.initial_epochs + 1, self.cfg.total_epochs + 1):
                t0 = time.time()
                tr_loss, tr_acc = self._run_epoch(self.loaders["train"], True)
                val_loss, val_acc = self._run_epoch(self.loaders["val"], False)
                self.sched.step()

                if self.rank == 0:
                    mlflow.log_metrics(
                        {
                            "train_loss": tr_loss,
                            "train_acc": tr_acc,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "lr": self.opt.param_groups[0]["lr"],
                        },
                        step=epoch,
                    )
                train.report({"val_accuracy": val_acc})

        if self.rank == 0:
            mlflow.end_run()

    def evaluate_splits(self, splits: List[str]):
        results = {}
        if self.rank == 0:
            mlflow.start_run(run_name="evaluation", nested=True)

        for split in splits:
            if split not in self.loaders:
                if self.rank == 0:
                    print(f"[INFO] skip {split} – no loader")
                continue

            preds, gts = [], []
            for img1, img2, y in self.loaders[split]:
                img1, img2 = img1.to(self.device), img2.to(self.device)
                out = self.model((img1, img2))
                preds.append(out.argmax(1).cpu())
                gts.append(y)

            preds = torch.cat(preds)
            gts = torch.cat(gts)
            recall = self._recall(preds, gts, self.cfg.num_classes)
            loss, acc = self._run_epoch(self.loaders[split], False)

            results[split] = {"loss": loss, "accuracy": acc, "recall": recall}
            if self.rank == 0:
                mlflow.log_metrics(
                    {
                        f"{split}_loss": loss,
                        f"{split}_accuracy": acc,
                        f"{split}_recall": recall,
                    }
                )
        if self.rank == 0:
            mlflow.end_run()
        return results

@contextmanager
def _nullcontext():
    yield


def train_loop_func(cfg_dict: Dict[str, Any]):
    trainer = RayTrainWristTrainer(TrainerConfig(**cfg_dict))
    trainer.train()
    trainer.evaluate_splits(["test"])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default="/mnt/object/wrist-data/")
    parser.add_argument("--total_epochs", type=int, default=20)
    parser.add_argument("--initial_epochs", type=int, default=5)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--initial_lr", type=float, default=3e-4)
    parser.add_argument("--fine_tune_lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpu_per_worker", type=float, default=1)
    parser.add_argument("--mlflow_experiment_name", type=str, default="Wrist-Fracture-Classifier")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=2)
    args = parser.parse_args()

    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise RuntimeError("Set MLFLOW_TRACKING_URI environment variable")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    backbone = [
        {"name": "resnet18", "input_channels": 1},
        {"name": "efficientnet_b0", "input_channels": 1},
    ]

    cfg = TrainerConfig(
        backbone_configs=backbone,
        num_classes=args.num_classes,
        dropout=args.dropout,
        pretrained=not args.no_pretrain,
        data_root=args.data_root,
        batch_size=args.bs,
        lr=args.initial_lr,
        fine_tune_lr=args.fine_tune_lr,
        initial_epochs=args.initial_epochs,
        total_epochs=args.total_epochs,
        num_workers=args.num_workers,
        gpu_per_worker=args.gpu_per_worker,
        mlflow_experiment_name=args.mlflow_experiment_name,
        img_size=args.img_size,
    )
    KVM_FLOATING_IP = os.getenv("KVM_FLOATING_IP")
    if not ray.is_initialized():
        ray.init(address=f"ray://{KVM_FLOATING_IP}:10001")
        print("Ray initialized.")
    else:
        print("Ray already initialized.")
        
    trainer = TorchTrainer(
        train_loop_fn=train_loop_func,
        train_loop_config={k: str(v) if isinstance(v, Path) else v for k, v in vars(cfg).items()},
        scaling_config=ScalingConfig(
            num_workers=cfg.num_workers,
            use_gpu=(cfg.gpu_per_worker > 0),
        ),
        run_config=RunConfig(
            checkpoint_config=CheckpointConfig(checkpoint_frequency=5, num_to_keep=2)
        ),
    )

    res = trainer.fit()

    if ray.train.get_context().get_world_rank() == 0:
        best = res.get_best_checkpoint("val_accuracy", "max")
        if best:
            model = MultiViewPreTrainedModel(
                backbone_configs=cfg.backbone_configs,
                num_classes=cfg.num_classes,
                dropout=cfg.dropout,
                pretrained=False,
                img_size=cfg.img_size,
            )
            state = torch.load(best.to_directory() / "model_state_dict.pt", map_location="cpu")
            model.load_state_dict(state)
            mlflow.set_experiment(cfg.mlflow_experiment_name)
            with mlflow.start_run(run_name="register-model") as run:
                mlflow.pytorch.log_model(model, artifact_path="model")
                uri = f"runs:/{run.info.run_id}/model"
                mlflow.register_model(uri, "wrist-fracture-classifier-1")
                print("Model logged & registered:", uri)


if __name__ == "__main__":
    main()