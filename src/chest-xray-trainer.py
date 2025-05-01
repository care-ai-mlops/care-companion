from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple
import time, copy, argparse, os, subprocess
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from CustomDatasetXray import CustomDatasetXray 
import mlflow
import mlflow.pytorch

import warnings
warnings.filterwarnings("ignore")


class PreTrainedClassifier(nn.Module):
    """ResNet-18 backbone → custom FC head for N classes."""

    def __init__(self, 
                 num_classes: int,
                 dropout: float = 0.5, 
                 pretrained: bool = True,
                 model_backbone: Optional[str] = "resnet18",
                 ) -> None:
        super().__init__()
        self.dropout = dropout
        if model_backbone == "resnet18" and pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        elif model_backbone == "resnet18" and not pretrained:
            weights = None
        elif model_backbone == "resnet50" and pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        elif model_backbone == "resnet50" and not pretrained:
            weights = None
        else:
            raise ValueError(f"Unsupported model backbone: {model_backbone}")
            
        self.backbone = models.resnet18(weights=weights)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_feat, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(256, num_classes)
        )
        self.classifier = self.backbone.fc

    def forward(self, x):
        return self.backbone(x)

@dataclass
class Trainer:
    model: nn.Module
    loaders: dict[str, DataLoader]
    save_root: Path = Path("checkpoints/")
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr: float = 3e-4
    initial_epochs: int = 5
    total_epochs: int = 20
    patience: int = 5
    fine_tune_lr: float = 3e-5
    dropout: float = 0.5
    use_mlflow: bool = True

    def __post_init__(self):
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        if self.use_mlflow:
            mlflow.set_experiment("chest-xray-classifier")


    @staticmethod
    def _accuracy(out, y):
        return (out.argmax(1) == y).float().mean().item()
    
    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
                param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        print("Backbone frozen. Only classifier parameters will be trained.")

    def unfreeze_backbone(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = True
        print("Backbone unfrozen. All parameters will be trained.")

    def _run_epoch(self, split: str, train: bool):
        loader = self.loaders[split]
        self.model.train(train)
        loss_sum = acc_sum = 0.0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                loss_sum += loss.item() * x.size(0)
                acc_sum  += self._accuracy(out, y) * x.size(0)

        n = len(loader.dataset)
        return loss_sum / n, acc_sum / n

    def fit_initial(self, use_chek_pt: bool = False):
        self.freeze_backbone()
        trainable_params_ch = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        best_acc, best_wts = 0.0, copy.deepcopy(self.model.state_dict())
          
        with mlflow.start_run(run_name="initial_train", log_system_metrics=True):
            gpu_info = next( 
                (subprocess.run(cmd, capture_output=True, text=True).stdout 
                for cmd in ["nvidia-smi", "rocm-smi"] 
                if subprocess.run(f"command -v {cmd}", shell=True, capture_output=True).returncode == 0),
                "No GPU found."
            )
            mlflow.log_params({
                "phase": "initial",
                "model": self.model.__class__.__name__,
                "lr": self.lr,
                "initial_epochs": self.initial_epochs,
                "total_epochs": self.total_epochs,
                "batch_size": self.loaders["train"].batch_size,
                "optimizer": self.optimizer.__class__.__name__,
                "scheduler": self.scheduler.__class__.__name__,
                "dropout": self.dropout,
                "trainable_params": trainable_params_ch
            })
            mlflow.log_text(gpu_info, "gpu-info.txt")
            for epoch in range(1, self.initial_epochs):
                if use_chek_pt:
                    ckpt_path = self.save_root / f"best_resnet.pt"
                    if os.path.exists(ckpt_path):
                        self.model.load_state_dict(torch.load(ckpt_path))
                        print(f"Checkpoint loaded from {ckpt_path}")
                t0 = time.time()
                tr_loss, tr_acc = self._run_epoch("train", train=True)
                val_loss, val_acc = self._run_epoch("val",   train=False)
                self.scheduler.step()

                print(f"Epochs: [{epoch:02d}/{self.total_epochs}]  "
                    f"Train Loss: {tr_loss:.4f} Train Accuracy:{tr_acc:.4f}  "
                    f"val Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc:.4f}  "
                    f"Epoch Time: {time.time()-t0:.1f}s")
                
                mlflow.log_metrics(
                    {"epoch_time": time.time()-t0, 
                    "train_loss": tr_loss, 
                    "train_accuracy": tr_acc, 
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "trainable_params": trainable_params_ch}, step=epoch
                    )

                if val_acc > best_acc:
                    best_acc, best_wts = val_acc, copy.deepcopy(self.model.state_dict())
                    ckpt_dir = self.save_root
                    ckpt_dir.mkdir(exist_ok=True)
                    ckpt_path = ckpt_dir / "best_resnet.pt"
                    torch.save(best_wts, ckpt_path)

                    mlflow.log_artifacts(str(ckpt_path), artifact_path="Models")
                    print(f"Best val-acc {best_acc:.4f} saved to {ckpt_path}")

            self.model.load_state_dict(best_wts)
    
    def fit_fine_tune(self, use_chek_pt: bool = False):
        self.unfreeze_backbone()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.fine_tune_lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        trainable_params_ch = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        best_acc, best_wts = 0.0, copy.deepcopy(self.model.state_dict())

        with mlflow.start_run(run_name="fine_tune", log_system_metrics=True):
            gpu_info = next( 
                (subprocess.run(cmd, capture_output=True, text=True).stdout 
                for cmd in ["nvidia-smi", "rocm-smi"] 
                if subprocess.run(f"command -v {cmd}", shell=True, capture_output=True).returncode == 0),
                "No GPU found."
            )
            mlflow.log_params({
                "phase": "fine_tune",
                "model": self.model.__class__.__name__,
                "lr": self.lr,
                "initial_epochs": self.initial_epochs,
                "total_epochs": self.total_epochs,
                "batch_size": self.loaders["train"].batch_size,
                "optimizer": self.optimizer.__class__.__name__,
                "scheduler": self.scheduler.__class__.__name__,
                "dropout": self.dropout,
                "trainable_params": trainable_params_ch
            })
            mlflow.log_text(gpu_info, "gpu-info.txt")

            for epoch in range(self.initial_epochs + 1, self.total_epochs + 1):
                if use_chek_pt:
                    ckpt_path = self.save_root / f"Models" / f"best_resnet.pt"
                    if os.path.exists(ckpt_path):
                        self.model.load_state_dict(torch.load(ckpt_path))
                        print(f"Checkpoint loaded from {ckpt_path}")
                    
                t0 = time.time()
                tr_loss, tr_acc = self._run_epoch("train", train=True)
                val_loss, val_acc = self._run_epoch("val",   train=False)
                self.scheduler.step()

                print(f"Epochs: [{epoch:02d}/{self.total_epochs}] "
                    f"Train Loss: {tr_loss:.4f} Train Accuracy:{tr_acc:.3f}  "
                    f"val Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc:.3f}  "
                    f"Epoch Time: {time.time()-t0:.1f}s")
                mlflow.log_metrics(
                    {"epoch_time": time.time()-t0, 
                    "train_loss": tr_loss, 
                    "train_accuracy": tr_acc, 
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "trainable_params": trainable_params_ch}, step=epoch
                    )

                if val_acc > best_acc:
                    best_acc, best_wts = val_acc, copy.deepcopy(self.model.state_dict())
                    ckpt_dir = self.save_root
                    ckpt_dir.mkdir(exist_ok=True)
                    ckpt_path = ckpt_dir / "best_resnet.pt"
                    torch.save(best_wts, ckpt_path)

                    mlflow.log_artifacts(str(ckpt_path), artifact_path="Models")
                    print(f"Best val-acc {best_acc:.3f} saved to {ckpt_path}")

            self.model.load_state_dict(best_wts)


    def evaluate(self, split: str):
        loss, acc = self._run_epoch(split, train=False)
        print(f"{split:>20}: loss {loss:.4f}  acc {acc:.3f}")
        with mlflow.start_run(run_name="evaluate", log_system_metrics=True):
            mlflow.log_params({
                "phase": "evaluate",
                "split": split,
                "model": self.model.__class__.__name__,
            })
            mlflow.log_metrics(
                {f"{split}_loss": loss, 
                f"{split}_accuracy": acc}
                )
        return loss, acc


def main():
    p = argparse.ArgumentParser("Chest Data trainer")
    p.add_argument("--root", type=Path, default=Path("/mnt/object/chest-data"))
    p.add_argument("--total_epochs", type=int, default=20)
    p.add_argument("--initial_epochs", type=int, default=5)
    p.add_argument("--model_backbone", type=str, default="resnet18")
    p.add_argument("--save_root", type=Path, default=Path("/models/checckpoints/"))
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--initial_lr", type=float, default=3e-4)
    p.add_argument("--fine_tune_lr", type=float, default=3e-5)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--no-pretrain", action="store_true")
    args = p.parse_args()
    print("Inside Trainer File")
    data = CustomDatasetXray(root=args.root, batch_size=args.bs, augment=True)
    loaders, mapping = data.get_loaders()
    n_classes = len(mapping)

    print(mapping)
    print(f"Number of classes: {n_classes}")
    print(f"Number of training samples: {len(loaders['train'].dataset)}")
    print(f"Number of validation samples: {len(loaders['val'].dataset)}")
    print(f"Number of test samples: {len(loaders['test'].dataset)}")
    print(f"Number of canary testing samples: {len(loaders['canary_testing_data'].dataset)}")
    print(f"Number of production samples: {len(loaders['production_data'].dataset)}")

    net = PreTrainedClassifier(
        num_classes=n_classes, 
        pretrained=not args.no_pretrain, 
        model_backbone=args.model_backbone,
        dropout=args.dropout
    )

    trainer = Trainer(
        model=net,
        loaders=loaders,
        lr=args.initial_lr,
        fine_tune_lr=args.fine_tune_lr,
        initial_epochs=args.initial_epochs,
        total_epochs=args.total_epochs,
        dropout=0.5
    )

    trainer.fit_initial()
    trainer.fit_fine_tune()

    for split in ["test", "canary_testing_data", "production_data"]:
        trainer.evaluate(split)

    print("Training complete.")

if __name__ == "__main__":
    main()

''' 
if epoch == self.initial_epochs + 1:
self.unfreeze_backbone()
self.optimizer = optim.AdamW(self.model.parameters(), lr=self.fine_tine_lr, weight_decay=1e-4)
self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
trainable_params_ft = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
'''