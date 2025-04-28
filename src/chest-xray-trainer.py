from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple
import time, copy, argparse, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from CustomDatasetXray import CustomDatasetXray 

import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")


class ResNetClassifier(nn.Module):
    """ResNet-18 backbone → custom FC head for N classes."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = (
            models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = models.resnet18(weights=weights)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feat, num_classes)

    def forward(self, x):
        return self.backbone(x)

@dataclass
class Trainer:
    model: nn.Module
    loaders: dict[str, DataLoader]
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr: float = 3e-4
    epochs: int = 10

    def __post_init__(self):
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)


    @staticmethod
    def _accuracy(out, y):
        return (out.argmax(1) == y).float().mean().item()

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

    def fit(self, ckpt_path: Path = Path("best_resnet.pt")):
        best_acc, best_wts = 0.0, copy.deepcopy(self.model.state_dict())

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._run_epoch("train", train=True)
            val_loss, val_acc = self._run_epoch("val",   train=False)
            self.scheduler.step()

            print(f"Epochs: [{epoch:02d}/{self.epochs}] "
                  f"Train Loss: {tr_loss:.4f} Train Accuracy:{tr_acc:.3f}  "
                  f"val Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc:.3f}  "
                  f"Epoch Time: {time.time()-t0:.1f}s")

            if val_acc > best_acc:
                best_acc, best_wts = val_acc, copy.deepcopy(self.model.state_dict())

        torch.save(best_wts, ckpt_path)
        print(f"Best val-acc {best_acc:.3f} saved to {ckpt_path}")
        self.model.load_state_dict(best_wts)

    def evaluate(self, split: str):
        loss, acc = self._run_epoch(split, train=False)
        print(f"{split:>20}: loss {loss:.4f}  acc {acc:.3f}")
        return loss, acc


def main():
    p = argparse.ArgumentParser("OOP ResNet18 trainer")
    p.add_argument("--root", type=Path, default=Path("data/data"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--no-pretrain", action="store_true")
    args = p.parse_args()

    data = CustomDatasetXray(root=args.root, batch_size=args.bs, augment=True)
    loaders, mapping = data.loaders()
    n_classes = len(mapping)
    print(mapping) 
    print(f"Number of classes: {n_classes}")
    print(f"Number of training samples: {len(loaders['train'].dataset)}")
    print(f"Number of validation samples: {len(loaders['val'].dataset)}")
    print(f"Number of test samples: {len(loaders['test'].dataset)}")
    print(f"Number of canary testing samples: {len(loaders['canary_testing_data'].dataset)}")
    print(f"Number of production samples: {len(loaders['production_data'].dataset)}")

    net = ResNetClassifier(n_classes, pretrained=not args.no_pretrain)
    trainer = Trainer(net, loaders, lr=args.lr, epochs=args.epochs)

    trainer.fit()

    for split in ["test", "canary_testing_data", "production_data"]:
        trainer.evaluate(split)
    print("Training complete.")
