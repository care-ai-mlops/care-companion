from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse, time, copy

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

import mlflow
import mlflow.pytorch

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from CustomDatasetXray import CustomDatasetXray

class TwoProjectionResNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(in_feat * 2, num_classes)

    def forward(self, x1, x2):
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        f = torch.cat([f1, f2], dim=1)
        return self.head(f)

@dataclass
class Trainer:
    model: nn.Module
    loaders: dict[str, DataLoader]
    device: torch.device
    optimizer: optim.Optimizer
    criterion: nn.Module
    num_epochs: int
    scheduler: optim.lr_scheduler._LRScheduler | None = None
    save_path: Path = Path("best_model.pth")
    use_mlflow: bool = False

    def train(self):
        best_acc = 0.0
        if self.use_mlflow:
            mlflow.start_run()
            mlflow.log_params({
                "batch_size": self.loaders["train"].batch_size,
                "lr": self.optimizer.param_groups[0]['lr'],
                "epochs": self.num_epochs
            })

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            running_loss, running_corrects, total = 0.0, 0, 0
            for batch in self.loaders["train"]:
                x1 = batch["proj1"].to(self.device)
                x2 = batch["proj2"].to(self.device)
                y  = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                o = self.model(x1, x2)
                loss = self.criterion(o, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * x1.size(0)
                preds = o.argmax(dim=1)
                running_corrects += (preds == y).sum().item()
                total += y.size(0)

            epoch_loss = running_loss / len(self.loaders["train"].dataset)
            epoch_acc  = running_corrects / total

            val_loss, val_acc = self.validate()

            if self.use_mlflow:
                mlflow.log_metrics({
                    "train_loss": epoch_loss,
                    "train_acc": epoch_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                }, step=epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), self.save_path)

            if self.scheduler:
                self.scheduler.step()

        if self.use_mlflow:
            mlflow.pytorch.log_model(self.model, artifact_path="model")
            mlflow.end_run()

    def validate(self) -> tuple[float, float]:
        self.model.eval()
        running_loss, running_corrects, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in self.loaders["val"]:
                x1 = batch["proj1"].to(self.device)
                x2 = batch["proj2"].to(self.device)
                y  = batch["label"].to(self.device)

                o = self.model(x1, x2)
                loss = self.criterion(o, y)

                running_loss += loss.item() * x1.size(0)
                preds = o.argmax(dim=1)
                running_corrects += (preds == y).sum().item()
                total += y.size(0)

        return (
            running_loss / len(self.loaders["val"].dataset),
            running_corrects / total
        )

def train_ray(config, data_dir="data", num_epochs=10, num_classes=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225]),
    ])

    train_ds = CustomDatasetXray(data_dir, split="train", transform=train_tf)
    val_ds   = CustomDatasetXray(data_dir, split="val",   transform=val_tf)
    loaders = {
        "train": DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True,  num_workers=4),
        "val":   DataLoader(val_ds,   batch_size=int(config["batch_size"]), shuffle=False, num_workers=4),
    }

    model = TwoProjectionResNet(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    trainer = Trainer(
        model=model,
        loaders=loaders,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        save_path=Path(f"model_{time.time()}.pth"),
        use_mlflow=True
    )
    trainer.train()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    type=Path, default="data")
    p.add_argument("--batch_size",  type=int,  default=16)
    p.add_argument("--epochs",      type=int,  default=25)
    p.add_argument("--lr",          type=float,default=1e-4)
    p.add_argument("--num_classes", type=int,  default=2)
    p.add_argument("--use_ray",     action="store_true")
    return p.parse_args()

def main():
    args = parse_args()

    if args.use_ray:
        config = {
            "lr": tune.loguniform(1e-5, 1e-3),
            "batch_size": tune.choice([16, 32, 64]),
        }
        scheduler = ASHAScheduler(metric="val_acc", mode="max", max_t=args.epochs)
        reporter = CLIReporter(metric_columns=["val_acc", "training_iteration"])

        tune.run(
            tune.with_parameters(train_ray, data_dir=str(args.data_dir), num_epochs=args.epochs, num_classes=args.num_classes),
            resources_per_trial={"cpu": 4, "gpu": 1},
            config=config,
            num_samples=10,
            scheduler=scheduler,
            progress_reporter=reporter,
            name="ray_tune_xray"
        )
    else:
        config = {
            "lr": args.lr,
            "batch_size": args.batch_size
        }
        train_ray(config, data_dir=str(args.data_dir), num_epochs=args.epochs, num_classes=args.num_classes)

if __name__ == "__main__":
    main()
