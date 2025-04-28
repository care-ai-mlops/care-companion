#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
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
        combined = torch.cat([f1, f2], dim=1)
        return self.head(combined)

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

    def train(self):
        best_acc = 0.0
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            running_loss, running_corrects, total = 0.0, 0, 0
            for batch in self.loaders["train"]:
                x1 = batch["proj1"].to(self.device)
                x2 = batch["proj2"].to(self.device)
                y = batch["label"].to(self.device)

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
            epoch_acc = running_corrects / total
            print(f"[Epoch {epoch}/{self.num_epochs}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            val_loss, val_acc = self.validate()
            print(f"[Epoch {epoch}/{self.num_epochs}] Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), self.save_path)

            if self.scheduler:
                self.scheduler.step()

        print(f"Training complete. Best val Acc: {best_acc:.4f}")

    def validate(self) -> tuple[float, float]:
        self.model.eval()
        running_loss, running_corrects, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in self.loaders["val"]:
                x1 = batch["proj1"].to(self.device)
                x2 = batch["proj2"].to(self.device)
                y = batch["label"].to(self.device)

                o = self.model(x1, x2)
                loss = self.criterion(o, y)

                running_loss += loss.item() * x1.size(0)
                preds = o.argmax(dim=1)
                running_corrects += (preds == y).sum().item()
                total += y.size(0)

        return (running_loss / len(self.loaders["val"].dataset), running_corrects / total)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default="data")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--output", type=Path, default="best_model.pth")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
    ])

    train_ds = CustomDatasetXray(args.data_dir, split="train", transform=train_tf)
    val_ds = CustomDatasetXray(args.data_dir, split="val", transform=val_tf)
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4),
    }

    model = TwoProjectionResNet(args.num_classes, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    trainer = Trainer(model=model, loaders=loaders, device=device, optimizer=optimizer,
                      criterion=criterion, num_epochs=args.epochs,
                      scheduler=scheduler, save_path=args.output)
    trainer.train()

if __name__ == "__main__":
    main()
