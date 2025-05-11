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
    use_mlflow: bool = True
    mlflow_experiment_name: str = "wrist-fracture-classifier" 


class RayTrainWristTrainer: 
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config.use_mlflow:
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            ctx  = train.get_context()
            rank = ctx.get_world_rank() if ctx else 0
            if rank == 0:
                 mlflow.log_params(vars(self.config)) 
                 try:
                     gpu_info = next(
                         (subprocess.run(cmd, capture_output=True, text=True).stdout
                         for cmd in ["nvidia-smi", "rocm-smi"]
                         if subprocess.run(f"command -v {cmd}", shell=True, capture_output=True).returncode == 0),
                         "No GPU found."
                     )
                     mlflow.log_text(gpu_info, "gpu_info.txt")
                 except Exception as e:
                     print(f"Could not log GPU info: {e}")

        self.loaders = {}
        for split in ["train", "val", "test", "canary_testing_data", "production_data"]:
            dataset = CustomDatasetWrist(
                 root_dir=self.config.data_root,
                 split=split,
                 img_size=self.config.img_size,
                 transform=(split == "train")
             )
            if len(dataset) > 0:
                 self.loaders[split] = torch.utils.data.DataLoader(
                     dataset,
                     batch_size=self.config.batch_size,
                     shuffle=(split == "train"),
                     num_workers=0, 
                     pin_memory=False, 
                     drop_last=(split == "train") 
                 )
                 self.loaders[split] = train.torch.prepare_data_loader(self.loaders[split])
            else:
                ctx  = train.get_context()
                rank = ctx.get_world_rank() if ctx else 0
                if rank == 0:
                    print(f"Warning: Dataset for split '{split}' is empty. Skipping DataLoader creation.")

        self.model = MultiViewPreTrainedModel(
            backbone_configs=self.config.backbone_configs,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
            pretrained=self.config.pretrained,
            img_size=self.config.img_size 
        )
        self.criterion = nn.CrossEntropyLoss()

        self.model = train.torch.prepare_model(self.model)


    @staticmethod
    def _accuracy(out, y):
        preds  = out.argmax(dim=1).cpu()
        labels = y.cpu()
        return (preds == labels).float().mean().item()

    def _run_epoch(self, loader: DataLoader, train_mode: bool):
        ctx  = train.get_context()
        rank = ctx.get_world_rank() if ctx else 0

        self.model.train(train_mode)
        loss_sum = acc_sum = 0.0
        total_samples = 0

        loader_iter = tqdm(loader, 
                           desc="Training Batch" if train_mode else "Evaluation Batch", 
                           unit="batch", 
                           disable=(rank!= 0))

        with torch.set_grad_enabled(train_mode):
            for batch in loader_iter:
                img1, img2, labels = batch
                img1, img2 = img1.to(self.device), img2.to(self.device)
                inputs = (img1, img2)
                labels = labels.to(self.device)

                if train_mode:
                    self.optimizer.zero_grad()

                outputs = self.model(inputs) 
                loss = self.criterion(outputs, labels)

                if train_mode:
                    loss.backward()
                    self.optimizer.step()

                loss_sum += loss.item() * labels.size(0)
                acc_sum += self._accuracy(outputs, labels) * labels.size(0)
                total_samples += labels.size(0)


        return loss_sum / total_samples, acc_sum / total_samples

    def train(self):
        for name, param in self.model.named_parameters():
            if not name.startswith('module.classifier.') and not name.startswith('classifier.'):
                param.requires_grad = False
            else:
                 param.requires_grad = True
        ctx  = train.get_context()
        rank = ctx.get_world_rank() if ctx else 0
        if rank == 0:
            print("Backbones frozen for initial training.")
            trainable_params_initial = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Trainable parameters in initial phase: {trainable_params_initial}")
            if self.config.use_mlflow:
                 mlflow.log_metric("trainable_params_initial", trainable_params_initial, step=0)


        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                     lr=self.config.lr, 
                                     weight_decay=self.config.weight_decay)
        self.optimizer = train.torch.prepare_optimizer(self.optimizer)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=max(self.config.initial_epochs // 2, 1), gamma=0.1) 
        best_val_acc = -float('inf')

        for epoch in range(1, self.config.initial_epochs + 1):
            epoch_start_time = time.time()

            tr_loss, tr_acc = self._run_epoch(self.loaders["train"], train_mode=True)
            val_loss, val_acc = self._run_epoch(self.loaders["val"], train_mode=False)

            self.scheduler.step()
            epoch_time = time.time() - epoch_start_time
            if rank == 0:
                    print(f"Epoch {epoch + 1}/{self.config.total_epochs + 1}: "
                          f"Train Loss: {tr_loss:.4f}, Train Accuracy: {tr_acc:.4f}, "
                          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, "
                          f"Epoch Time: {epoch_time:.2f}s")
                

            train.report(
                {"train_loss": tr_loss, "train_accuracy": tr_acc,
                 "val_loss": val_loss, "val_accuracy": val_acc,
                 "learning_rate": self.optimizer.param_groups[0]["lr"],
                 "epoch_time": epoch_time},
                # Checkpoint will be managed by Ray Train based on RunConfig/CheckpointConfig

                checkpoint=train.Checkpoint.from_dict({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "best_val_acc": best_val_acc 
                 })
            )

            if val_acc > best_val_acc:
                 best_val_acc = val_acc
                 if rank == 0:
                    print(f"Epoch {epoch}: New best validation accuracy: {best_val_acc:.4f}")


        if self.config.total_epochs > self.config.initial_epochs:
            if train.get_context().get_world_rank() == 0:
                print("\nUnfreezing backbones and starting fine-tuning.")
            for param in self.model.parameters():
                param.requires_grad = True 
            if rank == 0:
                print(f"Unfreezing backbones for fine-tuning. Total trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.fine_tune_lr, weight_decay=self.config.weight_decay)
            self.optimizer = train.torch.prepare_optimizer(self.optimizer)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=max((self.config.total_epochs - self.config.initial_epochs) // 2, 1), gamma=0.1) 


            if rank == 0 and self.config.use_mlflow:
                      mlflow.log_metric("trainable_params_finetune", sum(p.numel() for p in self.model.parameters() if p.requires_grad), step=self.config.initial_epochs)

            if rank == 0: 
                print(f"Starting fine-tuning for {self.config.total_epochs - self.config.initial_epochs} epochs.")

            for epoch in range(self.config.initial_epochs + 1, self.config.total_epochs + 1):
                epoch_start_time = time.time()

                tr_loss, tr_acc = self._run_epoch(self.loaders["train"], train_mode=True)
                val_loss, val_acc = self._run_epoch(self.loaders["val"], train_mode=False)

                self.scheduler.step()
                epoch_time = time.time() - epoch_start_time

                if rank == 0:
                    print(f"Epoch {epoch + 1}/{self.config.total_epochs + 1}: "
                          f"Train Loss: {tr_loss:.4f}, Train Accuracy: {tr_acc:.4f}, "
                          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, "
                          f"Epoch Time: {epoch_time:.2f}s")
                
                train.report(
                    {"train_loss": tr_loss, "train_accuracy": tr_acc,
                     "val_loss": val_loss, "val_accuracy": val_acc,
                     "learning_rate": self.optimizer.param_groups[0]["lr"],
                     "epoch_time": epoch_time},
                    # Checkpoint will be managed by Ray Train based on RunConfig/CheckpointConfig
                    checkpoint=train.Checkpoint.from_dict({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "best_val_acc": best_val_acc
                    })
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if rank == 0:
                       print(f"Epoch {epoch}: New best validation accuracy: {best_val_acc:.4f}")


    def evaluate_splits(self, splits_to_evaluate: List[str]):
        """
        Evaluates the model on the specified data splits.
        Can be called after training is complete.
        """
        ctx  = train.get_context()
        rank = ctx.get_world_rank() if ctx else 0
        if rank == 0:
            print("\nStarting evaluation on specified splits.")

        results = {}
        for split in splits_to_evaluate:
            if split in self.loaders and len(self.loaders[split].dataset) > 0:
                if rank == 0:
                    print(f"Evaluating on {split} split...")
                loss, acc = self._run_epoch(self.loaders[split], train_mode=False)
                results[split] = {"loss": loss, "accuracy": acc}

                if rank == 0:
                    print(f"{split:>20}: loss {loss:.4f}  acc {acc:.3f}")

                if self.config.use_mlflow and train.get_context().get_world_rank() == 0:
                     mlflow.log_metric(f"{split}_loss", loss)
                     mlflow.log_metric(f"{split}_accuracy", acc)

            else:
                 if rank == 0: 
                    print(f"Skipping evaluation for empty or missing split: {split}")

        return results



def train_loop_func(train_loop_config: Dict):

    config = TrainerConfig(**train_loop_config)
    trainer_instance = RayTrainWristTrainer(config)

    trainer_instance.train()

    evaluation_splits = ["test", "canary_testing_data", "production_data"]
    trainer_instance.evaluate_splits(evaluation_splits)


def main():

    parser = argparse.ArgumentParser("Multi-View Wrist Fracture Trainer with Ray Train")
    parser.add_argument("--data_root", type=Path, default=Path("/mnt/object/wrist-xray-data"))
    parser.add_argument("--total_epochs", type=int, default=20)
    parser.add_argument("--initial_epochs", type=int, default=5)
    parser.add_argument("--save_root", type=Path, default=Path("/models/wrist-checkpoints/"))
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--initial_lr", type=float, default=3e-4)
    parser.add_argument("--fine_tune_lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of Ray Train workers")
    parser.add_argument("--gpu_per_worker", type=float, default=1, help="GPUs per Ray Train worker")
    parser.add_argument("--mlflow_experiment_name", type=str, default="Wrist-Fracture-Classifier")
    parser.add_argument("--img_size", type=int, default=224, help="Image size for resizing")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for classification")

    args = parser.parse_args()

    backbone_configs = [
        {"name": "resnet18", "input_channels": 1, "projection": 1},
        {"name": "efficientnet_b0", "input_channels": 1, "projection": 2},
    ]

    train_config = TrainerConfig(
        backbone_configs=backbone_configs,
        num_classes=args.num_classes,
        dropout=args.dropout,
        pretrained=not args.no_pretrain,
        data_root=args.data_root,
        batch_size=args.bs,
        lr=args.initial_lr,
        fine_tune_lr=args.fine_tune_lr,
        initial_epochs=args.initial_epochs,
        total_epochs=args.total_epochs,
        save_root=args.save_root,
        num_workers=args.num_workers,
        gpu_per_worker=args.gpu_per_worker,
        mlflow_experiment_name=args.mlflow_experiment_name,
        img_size=args.img_size,
    )

    # Initialize Ray
    KVM_FLOATING_IP = os.getenv("KVM_FLOATING_IP")
    if not ray.is_initialized():
        ray.init(address="ray://{KVM_FLOATING_IP}:10001")
        print("Ray initialized.")
    else:
        print("Ray already initialized.")

    # Set up scaling
    scaling_config = ScalingConfig(
        num_workers=train_config.num_workers,
        use_gpu=(train_config.gpu_per_worker > 0),
    )

    # Serialize Path fields to str
    tlc = vars(train_config).copy()
    tlc["data_root"] = str(train_config.data_root)
    tlc["save_root"] = str(train_config.save_root)

    trainer = TorchTrainer(
        train_loop_fn=train_loop_func,
        train_loop_config=tlc,
        scaling_config=scaling_config,
        run_config=RunConfig(
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=5,
                num_to_keep=2,
            )
        ),
    )

    # get world rank (0 for driver / head)
    ctx = ray.train.get_context()
    rank = ctx.get_world_rank() if ctx else 0

    if rank == 0:
        print("Starting Ray Train training...")
    result = trainer.fit()
    if rank == 0:
        print("Ray Train training finished.")

    # Load best checkpoint
    best_ckpt = result.get_best_checkpoint("val_accuracy", mode="max")
    if best_ckpt and rank == 0:
        ckpt_dir = best_ckpt.to_directory()
        state_path = os.path.join(ckpt_dir, "model_state_dict.pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(state_path, map_location=device)

        print(f"Best checkpoint found at: {ckpt_dir}")
        model = MultiViewPreTrainedModel(
            backbone_configs=train_config.backbone_configs,
            num_classes=train_config.num_classes,
            dropout=train_config.dropout,
            pretrained=False,
            img_size=train_config.img_size,
        )
        model.load_state_dict(state_dict)
        model.to(device).eval()

        # Final evaluation on test / canary / production
        loss_fn = torch.nn.CrossEntropyLoss()
        for split in ["test", "canary_testing_data", "production_data"]:
            ds = CustomDatasetWrist(
                root_dir=str(train_config.data_root),
                split=split,
                img_size=train_config.img_size,
                transform=False,
            )
            if len(ds) == 0:
                print(f"Skipping empty split: {split}")
                continue

            dl = DataLoader(
                ds,
                batch_size=train_config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )

            total_loss = total_correct = total_samples = 0
            with torch.no_grad():
                for img1, img2, labels in dl:
                    img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                    outputs = model((img1, img2))
                    loss = loss_fn(outputs, labels)

                    total_loss += loss.item() * labels.size(0)
                    preds = outputs.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)

            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            print(f"Evaluation on {split}: Loss {avg_loss:.4f}, Accuracy {accuracy:.3f}")

            # Log to MLflow
            mlflow.log_metric(f"final_eval_{split}_loss", avg_loss)
            mlflow.log_metric(f"final_eval_{split}_accuracy", accuracy)

    # ray.shutdown()  # if you want to cleanly shut down Ray when local

if __name__ == "__main__":
    main()
