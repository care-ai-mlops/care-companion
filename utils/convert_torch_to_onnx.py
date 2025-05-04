from chest_xray_trainer import PreTrainedClassifier
import torch

model_state_dict_path = "../models/torch/best_resnet.pt"
model_state_dict = torch.load(model_state_dict_path, map_location="cpu")

model = PreTrainedClassifier(num_classes=3)
model.load_state_dict(model_state_dict)
model.eval()

