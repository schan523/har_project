import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

#pre-trained model adapted for our task
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(weights="IMAGENET1K_V1")
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 15)



