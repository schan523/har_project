import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import regnet50
from models.model import NeuralNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"

#Creating an instance of the model
model = NeuralNetwork().to(device)