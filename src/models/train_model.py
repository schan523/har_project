import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import regnet50

device = "cuda" if torch.cuda.is_available() else "cpu"