import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models

#pre-trained model adapted for our task
n_classes = 15

class BaseClassifier(nn.Module):
    def __init__(self):
        super().__init__(self, n_classes)
        model = models.resnet50(weights="IMAGENET1K_V1")
        modules = list(model.children())[:-1] #deleting fully connected layer
        self.resnet = nn.Sequential(*modules)

        #freeze pre-trained layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(model.fc.in_features, n_classes)

    def forward(self, x):
        x = self.resnet(x)
        logits = self.fc(x)
        return logits
