import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import regnet50
from models.model import BaseClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BaseClassifier(15).to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

def train():
    for i, data in enumerate(train_dataloader):
        inputs, labels = data  #inputs = X, labels = y

        optimizer.zero_grad()
        outputs = model(inputs)
        pred = outputs.argmax(-1)

        #Calculating loss
        loss = loss_fn(pred, labels)
        accuracy = sum(pred==labels) / len(labels)
        loss.backward()
        optimizer.step()
        
        return loss, accuracy

