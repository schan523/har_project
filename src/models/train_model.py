import torch
from torch import nn
from torchsummary import summary    
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import regnet50
from torch.utils.tensorboard import SummaryWriter
from models.model import BaseClassifier

def train(model, train_loader, optimizer, loss_fn):
    for i, data in enumerate(train_loader):
        model.train()
        inputs, labels = data  #inputs = X, labels = y

        optimizer.zero_grad()
        outputs = model(inputs)
        #pred = outputs.argmax(-1)

        #Calculating loss
        loss = loss_fn(outputs, labels)
        #accuracy = sum(pred==labels) / len(labels)
        loss.backward()
        optimizer.step()
        return loss#, accuracy

@torch.no_grad()
def validate(model, train_loader, optimizer, loss_fn):
    for i, data in enumerate(train_loader):
        model.eval()
        inputs, labels = data
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        #pred = outputs.argmax(-1)
        #accuracy = sum(pred==labels) / len(labels)
        return loss#, accuracy

#Main training loop
num_epochs = 10
writer = SummaryWriter()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BaseClassifier(15).to(device)
print(summary(model, (1, 224, 224)))
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        train_loss, train_accuracy = train(model, train_loader, optimizer, loss_fn)
        writer.add_scalar("Train Loss", train_loss, epoch)


#Confusion matrices



