import torch
from torch import nn
from torchsummary import summary    
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import regnet50
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from models.model import BaseClassifier

def train(model, train_loader, optimizer, loss_fn):
    for i, data in enumerate(train_loader):
        model.train()
        inputs, labels = data  #inputs = X, labels = y

        optimizer.zero_grad()
        outputs = model(inputs)
        pred = outputs.argmax(-1)

        #Calculating loss
        loss = loss_fn(outputs, labels)
        accuracy = sum(pred==labels) / len(labels)
        loss.backward()
        optimizer.step()
        return loss, accuracy

@torch.no_grad()
def validate(model, val_loader, loss_fn):
    for i, data in enumerate(val_loader):
        model.eval()
        inputs, labels = data
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        pred = outputs.argmax(-1)
        accuracy = sum(pred==labels) / len(labels)
        return loss, accuracy

#Main training loop
num_epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BaseClassifier(15).to(device)
print(summary(model, (1, 224, 224)))

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0


for epoch in range(num_epochs):
    print('EPOCH {}:'.format(epoch_number + 1))
    for i, data in enumerate(train_loader):
        train_loss, train_accuracy = train(model, train_loader, optimizer, loss_fn)
        writer.add_scalar("Train Loss: ", train_loss, epoch)
        writer.add_scalar("Train accuracy: ", train_accuracy, epoch)
    
    for i, data in enumerate(val_loader):
        val_loss, val_accuracy = validate(model, val_loader, loss_fn)
        writer.add_scalar("Validation loss: ", val_loss, epoch)
        writer.add_scalar("Validation accuracy: ", val_accuracy, epoch)

    writer.flush()
    epoch_number += 1

model_path = 'model_{}_{}'.format(timestamp, epoch_number)
torch.save(model.state_dict(), model_path)

#Confusion matrices



