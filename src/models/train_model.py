import torch
from torch import nn
from torchsummary import summary    
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from data.make_dataset import ImageDataSet
from sklearn.model_selection import train_test_split
from .model import BaseClassifier

#data preprocessing
# Data augmentation for training set
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
])

# No data augmentation for testing set
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# DataSet and DataLoader
train_val_dataset = ImageDataSet(
    csv_file="/content/drive/MyDrive/Human Action Recognition/Training_set.csv",
    root_dir="/content/drive/MyDrive/Human Action Recognition/train/",
    transform=train_transform
)

#Split into train and validation sets
train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=0.18)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(dataset=val_dataset, batch_size=32, drop_last=True, shuffle=True, num_workers=0)
#pre-trained model adapted for our task
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



