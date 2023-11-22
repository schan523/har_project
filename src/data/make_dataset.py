import pandas as pd
import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

train_csv = pd.read_csv(".\data\Testing_set.csv")
test_csv = pd.read_csv(".\data\Testing_set.csv")

#labels to numbers
label_names = {'sitting':0,'using_laptop':1, 'hugging':2, 'sleeping':3,'drinking':4,'clapping':5, 'dancing':6, 'cycling':7,'calling':8,'laughing':9, 'eating':10,'fighting':11,'listening_to_music':12,'running':13,'texting':14}

# Custom image Dataset
class ImageDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir

        self.df = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            # std multiply by 255 to convert img of [0, 255]
            # to img of [0, 1]
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229*255, 0.224*255, 0.225*255))]
        )
        self.img_name = self.df['filename']
        self.transform = transforms
        #test doesnt have column


    def __len__(self):
        return self.df.shape[0] #self.y.shape[0]?

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root_dir, self.img_name[idx]))

        if self.transform is not None:
          img = self.transform(img)

        #label = torch.tensor(self.df['label'][idx], dtype=torch.long)  # Convert label to a PyTorch tensor
        label = self.df['label'][idx]
        label = label_to_number.get(label, -1)
        return img, label


