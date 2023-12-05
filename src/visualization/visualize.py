import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# obtain model directory for import
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join('models')
sys.path.append(model_dir)
from model_dir import BaseClassifier  # import classifier 

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_data():
    #TODO I don't know how to preprocess 
    return 

#VISUAL 1: method for plotting individual feature maps
def plot_feature_maps(model, loader):
    model.eval()
    images, _ = next(iter(loader))
    with torch.no_grad():
        feature_maps = model(images.to(get_device()), return_feature_maps=True)

    for fmap in feature_maps:
        if fmap.dim() == 4:  # plot feature maps with 4 dimensions (CNN format)
            fmap = fmap.cpu().numpy()
            for i in range(min(5, fmap.shape[1])):  # Plot first 5 feature maps for example
                plt.imshow(fmap[0][i], cmap='gray')
                plt.show()



