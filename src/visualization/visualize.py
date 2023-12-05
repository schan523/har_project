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

# Performance setting
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu" 

# load and return BaseClassifier model 
def load_model(model_path, n_classes):
    model = BaseClassifier(n_classes)
    model.load_state_dict(torch.load(model_path, map_location=get_device()))
    model.to(get_device())
    return model

# load dataset, preproc functions
def load_data():
    #TODO load dataset/preprocessing
    return 

#VISUAL 1: plotting individual feature maps
def plot_feature_maps(model, loader):
    model.eval()
    images, _ = next(iter(loader))
    with torch.no_grad():
        feature_maps = model(images.to(get_device()), return_feature_maps=True)

    for fmap in feature_maps:
        if fmap.dim() == 4:  # plot feature maps with 4 dimensions (CNN format)
            fmap = fmap.cpu().numpy()
            for i in range(min(5, fmap.shape[1])):  # Plots first 5 feature maps
                plt.imshow(fmap[0][i], cmap='gray')
                plt.show()


#VISUAL 2: Filter Visualization
def plot_filters(model):
    model.eval()
    filters = model.resnet[0].weight.data.clone()
    plt.figure(figsize=(8, 8))
    for i in range(min(10, filters.shape[0])):  # Plotting first 10 filters
        plt.subplot(3, 4, i + 1)
        plt.imshow(filters[i].cpu().numpy()[0], cmap='gray')
        plt.axis('off')
    plt.show()

#VISUAL 3: Confusion Matrix
def plot_confusion_matrix(model, loader, n_classes):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(get_device()))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(n_classes), yticklabels=range(n_classes))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

#Visual 4: t-SNE feature representation
def plot_tsne(model, loader, num_samples=500):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for i, (images, label) in enumerate(loader):
            if i == num_samples:  # Limit the number of samples for t-SNE
                break
            outputs = model.resnet(images.to(get_device()))
            features.append(outputs.cpu())
            labels.append(label)

    features = torch.cat(features).view(len(features), -1).numpy()
    labels = torch.cat(labels).numpy()

    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(features)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=1)
    plt.colorbar()
    plt.show()



