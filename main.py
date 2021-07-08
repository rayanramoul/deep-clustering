from collections import Counter
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam

from torchvision.models import resnet18
from torchvision import transforms as T
from torchvision.utils import make_grid
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors

def main():
    
    # data
    root = '../input/images'
    limit_images = 10000

    # clustering
    pca_dim = 50
    kmeans_clusters = 12

    # convnet
    batch_size = 64
    num_classes = 4
    num_epochs = 2

    dataset = FoodDataset(root=root, limit=limit_images)


if __name__ == '__main__':
    main()
    