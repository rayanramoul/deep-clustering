import os
import gc
import argparse, sys
from utils.dataset import CloudDataset
from DeepCluster.cluster import cluster
from DeepCluster.training import train_epoch
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from torch import nn
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from torch.optim import Adam
from utils.visualize import extract_features
from torchvision import transforms as T
from torch.utils.data import DataLoader
"""
parser=argparse.ArgumentParser()


parser.add_argument('--lr', help='Learning Rate, default = 0.0001', type=int, default=0.0001)
parser.add_argument('--epochs', help='Number of epochs, default = 100', type=int, default=300)
parser.add_argument('--load_trained', help='Load existing model', type=bool, default=False)
parser.add_argument('--test_percentage', help='Percentage of training dataset, default=0.3', type=int, default=0.3)
parser.add_argument('--path_image_folder', help='Path to the dataset (ImageFolder scheme).', type=str, default="../../DATASETS/CLASSIF_RESIZED")
#../../DATASETS/CLASSIF_BIGGEST_SQUARE/"
args=parser.parse_args()
"""


normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
transformer = T.Compose([
    T.Resize([256, 256]),
    # you can add other transformations in this list
    T.ToTensor(),
    normalize,
])

def main():
    # data
    root = './CLASSIF_BIGGEST_SQUARE'
    limit_images = 10000
    pca_dim = 2
    kmeans_clusters = 4
    batch_size = 8
    num_classes = 4
    num_epochs = 100
    raw_dataset = CloudDataset(root=root, transforms=transformer)
    
    #raw_dataset = ImageFolder(root=root, transform=transformer)
    n = len(raw_dataset)
    n_test = int(0.1 * n)  # take ~10% for test
    train_set, test_set = torch.utils.data.random_split(raw_dataset, [len(raw_dataset)-n_test, n_test], generator=torch.Generator().manual_seed(42))

    print("Len train set : "+str(len(train_set)))
    print("Len test set : "+str(len(test_set)))


    train_dataloader  = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1) 
    test_dataloader  = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1)
    print("Classes found ")
    print(str(raw_dataset.classes))
    # convnet
    
    print("Initialize model.")

    # load resnet and alter last layer
    model = resnet18()
    model.fc = nn.Linear(512, num_classes)
    model.cuda();

    print("PCA initialization.")
    pca = IncrementalPCA(n_components=pca_dim, batch_size=512, whiten=True)
    kmeans = MiniBatchKMeans(n_clusters=kmeans_clusters, batch_size=512, init_size=3*kmeans_clusters)
    optimizer = Adam(model.parameters())
    # clustering
    
    print("Dataset loading.")
    raw_dataset = CloudDataset(root=root, transforms=transformer, limit=limit_images)
    pseudo_labels, features = cluster(pca, kmeans, model, raw_dataset, batch_size, return_features=True)

    print("Training ...")
    for i in range(num_epochs):
        print("Epoch : ("+str(i)+"/"+str(num_epochs)+")")
        pseudo_labels = cluster(pca, kmeans, model, raw_dataset, batch_size) # generate labels
        labeled_dataset = CloudDataset(root=root, labels=pseudo_labels, transforms=transformer, limit=limit_images) # make new dataset with labels matched to images
        train_epoch(model, optimizer, labeled_dataset, batch_size) 
    
    PATH = "./model/model.ckpt"
    torch.save(model, PATH)

if __name__ == '__main__':
    main()
    