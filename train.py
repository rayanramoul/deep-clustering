
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
import pickle
"""

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
    #normalize,
])

def main(args):
    # data
    root = args.train_dataset
    limit_images = 999999
    pca_dim = 2
    kmeans_clusters = 12
    batch_size = 16
    num_classes = 12
    num_epochs = args.epochs
    raw_dataset = CloudDataset(root=root, transforms=transformer)
    
    print("Classes found ")
    print(str(raw_dataset.classes))
    # convnet
    
    print("Initialize model.")

    # load resnet and alter last layer
    model = resnet18()
    for param in model.parameters():
	    param.requires_grad = False
    
    model.fc = nn.Sequential(
    nn.Linear(512, 200),
    nn.ReLU(),
    nn.Dropout(0.5),
    
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Dropout(0.5),
    
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Dropout(0.5),
    
    nn.Linear(50, 10),
    nn.ReLU(),
    nn.Dropout(0.5),
    
    nn.Linear(10, num_classes),
    )
    model.cuda();


    print("PCA initialization.")
    pca = IncrementalPCA(n_components=pca_dim, batch_size=512, whiten=True)
    kmeans = MiniBatchKMeans(n_clusters=kmeans_clusters, batch_size=512, init_size=3*kmeans_clusters)
    optimizer = Adam(model.parameters())
    # clustering
    
    print("Dataset loading.")
    raw_dataset = CloudDataset(root=root, transforms=transformer, limit=limit_images)
    pseudo_labels, features = cluster(pca, kmeans, model, raw_dataset, batch_size, return_features=True, learn="fit_predict")


    print("Training ...")
    
    PATH = args.path_save
    global_min_loss = 9999
    for i in range(num_epochs):
        print("Epoch : ("+str(i)+"/"+str(num_epochs)+")")
        pseudo_labels = cluster(pca, kmeans, model, raw_dataset, batch_size, learn="fit_predict") # generate labels
        labeled_dataset = CloudDataset(root=root, labels=pseudo_labels, transforms=transformer, limit=limit_images) # make new dataset with labels matched to images
        loss = train_epoch(model, optimizer, labeled_dataset, batch_size) 
        loss = loss.item()
        print("global min loss = "+str(global_min_loss))
        print("loss = "+str(loss))
        if loss<global_min_loss:
            print("Saving...")
            global_min_loss = loss
            torch.save(model, PATH)
            with open(args.kmean_save, "wb") as f:
                pickle.dump(kmeans, f)
            with open(args.pca_save, "wb") as f:
                pickle.dump(pca, f)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    BATCH_SIZE = 64
    parser.add_argument('--lr', help='Learning Rate, default = 0.0001', type=float, default=0.001)
    parser.add_argument('--epochs', help='Number of epochs, default = 100', type=int, default=300)
    parser.add_argument('--load_trained', help='Load existing model', type=bool, default=False)
    parser.add_argument('--train_dataset', help='Path to the train dataset (ImageFolder scheme).', type=str)
    parser.add_argument('--val_dataset', help='Path to the validation dataset (ImageFolder scheme).', type=str)
    parser.add_argument('--path_save', help='Where to save model.', type=str, default="autoencoder.model")
    parser.add_argument('--kmean_save', help='Where to save kmeans model.', type=str, default="kmeans.model")
    parser.add_argument('--pca_save', help='Where to save pca.', type=str, default="pca.model")
    #../../DATASETS/CLASSIF_BIGGEST_SQUARE/"
    args=parser.parse_args()
    main(args)