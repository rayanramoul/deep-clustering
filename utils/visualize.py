from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch.autograd import Variable

def show_cluster(cluster, labels, dataset, limit=32):
    images = []
    labels = np.array(labels)
    indices = np.where(labels==cluster)[0]
    
    if not indices.size:
        print(f'cluster: {cluster} is empty.')
        return None
    
    for i in indices[:limit]:
        image, _ = dataset[i]
        images.append(image)
        
    gridded = make_grid(images)
    plt.figure(figsize=(15, 10))
    plt.title(f'cluster: {cluster}')
    plt.imshow(gridded.permute(1, 2, 0))
    plt.axis('off')
    
    
def show_neighbors(neighbors, dataset):
    images = []
    for n in neighbors:
        images.append(dataset[n][0])

    gridded = make_grid(images)
    plt.figure(figsize=(15, 10))
    plt.title(f'image and nearest neighbors')
    plt.imshow(gridded.permute(1, 2, 0))
    
def show_grid(count, labels, dataset, limit=32):
    limit = 10
    images = []
    for j in range(12):
        cluster = count[rx][0]
        labels = np.array(labels)
        indices = np.where(labels==cluster)[0]
        
        if not indices.size:
            print(f'cluster: {cluster} is empty.')
            return None
        
        for i in indices[:limit]:
            image, _ = dataset[i]
            images.append(image)
        
    gridded = make_grid(images)
    plt.figure(figsize=(15, 10))
    plt.title(f'cluster: {cluster}')
    plt.imshow(gridded.permute(1, 2, 0))
    plt.axis('off')


def extract_features(model, dataset, batch_size=32):
    """
    Gets the output of a pytorch model given a dataset.
    """
    loader = DataLoader(dataset, batch_size=batch_size)
    features = []
    for image, _ in tqdm(loader, desc='extracting features'):
        output = model(Variable(image).cuda())
        features.append(output.data.cpu())
    return torch.cat(features).numpy() 