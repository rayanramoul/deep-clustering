from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
from PIL import Image

class CloudDataset(Dataset):
    def __init__(self, root, transforms=None, labels=[], limit=None):
        self.root = Path(root)
        self.image_paths = list(Path(root).rglob('*.jpeg'))
        print("Number of cloud images : "+str(len(self.image_paths)))
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.labels = labels
        self.transforms = transforms
        self.classes = set([path.parts[-2] for path in self.image_paths])
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index] if self.labels else 0
        image = Image.open(image_path)
        if self.transforms:
            return self.transforms(image), label
        return image, label
            
    def __len__(self):
        return len(self.image_paths)    
