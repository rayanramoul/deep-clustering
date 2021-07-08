from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

def train_epoch(model, optimizer, train_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    total_loss = 0
    pbar = tqdm(train_loader)
    for batch, (images, labels) in enumerate(pbar):
        optimizer.zero_grad()
        images = Variable(images).cuda()
        labels = Variable(labels).cuda().long()
        out = model(images)
        loss = F.cross_entropy(out, labels)
        total_loss += loss.data
        pbar.set_description(f'training - loss: {total_loss / (batch + 1)}')
        loss.backward()
        optimizer.step()