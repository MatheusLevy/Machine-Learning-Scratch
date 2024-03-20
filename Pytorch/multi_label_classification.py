#Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from skimage import io
import torchvision
from tqdm import tqdm
import numpy as np
from PIL import Image
from utils import *

# Multi-Label Classificion in NHICC dataset

class XrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 2])
        y_label = torch.tensor(self.annotations.iloc[index, range(8, 22)].values.astype(np.float32))
        image = io.imread(img_path)
        if len(image.shape) == 3: # safe check for a image that outputs 1024, 1024, 4
           image = image[..., 0] 
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return (image, y_label)

train_transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(p = 0.3),
    transforms.Resize(size = (256, 256), antialias = True),
    transforms.ToTensor(),
    transforms.Normalize((0.4898, 0.4898, 0.4898), (0.2471, 0.2471, 0.2471))
])

# Validation transformer
val_transformer = transforms.Compose([
    transforms.Resize(size = (256, 256), antialias = True),
    transforms.ToTensor(),
    transforms.Normalize((0.4929, 0.4929, 0.4929), (0.2479, 0.2479, 0.2479))
])


device = 'cuda'
batch_size = 32
learning_rate = 0.0001
epochs = 20
n_classes = 14
labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia' , 'Infiltration' , 'Mass' ,'Nodule', 'Pleural_Thickening' , 'Pneumonia,Pneumothorax']

train_csv = pd.read_csv('/home/matheus_levy/workspace/lucas/dataset/df_train.csv')
train_csv = train_csv.drop(columns=['No Finding'])

val_csv = pd.read_csv('/home/matheus_levy/workspace/lucas/dataset/df_val.csv')
val_csv = val_csv.drop(columns=['No Finding'])

test_csv = pd.read_csv('/home/matheus_levy/workspace/lucas/dataset/df_test.csv')
test_csv = test_csv.drop(columns=['No Finding'])

train_dataset = XrayDataset(csv_file=train_csv,
                      root_dir='/home/matheus_levy/workspace/lucas/dataset/images',
                      transform= train_transformer)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 

val_dataset = XrayDataset(csv_file=val_csv,
                        root_dir='/home/matheus_levy/workspace/lucas/dataset/images',
                        transform= val_transformer)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False) 

test_dataset = XrayDataset(csv_file=test_csv,
                        root_dir='/home/matheus_levy/workspace/lucas/dataset/images',
                        transform= val_transformer)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) 

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.avgpool = Identity()
model.classifier = nn.Linear(1280*8*8, n_classes)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.1)

def check_auc(loader, model, save= False, Train= True):
    if Train:
        print('Checking auc on training data')
    else:
        print('Checking auc on val/test data')
    model.eval()
    trues, preds_all = [], []
    loop = tqdm(enumerate(loader), total=len(loader), leave=False, )
    with torch.no_grad:
        for i, (data, targets) in loop:
            data = data.to(device)
            targets.to(device)
            scores = model(data)
            preds = torch.sigmoid(scores)
            trues.extend(targets.cpu().tolist())
            preds_all.extend(preds.cpu().tolist())
    results = evaluate_classification_model(trues, preds_all, labels)
    auc = results['auc_scores']
    auc_macro = results['auc_macro']
    auc_micro = results['auc_micro']
    auc_weighted = results['auc_weighted']
    print(f'AUC: {auc} Auc-Macro: {auc_macro} Auc-Micro: {auc_micro} Auc-Weighted: {auc_weighted}')
    if save:
        store_test_metrics(results, path='/home/matheus_levy/workspace/pytorch_study/metrics',
                            filename=f"metrics_global", name='model', json=True, result_path='/home/matheus_levy/workspace/pytorch_study/metrics/result.json')

# Early Stopping
patience = 5
minDelta = 0.01
currentPatience = 0
bestLoss = float('inf')


for epoch in range(epochs):
    model.train()
    runningLoss = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, )
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    for batch_idx, (data, targets) in loop:
        # Send to device

        data = data.to(device)
        targets = targets.to(device)

        # foward
        scores = model(data)
        loss = criterion(scores, targets)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()

        # log
        runningLoss += loss.item()

        # otmizar
        optimizer.step()
        
        # update progress bar
        loop.set_description(f'Epoch [{epoch}/{epochs}]')
        loop.set_postfix(loss = loss.item())
    check_auc(val_loader, model, save= False, Train=False)
    # checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    # save_checkpoint(checkpoint, filename= f'my_checkpoint_epoch_{epoch}.pth.tar')

    if runningLoss < bestLoss - minDelta:
        bestLoss = runningLoss
        currentPatience = 0
    else:
        currentPatience += 1
    if currentPatience >= patience:
        print('Early stopping triggered.')
        break

print('Check AUC for Test Set')
check_auc(test_loader, model, save= True, Train=False)