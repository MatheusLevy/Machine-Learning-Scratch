#Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
import torchvision
from tqdm import tqdm
from utils import *
from BinaryFocalLoss import BinaryFocalLoss
from data_loader import XrayDataset

# Multi-Label Classificion in NHICC dataset

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


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
batch_size = 1
learning_rate = 0.001
epochs = 100
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

criterion = BinaryFocalLoss(gamma=4, alpha=0.8)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.1)

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

    check_auc(val_loader, model,device=device, labels=labels, save= False, Train=False)
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
check_auc(test_loader, model,device=device, labels=labels, save= True, Train=False)