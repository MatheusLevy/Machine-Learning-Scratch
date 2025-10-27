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
    transforms.Normalize((0.4898, 0.4898, 0.4898), (0.2471, 0.2471, 0.2471))
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
learning_rate = 0.0001
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
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=5) 

val_dataset = XrayDataset(csv_file=val_csv,
                        root_dir='/home/matheus_levy/workspace/lucas/dataset/images',
                        transform= val_transformer)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=3) 

test_dataset = XrayDataset(csv_file=test_csv,
                        root_dir='/home/matheus_levy/workspace/lucas/dataset/images',
                        transform= val_transformer)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) 

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class fusion_model(nn.Module):
    def __init__(self, model1, model_1_out_channels, model2, model2_out_channels, num_classes):
        super(fusion_model, self).__init__()
        self.branch1 = model1
        self.branch2 = model2
        self.attention = SEBlock(2048)
        self.conv1 = nn.Conv2d(model_1_out_channels, 1024, kernel_size=(1,1), stride=1, padding=0)
        self.conv2 = nn.Conv2d(model2_out_channels, 1024, kernel_size=(1,1), stride=1, padding=0)
        self.fc = nn.Linear(2048, num_classes)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        concat = torch.cat((x1, x2), dim=1)
        x = self.attention(concat)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

model1 = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
model1 = nn.Sequential(*list(model1.children())[:-1])

model2 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
model2 = nn.Sequential(*list(model2.children())[:-2])


f_model = fusion_model(model1= model1, model_1_out_channels=1024, model2=model2, model2_out_channels=2048, num_classes=14)
f_model.to(device)

criterion = BinaryFocalLoss(gamma=4, alpha=0.8)
optimizer = optim.Adam(f_model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.1)

# Early Stopping
patience = 5
minDelta = 0.01
currentPatience = 0
bestLoss = float('inf')


for epoch in range(epochs):
    f_model.train()
    runningLoss = 0.0
    valid_loss = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, )
    for batch_idx, (data, targets) in loop:
        # Send to device

        data = data.to(device)
        targets = targets.to(device)

        # foward
        scores = f_model(data)
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

    # Validation phase
    f_model.eval()
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)
            outputs = f_model(data)
            loss = criterion(outputs, targets)
            valid_loss += loss.item()

    valid_loss /= len(val_loader)
    scheduler.step(valid_loss)
    print(f"Val Loss: {valid_loss}")
    check_auc(val_loader, f_model,device=device, labels=labels, save= False, Train=False)

    checkpoint = {'state_dict': f_model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint, filename= f'my_checkpoint_epoch_fusion_pytorch_{epoch}.pth.tar')

    if runningLoss < bestLoss - minDelta:
        bestLoss = runningLoss
        currentPatience = 0

    else:
        currentPatience += 1
    if currentPatience >= patience:
        print('Early stopping triggered.')
        break

print('Check AUC for Test Set')
check_auc(test_loader, f_model, device=device, labels=labels, save= True, Train=False)