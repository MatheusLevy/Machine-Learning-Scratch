from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from skimage import io
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm

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
    transforms.Resize(size = (256, 256), antialias = True),
    transforms.ToTensor(),
])

# Validation transformer
val_transformer = transforms.Compose([
    transforms.Resize(size = (256, 256), antialias = True),
    transforms.ToTensor(),
])

device = 'cuda'
batch_size = 1
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

psum = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

# loop through images
for inputs, _ in tqdm(val_loader):
    psum += inputs.sum(axis=[0, 2, 3])
    psum_sq += (inputs**2).sum(axis=[0, 2, 3])

# pixel count
image_size= 256

count = len(val_loader) * image_size * image_size

# mean and std
total_mean = psum / count
total_var = (psum_sq / count) - (total_mean**2)
total_std = torch.sqrt(total_var)

# output
print("mean: " + str(total_mean))
print("std:  " + str(total_std))