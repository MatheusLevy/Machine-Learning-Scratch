from torch.utils.data import Dataset
import torch
from skimage import io
import os
import numpy as np
from PIL import Image

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