# Imports
import torch
import torch.nn as nn
import torch.optim  as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm

batch_size = 32
learning_rate = 0.001
epochs = 2

device = 'cpu'

train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset,  batch_size= batch_size, shuffle=True )
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

# Load Pre-trained model
model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad= False

model.avgpool = Identity()
model.classifier = nn.Linear(512, 10)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, factor=0.1)

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Flatten if using only NN
            # x = x.reshape(x.shape[0], -1)

            scores = model(x)
            # 64x10
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    acc = float(num_correct/float(num_samples))
    print(f'{num_correct}/{num_samples} with accuracy {float(num_correct/float(num_samples))*100:.2f}')
    model.train()
    return acc

def save_checkpoint(state, filename= 'my_checkpoint.pth.tar'):
    print('=> saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print('=> Loading Checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

for epoch in range(epochs):
    model.train()
    runningLoss = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
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
        
    save_checkpoint(checkpoint, filename= f'checkpoint_{epoch}.pth.tar')
    trainLoss = runningLoss/ len(train_loader)
    scheduler.step(trainLoss)
    check_accuracy(train_loader, model)

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)

load_checkpoint(torch.load('checkpoint_0.pth.tar'))