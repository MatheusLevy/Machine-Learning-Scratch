# Imports
import torch
import torch.nn as nn
import torch.optim  as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Set device
device = 'cpu'

# Hyperparameters
input_size = 28
sequence_lenght = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs =  2

# Create RNN

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # N x time_sequence x time_features
        self.fc = nn.Linear(hidden_size*sequence_lenght, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # foward 
        out,_ = self.rnn(x, h0)
        print(out.shape)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


# Load Data
train_dataset = datasets.MNIST(root= 'dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) 

test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize Network
model = RNN(input_size, hidden_size, num_layers, num_classes ).to(device)

# Loss and Optimization
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
            x = x.to(device).squeeze(1)
            y = y.to(device)

            scores = model(x)
            # 64x10
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    acc = float(num_correct/float(num_samples))
    print(f'{num_correct}/{num_samples} with accuracy {float(num_correct/float(num_samples))*100:.2f}')
    model.train()
    return acc

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device).squeeze(1)
        targets = targets.to(device)

        # forward
        scores = model(data) 
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descente or adam step
        optimizer.step()
    print(f'Epoch {epoch}')
    check_accuracy(train_loader, model)

# Check accuracy on training & test 


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)