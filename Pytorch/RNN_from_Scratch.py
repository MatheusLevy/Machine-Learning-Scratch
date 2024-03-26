import torch
import torch.nn as nn 

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # Input to Hidden (recurrent loop)
        self.i2o = nn.Linear(input_size + hidden_size, output_size) # Output of RNN Cell
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):

        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined) # Hidden State
        output = self.i2o(combined) # Output State
        output = self.softmax(output) # Sofmax
        return output, hidden # Output of first loop of rnn cell
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
n_classes = 10

input_size = 20 
rnn = RNN(input_size, n_hidden, 10)

example = torch.randn(input_size, 1, 20) 

# Process the 20 sequences and pick the output of the last 

hidden =  rnn.init_hidden()
for i in range(example.size()[0]):
    output, hidden = rnn(example[i], hidden)

print(output.shape)
print(hidden.shape)