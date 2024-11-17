import torch
from torch import nn
from config import device, n_future


# the basics of RNN models layer

class Simple_RNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, drop_rate = 0.005535303510995099):
        super().__init__()

        # define the dimensions
        self.in_dim = in_dim
        self.hid_dim = in_dim
        self.out_dim = out_dim
        self.layer = num_layers

        # define the rnn layer
        self.rnn = nn.RNN(self.in_dim, self.hid_dim, self.layer, nonlinearity='tanh', batch_first=True)

        self.drop_out = nn.Dropout(p = drop_rate)
        # define fully connected layer for output
        self.fc = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self,x):

        # initialize the hidden layer
        h0 = torch.zeros(self.layer, x.size(0),self.hid_dim , device  = device)

        #initialize the rnn
        out, _ = self.rnn(x,h0)

        out = self.drop_out(out)
        out = self.fc(out[:,-n_future:,:])
        return out