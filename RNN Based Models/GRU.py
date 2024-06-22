
import torch
from torch.nn import nn

# the basics of LSTM models layer

class GRU(nn.Module):
    def __init__(self,in_dim, hid_dim, out_dim, num_layer, drop_rate = 0.1):
        super().__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.layer = num_layer
        self.drop_rat = nn.Dropout(p = drop_rate)

        self.gru = nn.GRU(self.in_dim, self.hid_dim, self.layer, batch_first=True)
        self.fc1 = nn.Linear(self.hid_dim , 128)
        self.fc2 = nn.Linear(128, self.out_dim)

    def forward(self,x):

        # initalize the hidden layers
        h0 = torch.zeros(self.layer, x.size(0), self.hid_dim).to(device)

        out, _ = self.gru(x,h0)
        out = self.drop_rat(out)
        out = self.fc1(out)
        out = self.fc2(out[:,-48:,:])
        return out