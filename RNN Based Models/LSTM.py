import torch
from torch.nn import nn


# the basics of LSTM models layer

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, drop_rate = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)
        self.drop_out = nn.Dropout(p=drop_rate)
        #self.fc1 = nn.Linear(hidden_size , 128)
        self.fc2 = nn.Linear(hidden_size, 1)


    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, device = device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size , device = device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.drop_out(out)
        #out = self.fc1(out)
        out = self.fc2(out[:, -n_future:, :])
        return out