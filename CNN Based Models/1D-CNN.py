import torch
from torch.nn import nn

# the basics of 1D-CNN models layer

class CNN_ForecastNet(nn.Module):
    def __init__(self, hidden_size=64, kernel_size=3, padding=1, drop_rate=0.1):
        super(CNN_ForecastNet, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout(drop_rate)
        self.max_pooling = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(hidden_size * 84, 100)
        self.fc2 = nn.Linear(100, 48 * 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.max_pooling(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.reshape(x.size(0), 48, 1)

        return x