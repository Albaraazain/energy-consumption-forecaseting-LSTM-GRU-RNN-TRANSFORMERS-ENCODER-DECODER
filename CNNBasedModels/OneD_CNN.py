import torch
from torch import nn
import os
import time
from pathlib import Path

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

device = get_device()

def setup_checkpoint_dir():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = Path(f"checkpoints/cnn/{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"CNN checkpoint directory created: {checkpoint_dir}")
    return checkpoint_dir

def save_model(model, checkpoint_dir, name, epoch=None):
    if epoch is not None:
        path = checkpoint_dir / f"{name}_epoch_{epoch}.pt"
    else:
        path = checkpoint_dir / f"{name}_final.pt"

    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'device': str(next(model.parameters()).device)
    }, path)
    print(f"CNN model saved to {path}")

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"CNN model loaded from {path}")
    return model, checkpoint['epoch']

class CNN_ForecastNet(nn.Module):
    def __init__(self, hidden_size=64, kernel_size=3, padding=1, drop_rate=0.1):
        super(CNN_ForecastNet, self).__init__()

        # Move all layers to GPU during initialization
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=padding
        ).to(device)

        self.relu = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout(drop_rate)
        self.max_pooling = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(hidden_size * 84, 100).to(device)
        self.fc2 = nn.Linear(100, 48 * 1).to(device)

        self.checkpoint_dir = setup_checkpoint_dir()

    def forward(self, x):
        # Ensure input is on GPU
        x = x.to(device)

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

    def save(self, epoch=None):
        save_model(self, self.checkpoint_dir, "cnn", epoch)

    def load(self, path):
        return load_model(self, path)

def move_to_gpu(model):
    if torch.cuda.is_available():
        model = model.to(device)
        print(f"CNN model moved to: {next(model.parameters()).device}")
    return model

# Example training function
"""
def train_cnn(model, train_loader, num_epochs=100, save_frequency=10):
    model = move_to_gpu(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
        
        # Save checkpoint
        if epoch % save_frequency == 0:
            model.save(epoch)
    
    # Save final model
    model.save()
    return model

# Usage example:
model = CNN_ForecastNet()
model = move_to_gpu(model)

# Train the model
model = train_cnn(model, train_loader)

# Load a saved model
new_model = CNN_ForecastNet()
new_model, epoch = new_model.load("checkpoints/cnn/20240118-123456/cnn_epoch_10.pt")
new_model = move_to_gpu(new_model)
"""