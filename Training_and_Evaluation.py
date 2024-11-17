import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from AttentionBasedModels.Decoder_only import Decoder
from AttentionBasedModels.Enc_and_Dec_Transformer import Transformer
from AttentionBasedModels.Encoder_only import Encoder
from CNNBasedModels.OneD_CNN import CNN_ForecastNet
from RNNBasedModels.GRU import GRU
from RNNBasedModels.LSTM import LSTM
from RNNBasedModels.RNN import Simple_RNN
from Preprocessing import CustomDataset, process_missing_and_duplicate_timestamps
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

# Device configuration
device = (
    "mps"
    if getattr(torch, "has_mps", False)
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# Data setup
input_sequence_length = 168
target_sequence_length = 48

path = "Datasets/DUQ_hourly.csv"
train_set, val_set, test_set = process_missing_and_duplicate_timestamps(filepath=path)

# Initialize datasets and dataloaders
train_dataset = CustomDataset(train_set, input_sequence_length, target_sequence_length, multivariate=False, target_feature=0)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=True)
test_dataset = CustomDataset(test_set, input_sequence_length, target_sequence_length, multivariate=False, target_feature=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)
val_dataset = CustomDataset(val_set, input_sequence_length, target_sequence_length, multivariate=False, target_feature=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)

# Utility functions
def mean_squared_error(act, pred):
    diff = pred - act
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()
    return mean_diff

def root_mean_squared_error(act, pred):
    diff = pred - act
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()
    rmse_val = np.sqrt(mean_diff)
    return rmse_val

def mean_absolute_error(act, pred):
    diff = pred - act
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff

def plot_predictions(k, m, title, points=96):
    plt.figure(figsize=(12, 6))
    plt.plot([x for x in range(points)], k[0][:points], label="Predicted")
    plt.plot([x for x in range(points)], m[0][:points], label="Actual")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.show()

def plot_losses(train_losses, val_losses, title):
    plt.figure(figsize=(12, 6))
    plt.plot([x for x in range(len(train_losses))], train_losses, label="train loss")
    plt.plot([x for x in range(len(val_losses))], val_losses, label="val loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.legend()
    plt.show()





# 1. RNN training step
print("\nStarting RNN Training...")

model = Simple_RNN(in_dim=1, hid_dim=1740, out_dim=1, num_layers=1,
                   drop_rate=0.0009001480178615212).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0012874179807017348)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 200
early_stop_count = 0
min_val_loss = float('inf')
train_loss = []
validation_loss = []
batch_loss = 0  # Initialize batch loss


for epoch in range(epochs):
    model.train()
    batch_losses = []  # Track losses for each batch
    for batch in train_loader:
        x_batch, _, y_batch = batch  # src, trg, trg_y
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        batch_loss = criterion(outputs, y_batch)  # Store the batch loss
        batch_loss.backward()
        optimizer.step()
        batch_losses.append(batch_loss.item())

    # Average loss for the epoch
    epoch_loss = np.mean(batch_losses)
    train_loss.append(epoch_loss)

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, _, y_batch = batch  # src, trg, trg_y
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            val_loss = criterion(outputs, y_batch)
            val_losses.append(val_loss.item())

    val_loss = np.mean(val_losses)
    validation_loss.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 20:
        print("Early stopping!")
        break

    print(f"Epoch {epoch + 1}/{epochs}, train loss: {epoch_loss:.6f}, validation Loss: {val_loss:.6f}")

# Convert losses to numpy arrays
train_losses_np = torch.tensor(train_loss).detach().cpu().numpy()
test_loss_np = torch.tensor(validation_loss).detach().cpu().numpy()

# Evaluation
model.eval()
predictions = []
actual = []
test_loss = []

with torch.no_grad():
    for batch in val_loader:
        x_batch, _, y_batch = batch  # src, trg, trg_y
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        test_loss.append(loss.detach().cpu().numpy())
        actual.append(y_batch.detach().cpu().numpy())
        predictions.append(outputs.detach().cpu().numpy())

predictions = np.vstack(predictions)
actual = np.vstack(actual)
test_loss = np.vstack(test_loss)

# Print metrics
print("\nRNN Model Evaluation Metrics:")
print(f"MSE : {mean_squared_error(actual, predictions):.6f}")
print(f"RMSE: {root_mean_squared_error(actual, predictions):.6f}")
print(f"MAE : {mean_absolute_error(actual, predictions):.6f}")

# Plot results
k = pd.DataFrame(predictions.squeeze(-1))
m = pd.DataFrame(actual.squeeze(-1))

plot_predictions(k, m, "RNN Predicted vs. Actual Value")
plot_losses(train_losses_np, test_loss_np, "RNN Training vs. Validation Loss")




# 2. LSTM training step
print("\nStarting LSTM Training...")

model = LSTM(input_size=1, hidden_size=100, num_stacked_layers=3,
             drop_rate=0.1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 200
early_stop_count = 0
min_val_loss = float('inf')
train_loss = []
validation_loss = []

for epoch in tqdm(range(epochs), desc="LSTM Training"):
    model.train()
    for batch in train_loader:
        x_batch, _, y_batch = batch  # src, trg, trg_y
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, _, y_batch = batch  # src, trg, trg_y
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_losses.append(loss.item())

    val_loss = np.mean(val_losses)
    scheduler.step(val_loss)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 20:
        print("Early stopping!")
        break

    train_loss.append(loss.item())
    validation_loss.append(val_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Train loss: {loss.item():.6f}, Validation Loss: {val_loss:.6f}")

# Convert losses to numpy arrays
train_losses_np = torch.tensor(train_loss).detach().cpu().numpy()
test_loss_np = torch.tensor(validation_loss).detach().cpu().numpy()

# Evaluation
model.eval()
predictions = []
actual = []
test_loss = []

with torch.no_grad():
    for batch in val_loader:
        x_batch, _, y_batch = batch  # src, trg, trg_y
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        test_loss.append(loss.detach().cpu().numpy())
        actual.append(y_batch.detach().cpu().numpy())
        predictions.append(outputs.detach().cpu().numpy())

predictions = np.vstack(predictions)
actual = np.vstack(actual)
test_loss = np.vstack(test_loss)

# Print metrics
print("\nLSTM Model Evaluation Metrics:")
print(f"MSE : {mean_squared_error(actual, predictions):.6f}")
print(f"RMSE: {root_mean_squared_error(actual, predictions):.6f}")
print(f"MAE : {mean_absolute_error(actual, predictions):.6f}")

# Plot results
k = pd.DataFrame(predictions.squeeze(-1))
m = pd.DataFrame(actual.squeeze(-1))

plot_predictions(k, m, "LSTM Predicted vs. Actual Value")
plot_losses(train_losses_np, test_loss_np, "LSTM Training vs. Validation Loss")





# 3. GRU training step
print("\nStarting GRU Training...")

model = GRU(in_dim=1, hid_dim=100, out_dim=1,
            num_layer=3, drop_rate=0.1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 200
early_stop_count = 0
min_test_loss = float('inf')
train_loss = []
test_loss1 = []

for epoch in tqdm(range(epochs), desc="GRU Training"):
    model.train()
    for batch in train_loader:
        x_batch, _, y_batch = batch  # src, trg, trg_y
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in val_loader:
            x_batch, _, y_batch = batch  # src, trg, trg_y
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            test_losses.append(loss.item())

    test_loss = np.mean(test_losses)
    scheduler.step(test_loss)

    if test_loss < min_test_loss:
        min_test_loss = test_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 20:
        print("Early stopping!")
        break

    train_loss.append(loss.item())
    test_loss1.append(test_loss)
    print(f"Epoch {epoch + 1}/{epochs}, train loss: {loss.item():.6f}, val loss: {test_loss:.6f}")

# Convert losses to numpy arrays
train_losses_np = torch.tensor(train_loss).detach().cpu().numpy()
test_loss_np = torch.tensor(test_loss1).detach().cpu().numpy()

# Evaluation
model.eval()
predictions = []
actual = []
test_loss = []

with torch.no_grad():
    for batch in test_loader:
        x_batch, _, y_batch = batch  # src, trg, trg_y
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        test_loss.append(loss.detach().cpu().numpy())
        actual.append(y_batch.detach().cpu().numpy())
        predictions.append(outputs.detach().cpu().numpy())

predictions = np.vstack(predictions)
actual = np.vstack(actual)
test_loss = np.vstack(test_loss)

# Print metrics
print("\nGRU Model Evaluation Metrics:")
print(f"MSE : {mean_squared_error(actual, predictions):.6f}")
print(f"RMSE: {root_mean_squared_error(actual, predictions):.6f}")
print(f"MAE : {mean_absolute_error(actual, predictions):.6f}")

# Plot results
k = pd.DataFrame(predictions.squeeze(-1))
m = pd.DataFrame(actual.squeeze(-1))

plot_predictions(k, m, "GRU Predicted vs. Actual Value", points=96)
plot_losses(train_losses_np, test_loss_np, "GRU Training vs. Validation Loss")




# 4. 1D-CNN training step
print("\nStarting 1D-CNN Training...")

model = CNN_ForecastNet(hidden_size=100, kernel_size=5,
                        padding=2, drop_rate=0.1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 200
early_stop_count = 0
min_val_loss = float('inf')
train_loss = []
validation_loss = []

for epoch in tqdm(range(epochs), desc="1D-CNN Training"):
    model.train()
    for batch in train_loader:
        x_batch, _, y_batch = batch  # src, trg, trg_y
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, _, y_batch = batch  # src, trg, trg_y
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_losses.append(loss.item())

    val_loss = np.mean(val_losses)
    scheduler.step(val_loss)

    if val_loss < min_val_loss:
        min_val_loss = val_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 20:
        print("Early stopping!")
        break

    train_loss.append(loss.item())
    validation_loss.append(val_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Train loss: {loss.item():.6f}, Validation Loss: {val_loss:.6f}")

# Convert losses to numpy arrays
train_losses_np = torch.tensor(train_loss).detach().cpu().numpy()
test_loss_np = torch.tensor(validation_loss).detach().cpu().numpy()

# Evaluation
model.eval()
predictions = []
actual = []
test_loss = []

with torch.no_grad():
    for batch in val_loader:
        x_batch, _, y_batch = batch  # src, trg, trg_y
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        test_loss.append(loss.detach().cpu().numpy())
        actual.append(y_batch.detach().cpu().numpy())
        predictions.append(outputs.detach().cpu().numpy())

predictions = np.vstack(predictions)
actual = np.vstack(actual)
test_loss = np.vstack(test_loss)

# Print metrics
print("\n1D-CNN Model Evaluation Metrics:")
print(f"MSE : {mean_squared_error(actual, predictions):.6f}")
print(f"RMSE: {root_mean_squared_error(actual, predictions):.6f}")
print(f"MAE : {mean_absolute_error(actual, predictions):.6f}")

# Plot results
k = pd.DataFrame(predictions.squeeze(-1))
m = pd.DataFrame(actual.squeeze(-1))

plot_predictions(k, m, "1D-CNN Predicted vs. Actual Value")
plot_losses(train_losses_np, test_loss_np, "1D-CNN Training vs. Validation Loss")




# 5. Encoder-only training step
print("\nStarting Encoder-only Transformer Training...")

model = Encoder(num_layers=1, D=32, H=1, hidden_mlp_dim=100,
                inp_features=1, out_features=1, dropout_rate=0.1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 50
early_stop_count = 0
min_test_loss = float('inf')
train_losses = []
test_losses1 = []

for epoch in tqdm(range(epochs), desc="Encoder-only Training"):
    model.train()
    for batch in train_loader:
        x_batch, _, y_batch = batch  # src, trg, trg_y
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        out, _ = model(x_batch)  # Encoder returns outputs and attention weights
        train_loss = criterion(y_batch, out)
        train_loss.backward()
        optimizer.step()

    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, _, y_batch = batch  # src, trg, trg_y
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            out, _ = model(x_batch)
            test_loss = criterion(y_batch, out)
            test_losses.append(test_loss.item())

    test_loss = np.mean(test_losses)
    scheduler.step(test_loss)

    if test_loss < min_test_loss:
        min_test_loss = test_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 5:
        print("Early stopping!")
        break

    train_losses.append(train_loss.item())
    test_losses1.append(test_loss)
    print(f"Epoch {epoch + 1}/{epochs}, train loss: {train_loss.item():.6f}, test Loss: {test_loss:.6f}")

# Convert losses to numpy arrays
train_losses_np = torch.tensor(train_losses).detach().cpu().numpy()
test_loss_np = torch.tensor(test_losses1).detach().cpu().numpy()

# Evaluation
model.eval()
val_losses = []
val_preds = []
y_val_batch = []

with torch.no_grad():
    for (x, _, y) in val_loader:  # src, trg, trg_y
        x = x.to(device)
        y = y.to(device)
        y_pred, _ = model(x)
        loss_test = criterion(y_pred, y)

        val_losses.append(loss_test.item())
        y_val_batch.append(y.detach().cpu().numpy())
        val_preds.append(y_pred.detach().cpu().numpy())

val_predicted = np.vstack(val_preds)
actual_val_batch = np.vstack(y_val_batch)

# Print metrics
print("\nEncoder-only Transformer Evaluation Metrics:")
print(f"MSE : {mean_squared_error(actual_val_batch, val_predicted):.6f}")
print(f"RMSE: {root_mean_squared_error(actual_val_batch, val_predicted):.6f}")
print(f"MAE : {mean_absolute_error(actual_val_batch, val_predicted):.6f}")

# Plot results
k = pd.DataFrame(val_predicted.squeeze(-1))
m = pd.DataFrame(actual_val_batch.squeeze(-1))

plot_predictions(k, m, "Encoder-only Transformer Predicted vs. Actual Value")

# Adjust training loss scale for visualization
tp = train_losses_np * 10
plot_losses(tp, test_loss_np, "Encoder-only Transformer Training vs. Validation Loss")



# 6. Decoder-only training step
print("\nStarting Decoder-only Transformer Training...")

def create_look_ahead_mask(size, device=device):
    mask = torch.ones((size, size), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask  # (size, size)

model = Decoder(num_layers=1, D=32, H=4, hidden_mlp_dim=32,
                inp_features=1, out_features=1, dropout_rate=0.1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 200
early_stop_count = 0
min_test_loss = float('inf')
train_losses = []
test_losses1 = []

for epoch in tqdm(range(epochs), desc="Decoder-only Training"):
    model.train()
    for batch in train_loader:
        x_batch, _, y_batch = batch  # src, trg, trg_y
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Create mask for decoder self-attention
        S = x_batch.shape[1]
        mask = create_look_ahead_mask(S)

        optimizer.zero_grad()
        out, _ = model(x_batch, mask)
        train_loss = criterion(y_batch, out)
        train_loss.backward()
        optimizer.step()

    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, _, y_batch = batch  # src, trg, trg_y
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            S = x_batch.shape[1]
            mask = create_look_ahead_mask(S)

            out, _ = model(x_batch, mask)
            test_loss = criterion(y_batch, out)
            test_losses.append(test_loss.item())

    test_loss = np.mean(test_losses)
    scheduler.step(test_loss)

    if test_loss < min_test_loss:
        min_test_loss = test_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 20:
        print("Early stopping!")
        break

    train_losses.append(train_loss.item())
    test_losses1.append(test_loss)
    print(f"Epoch {epoch + 1}/{epochs}, train loss: {train_loss.item():.6f}, test Loss: {test_loss:.6f}")

# Convert losses to numpy arrays
train_losses_np = torch.tensor(train_losses).detach().cpu().numpy()
test_loss_np = torch.tensor(test_losses1).detach().cpu().numpy()

# Evaluation
model.eval()
val_losses = []
val_preds = []
y_val_batch = []

with torch.no_grad():
    for (x, _, y) in val_loader:  # src, trg, trg_y
        x = x.to(device)
        y = y.to(device)

        S = x.shape[-2]
        mask = create_look_ahead_mask(S)

        y_pred, _ = model(x, mask=mask)
        loss_test = criterion(y_pred, y)

        val_losses.append(loss_test.item())
        y_val_batch.append(y.detach().cpu().numpy())
        val_preds.append(y_pred.detach().cpu().numpy())

val_predicted = np.vstack(val_preds)
actual_val_batch = np.vstack(y_val_batch)

# Print metrics
print("\nDecoder-only Transformer Evaluation Metrics:")
print(f"MSE : {mean_squared_error(actual_val_batch, val_predicted):.6f}")
print(f"RMSE: {root_mean_squared_error(actual_val_batch, val_predicted):.6f}")
print(f"MAE : {mean_absolute_error(actual_val_batch, val_predicted):.6f}")

# Plot results
k = pd.DataFrame(val_predicted.squeeze(-1))
m = pd.DataFrame(actual_val_batch.squeeze(-1))

plot_predictions(k, m, "Decoder-only Transformer Predicted vs. Actual Value", points=48)
plot_losses(train_losses_np, test_loss_np, "Decoder-only Transformer Training vs. Validation Loss")




# 7. Enc&Dec Transformer training step
print("\nStarting Full Transformer Training...")

def generate_square_subsequent_mask(dim1: int, dim2: int, device=device) -> Tensor:
    """Generate mask for transformer attention"""
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1).to(device)

model = Transformer(
    input_size=1,
    dec_seq_len=48,
    num_predicted_features=1
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 200
early_stop_count = 0
min_test_loss = float('inf')
train_losses = []
test_losses1 = []

for epoch in tqdm(range(epochs), desc="Full Transformer Training"):
    model.train()
    for batch in train_loader:
        src, trg, trg_y = batch
        src = src.to(device)
        trg = trg.to(device)
        trg_y = trg_y.to(device)

        # Generate masks
        tgt_mask = generate_square_subsequent_mask(
            dim1=target_sequence_length,
            dim2=target_sequence_length
        )
        src_mask = generate_square_subsequent_mask(
            dim1=target_sequence_length,
            dim2=input_sequence_length
        )

        optimizer.zero_grad()
        out = model(src, trg, src_mask, tgt_mask)
        train_loss = criterion(trg_y, out)
        train_loss.backward()
        optimizer.step()

    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            src, trg, trg_y = batch
            src = src.to(device)
            trg = trg.to(device)
            trg_y = trg_y.to(device)

            tgt_mask = generate_square_subsequent_mask(
                dim1=target_sequence_length,
                dim2=target_sequence_length
            )
            src_mask = generate_square_subsequent_mask(
                dim1=target_sequence_length,
                dim2=input_sequence_length
            )

            out = model(src, trg, src_mask, tgt_mask)
            test_loss = criterion(trg_y, out)
            test_losses.append(test_loss.item())

    test_loss = np.mean(test_losses)
    scheduler.step(test_loss)

    if test_loss < min_test_loss:
        min_test_loss = test_loss
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 20:
        print("Early stopping!")
        break

    train_losses.append(train_loss.item())
    test_losses1.append(test_loss)
    print(f"Epoch {epoch + 1}/{epochs}, train loss: {train_loss.item():.6f}, val Loss: {test_loss:.6f}")

# Convert losses to numpy arrays
train_losses_np = torch.tensor(train_losses).detach().cpu().numpy()
test_loss_np = torch.tensor(test_losses1).detach().cpu().numpy()

# Evaluation
model.eval()
val_losses = []
val_preds = []
y_val_batch = []

with torch.no_grad():
    for batch in val_loader:
        src, trg, trg_y = batch
        src = src.to(device)
        trg = trg.to(device)
        trg_y = trg_y.to(device)

        tgt_mask = generate_square_subsequent_mask(
            dim1=target_sequence_length,
            dim2=target_sequence_length
        )
        src_mask = generate_square_subsequent_mask(
            dim1=target_sequence_length,
            dim2=input_sequence_length
        )

        out = model(src, trg, src_mask, tgt_mask)
        val_loss = criterion(trg_y, out)

        val_losses.append(val_loss.item())
        val_preds.append(out.detach().cpu().numpy())
        y_val_batch.append(trg_y.detach().cpu().numpy())

val_predicted = np.vstack(val_preds)
actual_val_batch = np.vstack(y_val_batch)

# Print metrics
print("\nFull Transformer Evaluation Metrics:")
print(f"MSE : {mean_squared_error(actual_val_batch, val_predicted):.6f}")
print(f"RMSE: {root_mean_squared_error(actual_val_batch, val_predicted):.6f}")
print(f"MAE : {mean_absolute_error(actual_val_batch, val_predicted):.6f}")

# Plot results
k = pd.DataFrame(val_predicted.squeeze(-1))
m = pd.DataFrame(actual_val_batch.squeeze(-1))

plot_predictions(k, m, "Full Transformer Predicted vs. Actual Values", points=96)
plot_losses(train_losses_np, test_loss_np, "Full Transformer Training vs. Validation Loss")

print("\nTraining complete for all models!")







# Model Comparison and Analysis
def compare_all_models_metrics(results_dict):
    """
    Compare metrics across all models
    """
    print("\n=== Model Comparison ===")
    metrics_df = pd.DataFrame(results_dict).round(6)
    print("\nMetrics Comparison:")
    print(metrics_df)

    # Plot comparison bar charts
    metrics = ['MSE', 'RMSE', 'MAE']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        plt.bar(metrics_df.index, metrics_df[metric])
        plt.title(f'{metric} Comparison Across Models')
        plt.xticks(rotation=45)
        plt.ylabel(metric)
        plt.tight_layout()
        plt.show()

# Collect results from all models
results = {
    'RNN': {
        'MSE': mean_squared_error(actual, predictions),
        'RMSE': root_mean_squared_error(actual, predictions),
        'MAE': mean_absolute_error(actual, predictions)
    },
    'LSTM': {
        'MSE': mean_squared_error(actual, predictions),
        'RMSE': root_mean_squared_error(actual, predictions),
        'MAE': mean_absolute_error(actual, predictions)
    },
    'GRU': {
        'MSE': mean_squared_error(actual, predictions),
        'RMSE': root_mean_squared_error(actual, predictions),
        'MAE': mean_absolute_error(actual, predictions)
    },
    '1D-CNN': {
        'MSE': mean_squared_error(actual, predictions),
        'RMSE': root_mean_squared_error(actual, predictions),
        'MAE': mean_absolute_error(actual, predictions)
    },
    'Encoder-only': {
        'MSE': mean_squared_error(actual_val_batch, val_predicted),
        'RMSE': root_mean_squared_error(actual_val_batch, val_predicted),
        'MAE': mean_absolute_error(actual_val_batch, val_predicted)
    },
    'Decoder-only': {
        'MSE': mean_squared_error(actual_val_batch, val_predicted),
        'RMSE': root_mean_squared_error(actual_val_batch, val_predicted),
        'MAE': mean_absolute_error(actual_val_batch, val_predicted)
    },
    'Full-Transformer': {
        'MSE': mean_squared_error(actual_val_batch, val_predicted),
        'RMSE': root_mean_squared_error(actual_val_batch, val_predicted),
        'MAE': mean_absolute_error(actual_val_batch, val_predicted)
    }
}

# Compare all models
compare_all_models_metrics(results)

# Save results to CSV
results_df = pd.DataFrame(results).round(6)
results_df.to_csv('model_comparison_results.csv')

# Final plots comparing predictions across all models
plt.figure(figsize=(15, 8))
plt.plot(m[0][:96], label='Actual', color='black', linewidth=2)
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
models = ['RNN', 'LSTM', 'GRU', '1D-CNN', 'Encoder-only', 'Decoder-only', 'Full-Transformer']

for model, color in zip(models, colors):
    if model in ['Encoder-only', 'Decoder-only', 'Full-Transformer']:
        plt.plot(k[0][:96], label=model, color=color, alpha=0.7)
    else:
        plt.plot(k[0][:96], label=model, color=color, alpha=0.7)

plt.title('Prediction Comparison Across All Models')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Print execution information
print("\nExecution Complete!")
print(f"Results saved to: model_comparison_results.csv")
print(f"Device used: {device}")
print(f"Total number of epochs per model:")
print(f"RNN, LSTM, GRU, 1D-CNN, Decoder-only, Full-Transformer: 200")
print(f"Encoder-only: 50")