import torch
from torch.nn import nn, Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from Preprocessing import CustomDataset, process_missing_and_duplicate_timestamps
from torch.utils.data import TensorDataset , DataLoader, Dataset
import pandas as pd
from matplotlib.pyplot import pyplot as plt
from tqdm import tqdm

device = (
    "mps"
    if getattr(torch, "has_mps", False)
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


input_sequence_length = 168
target_sequence_length = 48

path = "Dataset//Dataset_name.csv"
train_set , val_set , test_set = process_missing_and_duplicate_timestamps(filepath = path)

train_dataset = CustomDataset(train_set, input_sequence_length, target_sequence_length, multivariate=False, target_feature=0)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False , drop_last=True)
test_dataset = CustomDataset(test_set, input_sequence_length, target_sequence_length, multivariate=False, target_feature=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False , drop_last=True)
val_dataset = CustomDataset(val_set, input_sequence_length, target_sequence_length, multivariate=False, target_feature=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False , drop_last=True)

# 1. RNN training step

from RNN import Simple_RNN

model = Simple_RNN(in_dim = 1, hid_dim = 1740, out_dim = 1, num_layers = 1, drop_rate = 0.0009001480178615212).to(device)
model

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0012874179807017348)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 200
early_stop_count = 0
min_val_loss = float('inf')
train_loss = []
validation_loss = []
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch, y_batch

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch, y_batch
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
    train_loss.append(loss)
    validation_loss.append(val_loss)
    print(f"Epoch {epoch + 1}/{epochs}, train loss: {loss}, validation Loss: {val_loss}")

train_losses_np = torch.tensor(train_loss).detach().cpu().numpy()
test_loss_np = torch.tensor(validation_loss).detach().cpu().numpy()

model.eval()
predictions = []
actual = []
test_loss = []
with torch.no_grad():
      for batch in val_loader:
        x_batch, y_batch = batch
        x_batch = x_batch
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        test_loss.append(loss.detach().cpu().numpy())
        actual.append(y_batch.detach().cpu().numpy())
        predictions.append(outputs.detach().cpu().numpy())

predictions = np.vstack(predictions)
actual = np.vstack(actual)
test_loss = np.vstack(test_loss)

def mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()

   return mean_diff

print(f"MSE : {mean_squared_error(actual , predictions)}")

def root_mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   rmse_val = np.sqrt(mean_diff)
   return rmse_val
print(f"RMSE : {root_mean_squared_error(actual , predictions)}")

def mean_absolute_error(act, pred):
    diff = pred - act
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff
print(f"MAE : {mean_absolute_error(actual , predictions)}")

k = pd.DataFrame(predictions.squeeze(-1))
m = pd.DataFrame(actual.squeeze(-1))

plt.plot([x for x in range(96)], k[0][:96], label = "Predicted")
plt.plot([x for x in range(96)] , m[0][:96], label = "Actual")
plt.title("RNN Predicted vs. Actual Value")
plt.legend(loc = "upper right")


plt.plot([x for x in range(train_losses_np.size)] , train_losses_np, label = "train loss")
plt.plot([x for x in range(test_loss_np.size)] , test_loss_np , label = "val loss")
#plt.plot([x for x in range(len(val_losses[:200]))] , val_losses[:200] , label = "val loss")
plt.title("RNN train vs. val loss")
plt.xlabel("Epochs")
plt.ylabel("losses")
plt.legend()


# 2. LSTM training step

from LSTM import LSTM

model = LSTM(input_size = 1, hidden_size =100 , num_stacked_layers = 3, drop_rate = 0.1).to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)


epochs = 200
early_stop_count = 0
min_val_loss = float('inf')
train_loss = []
validation_loss = []
for epoch in tqdm(range(epochs)):
    model.train()
    for batch in train_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch, y_batch

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch, y_batch
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
    train_loss.append(loss)
    validation_loss.append(val_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Train loss: {loss},Validation Loss: {val_loss}")
    
train_losses_np = torch.tensor(train_loss).detach().cpu().numpy()
test_loss_np = torch.tensor(validation_loss).detach().cpu().numpy()

model.eval()
predictions = []
actual = []
test_loss = []
with torch.no_grad():
    for batch in val_loader:
        x_batch, y_batch = batch
        x_batch = x_batch
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        test_loss.append(loss.detach().cpu().numpy())
        actual.append(y_batch.detach().cpu().numpy())
        predictions.append(outputs.detach().cpu().numpy())
        
predictions = np.vstack(predictions)
actual = np.vstack(actual)
test_loss = np.vstack(test_loss)

def mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()

   return mean_diff

print(f"MSE : {mean_squared_error(actual , predictions)}")


def root_mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   rmse_val = np.sqrt(mean_diff)
   return rmse_val


print(f"RMSE : {root_mean_squared_error(actual , predictions)}")

def mean_absolute_error(act, pred):
    diff = pred - act
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff

print(f"MAE : {mean_absolute_error(actual , predictions)}")

k = pd.DataFrame(predictions.squeeze(-1))
m = pd.DataFrame(actual.squeeze(-1))

plt.plot([x for x in range(96)], k[0][:96], label = "Predicted")
plt.plot([x for x in range(96)] , m[0][:96], label = "Actual")
plt.title("First 96 LSTM Predicted vs Actual value")
plt.legend()

plt.plot([x for x in range(train_losses_np.size)] , train_losses_np, label = "train loss")
plt.plot([x for x in range(test_loss_np.size)] , test_loss_np , label = "val loss")
#plt.plot([x for x in range(len(val_losses[0:epoch+1]))] , val_losses[0:epoch+1] , label = "val loss")
plt.title("LSTM train vs. val loss")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend()



# 3. GRU training step

from GRU import GRU
model = GRU(in_dim = 7, hid_dim = 100, out_dim = 1, num_layer = 3 , drop_rate = 0.1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 200
early_stop_count = 0
min_test_loss = float('inf')
train_loss = []
test_loss1 = []
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch, y_batch

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in val_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch, y_batch
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
    train_loss.append(loss)
    test_loss1.append(test_loss)
    print(f"Epoch {epoch + 1}/{epochs}, train loss: {loss}, val loss: {test_loss}")
    
train_losses_np = torch.tensor(train_loss).detach().cpu().numpy()
test_loss_np = torch.tensor(test_loss1).detach().cpu().numpy()

model.eval()
predictions = []
actual = []
test_loss = []
with torch.no_grad():
    for batch in test_loader:
        x_batch, y_batch = batch
        x_batch = x_batch
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        test_loss.append(loss.detach().cpu().numpy())
        actual.append(y_batch.detach().cpu().numpy())
        predictions.append(outputs.detach().cpu().numpy())
        
predictions = np.vstack(predictions)
actual = np.vstack(actual)
test_loss = np.vstack(test_loss)

def mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()

   return mean_diff

print(f"MSE : {mean_squared_error(actual , predictions)}")

def root_mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   rmse_val = np.sqrt(mean_diff)
   return rmse_val

print(f"RMSE : {root_mean_squared_error(actual , predictions)}")

def mean_absolute_error(act, pred):
    diff = pred - act
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff

print(f"MAE : {mean_absolute_error(actual , predictions)}")

k = pd.DataFrame(predictions.squeeze(-1))
m = pd.DataFrame(actual.squeeze(-1))

plt.plot([x for x in range(169)], k[0][:169], label = "Predicted")
plt.plot([x for x in range(10)] , m[0][:10], label = "Actual")
plt.title("Multivariate GRU Predicted vs Actual value for next 2 dayes")
plt.legend()

plt.plot([x for x in range(train_losses_np.size)] , train_losses_np, label = "train loss")
plt.plot([x for x in range(test_loss_np.size)] , test_loss_np , label = "val loss")
#plt.plot([x for x in range(len(val_losses[0:epoch+1]))] , val_losses[0:epoch+1] , label = "val loss")
plt.title("Multivariate GRU train vs. val losses")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend()


# 4. 1D-CNN training step

from CNN import CNN_ForecastNet
model = CNN_ForecastNet(hidden_size=100, kernel_size=5, padding=2, drop_rate=0.1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 200
early_stop_count = 0
min_val_loss = float('inf')
train_loss = []
validation_loss = []
for epoch in tqdm(range(epochs)):
    model.train()
    for batch in train_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch, y_batch

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch, y_batch
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
    train_loss.append(loss)
    validation_loss.append(val_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Train loss: {loss},Validation Loss: {val_loss}")

train_losses_np = torch.tensor(train_loss).detach().cpu().numpy()
test_loss_np = torch.tensor(validation_loss).detach().cpu().numpy()


model.eval()
predictions = []
actual = []
test_loss = []
with torch.no_grad():
    for batch in val_loader:
        x_batch, y_batch = batch
        x_batch = x_batch
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        test_loss.append(loss.detach().cpu().numpy())
        actual.append(y_batch.detach().cpu().numpy())
        predictions.append(outputs.detach().cpu().numpy())
        
predictions = np.vstack(predictions)
actual = np.vstack(actual)
test_loss = np.vstack(test_loss)

def mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()

   return mean_diff

print(f"MSE : {mean_squared_error(actual , predictions)}")

def root_mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   rmse_val = np.sqrt(mean_diff)
   return rmse_val


print(f"RMSE : {root_mean_squared_error(actual , predictions)}")

def mean_absolute_error(act, pred):
    diff = pred - act
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff

print(f"MAE : {mean_absolute_error(actual , predictions)}")

k = pd.DataFrame(predictions.squeeze(-1))
m = pd.DataFrame(actual.squeeze(-1))

plt.plot([x for x in range(96)], k[0][:96], label = "Predicted")
plt.plot([x for x in range(96)] , m[0][:96], label = "Actual")
plt.title("First 96 LSTM Predicted vs Actual value")
plt.legend()

plt.plot([x for x in range(train_losses_np.size)] , train_losses_np, label = "train loss")
plt.plot([x for x in range(test_loss_np.size)] , test_loss_np , label = "val loss")
#plt.plot([x for x in range(len(val_losses[0:epoch+1]))] , val_losses[0:epoch+1] , label = "val loss")
plt.title("LSTM train vs. val loss")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend()


# 5. Encoder-only training step

from Encoder_only import Encoder

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

for epoch in tqdm(range(epochs)):
    model.train()
    for batch in train_loader:
        x_batch, y_batch = batch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        out, _ = model(x_batch)
        train_loss = criterion(y_batch , out)
        train_loss.backward()
        optimizer.step()

    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            out, _ = model(x_batch)
            test_loss = criterion(y_batch , out)
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
    train_losses.append(train_loss)
    test_losses1.append(test_loss)
    print(f"Epoch {epoch + 1}/{epochs}, train loss: {train_loss}, test Loss: {test_loss}")


train_losses_np = torch.tensor(train_losses).detach().cpu().numpy()
test_loss_np = torch.tensor(test_losses1).detach().cpu().numpy()

val_losses, val_preds, y_val_batch  = [], [] , []
model.eval()
for (x, y) in val_loader:
    x = x.to(device)
    y = y.to(device)
    y_pred, _ = model(x)
    loss_test = criterion(y_pred, y)  # (B,S)
    val_losses.append(loss_test.item())
    y_val_batch.append(y.detach().cpu().numpy())
    val_preds.append(y_pred.detach().cpu().numpy())
val_predicted = np.vstack(val_preds)
actual_val_batch = np.vstack(y_val_batch)

def mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()

   return mean_diff

print(f"MSE : {mean_squared_error(actual_val_batch , val_predicted)}")

def root_mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   rmse_val = np.sqrt(mean_diff)
   return rmse_val


print(f"RMSE : {root_mean_squared_error(actual_val_batch , val_predicted)}")

def mean_absolute_error(act, pred):
    diff = pred - act
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff

print(f"MAE : {mean_absolute_error(actual_val_batch , val_predicted)}")

k = pd.DataFrame(val_predicted.squeeze(-1))
m = pd.DataFrame(actual_val_batch.squeeze(-1))

plt.plot([x for x in range(96)], k[0][:96], label = "Predicted")
plt.plot([x for x in range(96)] , m[0][:96], label = "Actual")
#plt.title("First 96 Only-Ecoder Predicted vs. Actual")
plt.legend()

tp = train_losses_np*10

plt.plot([x for x in range(train_losses_np.size)] , tp, label = "train loss")
plt.plot([x for x in range(test_loss_np.size)] , test_loss_np , label = "val loss")
#plt.plot([x for x in range(len(val_losses[0:epoch+1]))] , val_losses[0:epoch+1] , label = "val loss")
plt.title("Encoder train vs. val loss")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend()


# 6. Decoder-only training step

from Decoder_only import Decoder
model = Decoder(num_layers=1, D=32, H=4, hidden_mlp_dim=32,
                                       inp_features=1, out_features=1, dropout_rate=0.1).to(device)

def create_look_ahead_mask(size, device=device):
    mask = torch.ones((size, size), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask  # (size, size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

epochs = 200
early_stop_count = 0
min_test_loss = float('inf')

#x_batch , y_batch = next(iter(train_loader))
#y_batch.unsqueeze(-1).shape

train_losses = []
test_losses1 = []

for epoch in tqdm(range(epochs)):
    model.train()
    for batch in train_loader:
        x_batch, y_batch = batch

        S = x_batch.shape[1]
        mask = create_look_ahead_mask(S)
        optimizer.zero_grad()
        out, _ = model(x_batch, mask)
        train_loss = criterion(y_batch , out)
        train_loss.backward()
        optimizer.step()

    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, y_batch = batch
            S = x_batch.shape[1]
            mask = create_look_ahead_mask(S)
            out, _ = model(x_batch , mask)
            test_loss = criterion(y_batch , out)
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
    train_losses.append(train_loss)
    test_losses1.append(test_loss)
    print(f"Epoch {epoch + 1}/{epochs}, train loss: {train_loss}, test Loss: {test_loss}")
    
train_losses_np = torch.tensor(train_losses).detach().cpu().numpy()
test_loss_np = torch.tensor(test_losses1).detach().cpu().numpy()

val_losses, val_preds, y_val_batch  = [], [] , []
model.eval()
for (x, y) in val_loader:
    x = x.to(device)
    y = y.to(device)
    S = x.shape[-2]
    y_pred, _ = model(x, mask=create_look_ahead_mask(S))
    loss_test = criterion(y_pred, y)  # (B,S)
    val_losses.append(loss_test.item())
    y_val_batch.append(y.detach().cpu().numpy())
    val_preds.append(y_pred.detach().cpu().numpy())
val_predicted = np.vstack(val_preds)
actual_val_batch = np.vstack(y_val_batch)

def mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()

   return mean_diff

print(f"MSE : {mean_squared_error(actual_val_batch , val_predicted)}")

def root_mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   rmse_val = np.sqrt(mean_diff)
   return rmse_val


print(f"RMSE : {root_mean_squared_error(actual_val_batch , val_predicted)}")

def mean_absolute_error(act, pred):
    diff = pred - act
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff

print(f"MAE : {mean_absolute_error(actual_val_batch , val_predicted)}")

k = pd.DataFrame(val_predicted.squeeze(-1))
m = pd.DataFrame(actual_val_batch.squeeze(-1))

plt.plot([x for x in range(48)], k[0][:48], label = "Predicted")
plt.plot([x for x in range(48)] , m[0][:48], label = "Actual")
plt.title("First 48 Decoder Predicted vs. Actual")
plt.legend()

plt.plot([x for x in range(train_losses_np.size)] , train_losses_np, label = "train loss")
plt.plot([x for x in range(test_loss_np.size)] , test_loss_np , label = "val loss")
#plt.plot([x for x in range(len(val_losses[0:epoch+1]))] , val_losses[0:epoch+1] , label = "val loss")
plt.title("Decoder train vs. val loss")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend()


# 7. Enc&Dec Transformer training step


from Enc_and_Dec_Transformer import Transformer

def generate_square_subsequent_mask(dim1: int, dim2: int , device = device) -> Tensor:
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


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

for epoch in tqdm(range(epochs)):
    model.train()
    for batch in train_loader:
        src, trg , trg_y = batch

        tgt_mask = generate_square_subsequent_mask(
            dim1=target_sequence_length,
            dim2=target_sequence_length
           ).to(device)
        src_mask = generate_square_subsequent_mask(
            dim1=target_sequence_length,
            dim2=input_sequence_length
           ).to(device)

        optimizer.zero_grad()
        out = model(src , trg, src_mask , tgt_mask)
        train_loss = criterion(trg_y , out)
        train_loss.backward()
        optimizer.step()

    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            src, trg , trg_y = batch
            tgt_mask = generate_square_subsequent_mask(
                dim1=target_sequence_length,
                dim2=target_sequence_length
               ).to(device)
            src_mask = generate_square_subsequent_mask(
                dim1=target_sequence_length,
                dim2=input_sequence_length
               ).to(device)

            out = model(src , trg , src_mask , tgt_mask)
            test_loss = criterion(trg_y , out)
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
    train_losses.append(train_loss)
    test_losses1.append(test_loss)
    print(f"Epoch {epoch + 1}/{epochs}, train loss: {train_loss}, val Loss: {test_loss}")
    
train_losses_np = torch.tensor(train_losses).detach().cpu().numpy()
test_loss_np = torch.tensor(test_losses1).detach().cpu().numpy()

val_losses, val_preds, y_val_batch  = [], [] , []
model.eval()
for batch in val_loader:
            src, trg , trg_y = batch
            tgt_mask = generate_square_subsequent_mask(
                dim1=target_sequence_length,
                dim2=target_sequence_length
               ).to(device)
            src_mask = generate_square_subsequent_mask(
                dim1=target_sequence_length,
                dim2=input_sequence_length
               ).to(device)

            out = model(src , trg , src_mask , tgt_mask)
            val_loss = criterion(trg_y , out)
            val_losses.append(val_loss.item())
            val_preds.append(out.detach().cpu().numpy())
            y_val_batch.append(trg_y.detach().cpu().numpy())
val_predicted = np.vstack(val_preds)
actual_val_batch = np.vstack(y_val_batch)

def mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()

   return mean_diff

print(f"MSE : {mean_squared_error(actual_val_batch , val_predicted)}")

def root_mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   rmse_val = np.sqrt(mean_diff)
   return rmse_val


print(f"RMSE : {root_mean_squared_error(actual_val_batch , val_predicted)}")

def mean_absolute_error(act, pred):
    diff = pred - act
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff

print(f"MAE : {mean_absolute_error(actual_val_batch , val_predicted)}")

k = pd.DataFrame(val_predicted.squeeze(-1))
m = pd.DataFrame(actual_val_batch.squeeze(-1))

plt.plot([x for x in range(96)], k[0][:96], label = "Predicted")
plt.plot([x for x in range(96)] , m[0][:96], label = "Actual")
plt.title("First 96 Transformer Predicted vs. Actual")
plt.legend(loc = "upper right")
plt.show()

plt.plot([x for x in range(train_losses_np.size)] , train_losses_np, label = "train loss")
plt.plot([x for x in range(test_loss_np.size)] , test_loss_np , label = "val loss")
#plt.plot([x for x in range(len(val_losses[0:epoch+1]))] , val_losses[0:epoch+1] , label = "val loss")
plt.title("Transformers train vs. val loss")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend()
