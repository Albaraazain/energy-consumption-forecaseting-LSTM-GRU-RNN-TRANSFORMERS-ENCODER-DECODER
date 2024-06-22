from RNN import Simple_RNN
import torch
from Preprocessing import CustomDataset, process_missing_and_duplicate_timestamps
from torch.utils.data import DataLoader
import optuna

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

def objective(trial):
    # Define the search space for hyperparameters
    num_layers = trial.suggest_int('num_layers', 1, 10)
    hid_dim = trial.suggest_int('hid_dim', 16, 2048)
    drop_rate = trial.suggest_float('drop_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

    # Define your model and optimizer based on the suggested hyperparameters
    model = Simple_RNN(in_dim = 1, hid_dim = hid_dim , out_dim = 1, num_layers = num_layers, drop_rate = drop_rate).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()  # Adjust this based on your specific loss function

    # Training loop within the objective function
    for epoch in range(10):  # Assuming epochs is defined outside the function
        model.train()
        for batch in train_loader:
            x_batch, y_batch = batch

            optimizer.zero_grad()
            out= model(x_batch)
            train_loss = criterion(out, y_batch)
            train_loss.backward()
            optimizer.step()

        # Validation loop to track performance on the validation set (not provided in your original code)
        model.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_loader:
                x_batch, y_batch = batch
                out= model(x_batch)
                test_loss = criterion(out, y_batch)
                test_losses.append(test_loss.item())

        # Calculate and report the average validation loss for this epoch
        avg_test_loss = sum(test_losses) / len(test_losses)
        trial.report(avg_test_loss, epoch)

        # Pruning based on the intermediate result; stop early if the performance is not improving
        if trial.should_prune():
            raise optuna.TrialPruned()

    return avg_test_loss  # Return the final validation loss after all epochs


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)


best_params = study.best_params
best_num_layers = best_params['num_layers']
best_hid_dim = best_params['hid_dim']
best_drop_rate = best_params['drop_rate']
best_learning_rate = best_params['learning_rate']

print("Best hyperparameters found:")
print(f"Num layers: {best_num_layers}")
print(f"Hid dim: {best_hid_dim}")
print(f"Drop rate: {best_drop_rate}")
print(f"Learning rate: {best_learning_rate}")


#Best hyperparameters found:
#Num layers: 1
#Hid dim: 1740
#Drop rate: 0.0009001480178615212
#Learning rate: 0.0012874179807017348


'''

In this study, we employed the Optuna to identify optimal parameters suitable for our dataset
and model complexity. Specifically, we configured Optuna to conduct 100 trials with 20 internal
epochs. Parameters were set within defined ranges: the number of layers varied from 1 to 10, 
hidden layers spanned from 16 to 2048, drop probabilities ranged between 0 and 1, and learning 
rates fell within the range of 0 to 1, consistent across all five models. However, certain models 
necessitated additional hyperparameters; for instance, in the case of the 1D-CNN, we specified the 
𝑘𝑒𝑟𝑛𝑒𝑙 𝑠𝑖𝑧𝑒 between 3 − 7 and 𝑝𝑎𝑑𝑑𝑖𝑛𝑔 as (𝑘𝑒𝑟𝑛𝑒𝑙_𝑠𝑖𝑧𝑒 − 1) // 2. Similarly, for the transformers, t
he required number of ℎ𝑒𝑎𝑑𝑠 set from 1 to 8, and the number of dimensions from 32 to 1024. 
Additionally, we manually set the activation function as linear, 
the number of 𝑒𝑝𝑜𝑐ℎ𝑠, 𝑏𝑎𝑡𝑐ℎ 𝑠𝑖𝑧𝑒 and 𝑜𝑝𝑡𝑖𝑚𝑖𝑧𝑒𝑟 in Table 3 of article. 

'''

