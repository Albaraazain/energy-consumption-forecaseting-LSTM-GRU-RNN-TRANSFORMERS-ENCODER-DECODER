import torch
from Preprocessing import CustomDataset, process_missing_and_duplicate_timestamps
from torch.utils.data import DataLoader
import optuna
from config import device, n_future, DATASET_PATH, BATCH_SIZE
from RNNBasedModels.RNN import Simple_RNN
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("\nStarting hyperparameter optimization")
print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

def objective(trial):
    print(f"\nStarting trial {trial.number}")
    try:
        # Define the search space for hyperparameters
        num_layers = trial.suggest_int('num_layers', 1, 10)
        hid_dim = trial.suggest_int('hid_dim', 16, 2048)
        drop_rate = trial.suggest_float('drop_rate', 0.0, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

        print(f"Trial parameters:")
        print(f"num_layers: {num_layers}")
        print(f"hid_dim: {hid_dim}")
        print(f"drop_rate: {drop_rate}")
        print(f"learning_rate: {learning_rate}")

        # Define your model
        model = Simple_RNN(
            in_dim=1,
            hid_dim=hid_dim,
            out_dim=1,
            num_layers=num_layers,
            drop_rate=drop_rate
        ).to(device)

        print(f"Model created successfully")
        print(f"Model structure:\n{model}")

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        # Training loop
        for epoch in range(10):
            print(f"\nEpoch {epoch+1}/10")
            model.train()
            epoch_losses = []

            for batch_idx, batch in enumerate(train_loader):
                if batch_idx == 0:  # Print shape info for first batch only
                    print(f"Batch shapes:")
                    print(f"x_batch: {batch[0].shape}")
                    print(f"trg: {batch[1].shape}")
                    print(f"y_batch: {batch[2].shape}")

                x_batch, _, y_batch = batch

                optimizer.zero_grad()
                out = model(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

                if batch_idx % 100 == 0:  # Print every 100 batches
                    print(f"Batch {batch_idx}: Loss = {loss.item():.6f}")

            avg_epoch_loss = np.mean(epoch_losses)
            print(f"Average epoch loss: {avg_epoch_loss:.6f}")

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    x_batch, _, y_batch = batch
                    out = model(x_batch)
                    val_loss = criterion(out, y_batch)
                    val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)
            print(f"Validation loss: {avg_val_loss:.6f}")

            trial.report(avg_val_loss, epoch)

            if trial.should_prune():
                print("Trial pruned")
                raise optuna.exceptions.TrialPruned()

        print(f"Trial completed successfully")
        return avg_val_loss

    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    try:
        print("\nLoading and preparing data...")
        # Load and prepare data
        train_set, val_set, test_set = process_missing_and_duplicate_timestamps(filepath=DATASET_PATH)

        print("\nCreating datasets...")
        # Create datasets
        train_dataset = CustomDataset(train_set, 168, 48, multivariate=False, target_feature=0)
        val_dataset = CustomDataset(val_set, 168, 48, multivariate=False, target_feature=0)
        test_dataset = CustomDataset(test_set, 168, 48, multivariate=False, target_feature=0)

        print("\nCreating data loaders...")
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

        print("\nStarting optimization study...")
        # Create and run study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)

        print("\nOptimization completed!")
        print("Best hyperparameters found:")
        print(f"Num layers: {study.best_params['num_layers']}")
        print(f"Hidden dim: {study.best_params['hid_dim']}")
        print(f"Drop rate: {study.best_params['drop_rate']}")
        print(f"Learning rate: {study.best_params['learning_rate']}")

    except Exception as e:
        print(f"\nMain execution failed with error: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
