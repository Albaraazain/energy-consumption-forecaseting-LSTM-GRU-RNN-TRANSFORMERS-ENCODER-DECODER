import pandas as pd
import numpy as np

import torch

from torch.utils.data import TensorDataset , DataLoader, Dataset

try:
    import google.colab
    COLAB = True
    print("Note: using Google CoLab")
except:
    print("Note: not using Google CoLab")
    COLAB = False

device = (
    "mps"
    if getattr(torch, "has_mps", False)
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")



def is_ne_in_df(df:pd.DataFrame):
    for col in df.columns:
        true_bool = (df[col] == "n/e")
        if any(true_bool):
            return True
    return False

def to_numeric_and_downcast_data(df: pd.DataFrame):
    fcols = df.select_dtypes('float').columns
    icols = df.select_dtypes('integer').columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')
    return df

def process_missing_and_duplicate_timestamps(filepath, train_size=80, val_size=50, verbose=True):
    df = pd.read_csv(filepath)
    df.sort_values('Datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    indices_to_remove = []
    rows_to_add = []
    hour_counter = 1
    prev_date = ''

    if verbose:
        print(filepath)

    for index, row in df.iterrows():
        date_str = row['Datetime']

        year_str = date_str[0:4]
        month_str = date_str[5:7]
        day_str = date_str[8:10]
        hour_str = date_str[11:13]
        tail_str = date_str[14:]

        def date_to_str():
            return '-'.join([year_str, month_str, day_str]) + ' ' + ':'.join([hour_str, tail_str])

        def date_with_hour(hour):
            hour = '0' + str(hour) if hour < 10 else str(hour)
            return '-'.join([year_str, month_str, day_str]) + ' ' + ':'.join([hour, tail_str])

        if hour_counter != int(hour_str):
            if prev_date == date_to_str():
                # Duplicate datetime, calculate the average and keep only one
                average = int((df.iat[index, 1] + df.iat[index - 1, 1]) / 2)  # Get the average
                df.iat[index, 1] = average
                indices_to_remove.append(index - 1)
                if verbose:
                    print('Duplicate ' + date_to_str() + ' with average ' + str(average))
            elif hour_counter < 23:
                # Missing datetime, add it using the average of the previous and next for the consumption (MWs)
                average = int((df.iat[index, 1] + df.iat[index - 1, 1]) / 2)
                rows_to_add.append(pd.Series([date_with_hour(hour_counter), average], index=df.columns))
                if verbose:
                    print('Missing ' + date_with_hour(hour_counter) + ' with average ' + str(average))
            else:
                print(date_to_str() + ' and hour_counter ' + str(hour_counter) + " with previous: " + prev_date)

            # Adjust for the missing/duplicate value
            if prev_date < date_to_str():
                hour_counter = (hour_counter + 1) % 24
            else:
                hour_counter = (hour_counter - 1) if hour_counter - 1 > 0 else 0

        # Increment the hour
        hour_counter = (hour_counter + 1) % 24
        prev_date = date_str

    df.drop(indices_to_remove, inplace=True)
    if rows_to_add:
        new_rows = pd.concat(rows_to_add, axis=1).transpose()
        df = pd.concat([df, new_rows], ignore_index=True)  # Concatenating the new rows

    # New rows are added at the end, sort them and also recalculate the indices
    df.sort_values('Datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.set_index("Datetime")
    if is_ne_in_df(df):
        raise ValueError("data frame contains 'n/e' values. These must be handled")
    df = to_numeric_and_downcast_data(df)
    data_mean = df.mean(axis=0)
    data_std = df.std(axis=0)
    df = (df - data_mean) / data_std
    stats = (data_mean, data_std)

    train = df[:len(df) * train_size // 100]
    val = df[len(train) : len(train) + ((len(df) - len(train)) * val_size) // 100]
    test = df[len(val) + len(train) : ]

    #scaler = MinMaxScaler()
    #train = scaler.fit_transform(train)
    #val = scaler.fit_transform(val)
    #test = scaler.fit_transform(test)

    train = np.array(train, dtype=np.float32)
    val = np.array(val, dtype=np.float32)
    test = np.array(test, dtype=np.float32)

    return train, val, test


path = "Datasets/DUQ_hourly.csv"
train_set , val_set , test_set = process_missing_and_duplicate_timestamps(filepath = path)


class CustomDataset(Dataset):
    def __init__(self, sequence, input_sequence_length, target_sequence_length, multivariate=False, target_feature=None):
        print(f"\nInitializing CustomDataset:")
        print(f"Input sequence shape: {sequence.shape}")
        print(f"Sequence dtype before conversion: {sequence.dtype}")

        # Convert sequence to float32 at initialization
        try:
            self.sequence = torch.tensor(sequence, dtype=torch.float32)
            print(f"Sequence successfully converted to tensor with shape: {self.sequence.shape}")
            print(f"Tensor dtype: {self.sequence.dtype}")
        except Exception as e:
            print(f"Error converting sequence to tensor: {str(e)}")
            raise

        self.input_sequence_length = input_sequence_length
        self.target_sequence_length = target_sequence_length
        self.window_size = input_sequence_length + target_sequence_length
        self.multivariate = multivariate
        self.target_feature = target_feature

        print(f"Dataset initialization complete. Window size: {self.window_size}")

    def __len__(self):
        return len(self.sequence) - self.window_size + 1

    def __getitem__(self, idx):
        print(f"\nFetching item at index: {idx}")
        try:
            src = self.sequence[idx:idx + self.input_sequence_length]
            trg = self.sequence[idx + self.input_sequence_length - 1:idx + self.window_size - 1]

            # print(f"Source shape: {src.shape}")
            # print(f"Target shape: {trg.shape}")

            if self.multivariate:
                trg_y = self.sequence[idx + self.input_sequence_length:idx + self.input_sequence_length + self.target_sequence_length, self.target_feature].unsqueeze(1)
                print(f"Multivariate target shape: {trg_y.shape}")
            else:
                trg_y = self.sequence[idx + self.input_sequence_length:idx + self.input_sequence_length + self.target_sequence_length]
                print(f"Univariate target shape: {trg_y.shape}")

            # Debug output shapes and types
            # print(f"Final shapes - src: {src.shape}, trg: {trg.shape}, trg_y: {trg_y.shape}")
            # print(f"Data types - src: {src.dtype}, trg: {trg.dtype}, trg_y: {trg_y.dtype}")

            return src.to(device), trg.to(device), trg_y.to(device)

        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {str(e)}")
            raise

    

input_sequence_length = 168
target_sequence_length = 48

train_dataset = CustomDataset(train_set, input_sequence_length, target_sequence_length, multivariate=False, target_feature=0)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False , drop_last=True)
test_dataset = CustomDataset(test_set, input_sequence_length, target_sequence_length, multivariate=False, target_feature=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False , drop_last=True)
val_dataset = CustomDataset(val_set, input_sequence_length, target_sequence_length, multivariate=False, target_feature=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False , drop_last=True)

