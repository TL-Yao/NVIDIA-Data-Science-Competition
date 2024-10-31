"""
MLP model training with 80 features as input
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import os
seed = 42

# 自定义 Dataset 类（如上所述）
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.features = dataframe.drop(columns=[dataframe.columns[0], dataframe.columns[-1]])
        self.labels = dataframe.iloc[:, -1] # label y
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features.iloc[idx].values, dtype=torch.float32)
        y = torch.tensor(self.labels.iloc[idx], dtype=torch.float32)
        return x, y

# load training data from csv file
def load_training_data_from_csv_file(file_path, validation_size=0.15, test_size=0.15):
    training_data = pd.read_csv(file_path)

    # split the data into training, validation, and testing 
    train_df, val_df, test_df = np.split(training_data.sample(frac=1, random_state=seed), [int(0.6*len(training_data)), int(0.8*len(training_data))])
    training_dataset = CustomDataset(train_df)
    validation_dataset = CustomDataset(val_df)
    testing_dataset = CustomDataset(test_df)

    return training_dataset, validation_dataset, testing_dataset

# create dataloader from dataset
def create_dataloader(dataset, batch_size=256, shuffle=True, num_workers=4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# using torch define an MLP for regression task, 80 features as input, 1 output, 5 hidden layers, ReLU activation function, batch normalization, L2 regularization, Adam optimizer
class MLP(torch.nn.Module):
    def __init__(self, input_size=80):
        super(MLP, self).__init__()
        # Define the architecture with 5 hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Output layer
        self.output_layer = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
    
def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=10):
    # device check cuda first, then mps, then cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}')

        evaluate_model(model, val_loader, loss_fn)

def evaluate_model(model, val_loader, loss_fn):
    # device check cuda first, then mps, then cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    val_loss = 0.0
    total_absolute_error = 0.0  # 用于计算 MAE
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
            
            # Calculate absolute error for MAE
            total_absolute_error += (outputs - targets).abs().sum().item()
            total += len(targets)
            
    # Average loss over all batches
    val_loss /= len(val_loader)
    # Calculate Mean Absolute Error (MAE)
    mae = total_absolute_error / total
    print(f'Validation Loss: {val_loss:.4f}, Mean Absolute Error (MAE): {mae:.4f}')

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

def main():
    print('loading data...')
    training_dataset, validation_dataset, testing_dataset = load_training_data_from_csv_file(os.path.join('.', 'dataset', 'pca_processed_data_sample.csv'))

    print('creating dataloader...')
    train_loader = create_dataloader(training_dataset, batch_size=1024, num_workers=8)
    val_loader = create_dataloader(validation_dataset, batch_size=1024, num_workers=8, shuffle=False)
    test_loader = create_dataloader(testing_dataset, batch_size=1024, num_workers=8, shuffle=False)

    print('creating model...')
    # input size depends on the number of features in training data
    print(f'Input size: {len(training_dataset[0][0])}')
    model = MLP(input_size=len(training_dataset[0][0]))
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print('training model...')
    train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=100)

    print('evaluating model...')
    evaluate_model(model, test_loader, loss_fn)

    print('saving model...')
    # save the model
    save_model(model, os.path.join('.', 'model', 'mlp_model.pth'))

if __name__ == '__main__':
    main()