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

seed = 42

# 自定义 Dataset 类（如上所述）
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.features = dataframe.drop(columns=[dataframe.columns[0], dataframe.columns[1]])
        self.labels = dataframe.iloc[:, 1] # label y
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features.iloc[idx].values, dtype=torch.float32)
        y = torch.tensor(self.labels.iloc[idx], dtype=torch.float32)
        return x, y

# load training data from csv file
def load_training_data_from_csv_file(file_path, test_size=0.2):
    training_data = pd.read_csv(file_path)
    # remove the third and forth columns
    training_data = training_data.drop(columns=[training_data.columns[2], training_data.columns[3]])

    train_df, val_df = train_test_split(training_data, test_size=test_size, random_state=seed)
    training_dataset = CustomDataset(train_df)
    validation_dataset = CustomDataset(val_df)
    return training_dataset, validation_dataset

# load testing data from csv file
def load_testing_data_from_csv_file(file_path):
    testing_data = pd.read_csv(file_path)
    testing_dataset = CustomDataset(testing_data)
    return testing_dataset

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
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
            
            correct += (outputs.round() == targets).sum().item()
            total += targets.size(0)
            
    val_loss /= len(val_loader)
    accuracy = correct / total
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

def main():
    training_dataset, validation_dataset = load_training_data_from_csv_file('../odsc-2024-nvidia-hackathon/data_sample_xs.csv')
    testing_dataset = load_testing_data_from_csv_file('../odsc-2024-nvidia-hackathon/test.csv')

    train_loader = create_dataloader(training_dataset)
    val_loader = create_dataloader(validation_dataset, shuffle=False)
    test_loader = create_dataloader(testing_dataset, shuffle=False)

    model = MLP(input_size=104)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, loss_fn, optimizer)

if __name__ == '__main__':
    main()