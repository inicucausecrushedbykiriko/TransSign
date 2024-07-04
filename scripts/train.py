import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Load the prepared data
prepared_data_dir = './data/prepared_data'

def load_data(language):
    X_train = np.load(os.path.join(prepared_data_dir, f'X_{language}_train.npy'))
    y_train = np.load(os.path.join(prepared_data_dir, f'y_{language}_train.npy'))
    X_test = np.load(os.path.join(prepared_data_dir, f'X_{language}_test.npy'))
    y_test = np.load(os.path.join(prepared_data_dir, f'y_{language}_test.npy'))
    return X_train, y_train, X_test, y_test

X_asl_train, y_asl_train, X_asl_test, y_asl_test = load_data('asl')
X_csl_train, y_csl_train, X_csl_test, y_csl_test = load_data('csl')

# Standardize the data
scaler_asl = StandardScaler()
X_asl_train = scaler_asl.fit_transform(X_asl_train)
X_asl_test = scaler_asl.transform(X_asl_test)

scaler_csl = StandardScaler()
X_csl_train = scaler_csl.fit_transform(X_csl_train)
X_csl_test = scaler_csl.transform(X_csl_test)

# Convert to PyTorch tensors
X_asl_train = torch.tensor(X_asl_train, dtype=torch.float32)
X_asl_test = torch.tensor(X_asl_test, dtype=torch.float32)
y_asl_train = torch.tensor(np.argmax(y_asl_train, axis=1), dtype=torch.long)
y_asl_test = torch.tensor(np.argmax(y_asl_test, axis=1), dtype=torch.long)

X_csl_train = torch.tensor(X_csl_train, dtype=torch.float32)
X_csl_test = torch.tensor(X_csl_test, dtype=torch.float32)
y_csl_train = torch.tensor(np.argmax(y_csl_train, axis=1), dtype=torch.long)
y_csl_test = torch.tensor(np.argmax(y_csl_test, axis=1), dtype=torch.long)

# Create DataLoader
train_dataset_asl = TensorDataset(X_asl_train, y_asl_train)
test_dataset_asl = TensorDataset(X_asl_test, y_asl_test)
train_loader_asl = DataLoader(train_dataset_asl, batch_size=32, shuffle=True)
test_loader_asl = DataLoader(test_dataset_asl, batch_size=32, shuffle=False)

train_dataset_csl = TensorDataset(X_csl_train, y_csl_train)
test_dataset_csl = TensorDataset(X_csl_test, y_csl_test)
train_loader_csl = DataLoader(train_dataset_csl, batch_size=32, shuffle=True)
test_loader_csl = DataLoader(test_dataset_csl, batch_size=32, shuffle=False)

# Define the model
class SignModel(nn.Module):
    def __init__(self):
        super(SignModel, self).__init__()
        self.fc1 = nn.Linear(X_asl_train.shape[1], 128)
        self.fc2 = nn.Linear(128, 11)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(train_loader, model_path, scaler, scaler_path):
    model = SignModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1000
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader)}")

    torch.save(model.state_dict(), model_path)
    torch.save(scaler, scaler_path)
    print(f"Model saved to {model_path}")

# Train and save the ASL model
train_model(train_loader_asl, 'models/asl_model.pth', scaler_asl, 'models/scaler_asl.pth')

# Train and save the CSL model
train_model(train_loader_csl, 'models/csl_model.pth', scaler_csl, 'models/scaler_csl.pth')

