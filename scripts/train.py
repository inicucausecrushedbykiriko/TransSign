import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class TransformerClassifier(nn.Module):
    """
    A Transformer-based classifier for hand gesture recognition.
    """
    def __init__(self, input_dim, max_len, num_classes, num_heads, num_layers, d_ff):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=d_ff, dropout=0.3, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

def train_model(features, labels, model_save_path, max_frames, num_features):
    """
    Train a Transformer-based model for hand gesture classification.

    Args:
        features (np.array): Normalized feature data.
        labels (np.array): Label data.
        model_save_path (str): Path to save the trained model.
        max_frames (int): Maximum number of frames per sequence.
        num_features (int): Number of features per frame.
    """
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels before adjustment: {np.unique(labels)}")

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features.reshape(-1, num_features)).reshape(features.shape)

    # Adjust labels to zero-indexing
    labels = labels - 1
    print(f"Unique labels after adjustment: {np.unique(labels)}")

    # Reshape features
    total_frames = features.shape[0]
    usable_frames = (total_frames // max_frames) * max_frames
    features = features[:usable_frames]
    labels = labels[:usable_frames // max_frames]

    print(f"Trimmed features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    # Convert to tensors
    X_tensor = torch.tensor(features, dtype=torch.float32).reshape(-1, max_frames, num_features)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    # Split into train and validation sets
    train_ratio = 0.8
    train_size = int(X_tensor.size(0) * train_ratio)
    X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model
    num_classes = len(set(labels))
    model = TransformerClassifier(
        input_dim=num_features, 
        max_len=max_frames, 
        num_classes=num_classes, 
        num_heads=3,  # Adjusted to a valid divisor of 177
        num_layers=2, 
        d_ff=256
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # Validate the model
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_accuracy = correct / total
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save the model and scaler
    torch.save(model.state_dict(), model_save_path)
    torch.save(scaler, './models/scaler.pth')
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    # Load data
    features = np.load('./data/features.npy')
    labels = np.load('./data/labels.npy')

    # Configuration
    model_save_path = './models/transformer_model.pth'
    max_frames = 236  # Update max frames
    num_features = 59 * 3

    train_model(features, labels, model_save_path, max_frames, num_features)
