import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scripts.model import get_model

def train_model(X, y, lang):
    input_size = X.shape[1]
    hidden_size = 128
    output_size = len(set(y))  # Assuming y contains the class labels

    model = get_model(input_size, hidden_size, output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    batch_size = 32

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), f'./models/{lang}_sign_language_model.pth')
    print(f"Model saved to ./models/{lang}_sign_language_model.pth")
