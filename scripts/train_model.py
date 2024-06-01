import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import SignLanguageDataset
from utils.model import SignLanguageModel

def train():
    dataset = SignLanguageDataset('data/processed')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SignLanguageModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), 'models/sign_language_model.pth')
