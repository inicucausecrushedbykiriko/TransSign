import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Load the prepared data
data = np.load('data/prepared_data/prepared_data.npz')
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

# Load the scaler
scaler = torch.load('models/scaler.pth')
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model (same as in train.py)
class SignModel(nn.Module):
    def __init__(self):
        super(SignModel, self).__init__()
        self.fc1 = nn.Linear(X_test.shape[1], 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 11)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = SignModel()
model.load_state_dict(torch.load('models/sign_model.pth'))

# Evaluate the model
model.eval()
correct = 0
total = 0
digit_correct = [0] * 11
digit_total = [0] * 11

all_predictions = []
all_probabilities = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        probabilities = nn.functional.softmax(outputs, dim=1)  # Convert logits to probabilities
        _, predicted = torch.max(probabilities, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

        for label, prediction in zip(labels, predicted):
            digit_total[label.item()] += 1
            if label == prediction:
                digit_correct[label.item()] += 1

accuracy = correct / total
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

# Print some sample predictions for verification
print("\nSample Predictions:")
for i in range(10):
    true_label = y_test[i].item()
    predicted_label = all_predictions[i]
    probabilities = all_probabilities[i]
    print(f"True: {true_label}, Predicted: {predicted_label}, Probabilities: {probabilities}")

# Calculate and print accuracy for each digit
for i in range(11):
    if digit_total[i] > 0:
        digit_accuracy = digit_correct[i] / digit_total[i]
        print(f"Digit {i+1} Accuracy: {digit_accuracy * 100:.2f}%")
    else:
        print(f"Digit {i+1} Accuracy: N/A (no samples)")

# # Detailed evaluation for each test sample
# print("\nDetailed Evaluation:")
# for i in range(len(y_test)):
#     true_label = y_test[i].item()
#     predicted_label = all_predictions[i]
#     probabilities = all_probabilities[i]
#     print(f"Sample {i+1}: True: {true_label}, Predicted: {predicted_label}, Probabilities: {probabilities}")
