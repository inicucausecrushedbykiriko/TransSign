import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SignModel(nn.Module):
    def __init__(self, input_size=226):  # Match input size used during training
        super(SignModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 11)  # Output size of 11 for 11 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)  # Output layer directly
        return x

def evaluate_model(test_loader, model_path, scaler):
    model = SignModel(input_size=226)  # Match the input size from training
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0
    digit_correct = [0] * 11
    digit_total = [0] * 11

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = scaler.transform(inputs.numpy())
            inputs = torch.tensor(inputs, dtype=torch.float32)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                digit_total[label.item()] += 1
                if label == prediction:
                    digit_correct[label.item()] += 1

    overall_accuracy = correct / total
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

    for digit in range(11):
        digit_accuracy = digit_correct[digit] / digit_total[digit] if digit_total[digit] > 0 else 0
        print(f"Digit {digit + 1}: {digit_accuracy * 100:.2f}% Accuracy")

def plot_training_log(log_path, graph_path):
    epochs, losses, accuracies = [], [], []

    with open(log_path, 'r') as log_file:
        log_file.readline()  # Skip header
        for line in log_file:
            epoch, loss, accuracy = line.strip().split(',')
            epochs.append(int(epoch))
            losses.append(float(loss))
            accuracies.append(float(accuracy))

    plt.figure()
    plt.plot(epochs, losses, label='Loss', color='blue')
    plt.plot(epochs, accuracies, label='Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title('Training Loss and Accuracy')
    plt.savefig(graph_path)
    plt.close()

if __name__ == "__main__":
    prepared_data_dir = './data/prepared_data'
    graph_dir = './data/graphs'
    log_dir = './data/logs'
    os.makedirs(graph_dir, exist_ok=True)

    def load_data(language):
        X_test = np.load(os.path.join(prepared_data_dir, f'X_{language}_test.npy'))
        y_test = np.load(os.path.join(prepared_data_dir, f'y_{language}_test.npy'))
        return X_test, y_test

    for language in ['asl', 'csl']:
        X_test, y_test = load_data(language)
        scaler = torch.load(f'models/scaler_{language}.pth')
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        print(f"Evaluating {language.upper()} Model...")
        evaluate_model(test_loader, f'models/{language}_model.pth', scaler)
        plot_training_log(
            os.path.join(log_dir, f'{language}_training_log.csv'), 
            os.path.join(graph_dir, f'{language}_training.png')
        )
