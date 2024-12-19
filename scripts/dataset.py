import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np

class SignLanguageDataset(Dataset):
    def __init__(self, csv_folder, max_frames=236, feature_dim=177):
        self.data = []
        self.labels = []
        
        # Read all CSV files
        for csv_file in os.listdir(csv_folder):
            if csv_file.endswith('.csv'):
                csv_path = os.path.join(csv_folder, csv_file)
                df = pd.read_csv(csv_path)
                
                # Extract features
                features = df.iloc[:, 3:].values  # Exclude 'Time', 'Digit', 'LanguageID'
                digit = df['Digit'][0]
                language_id = df['LanguageID'][0]

                # Pad sequences if necessary
                if len(features) < max_frames:
                    pad_size = max_frames - len(features)
                    padding = np.zeros((pad_size, feature_dim))
                    features = np.vstack([features, padding])
                
                self.data.append(features)
                self.labels.append((digit, language_id))
        
        # Convert to numpy arrays
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.data[idx]
        digit_label = self.labels[idx][0]
        language_label = self.labels[idx][1]
        return torch.tensor(features), torch.tensor(digit_label), torch.tensor(language_label)

if __name__ == "__main__":
    dataset = SignLanguageDataset(csv_folder='./data/features')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch in dataloader:
        features, digit_labels, language_labels = batch
        print(features.shape, digit_labels, language_labels)
