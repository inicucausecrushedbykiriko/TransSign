import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_training_data(csv_file, output_features, output_labels):
    """
    Prepares training data by separating features and labels, and normalizing features.

    Args:
        csv_file (str): Path to the merged CSV file.
        output_features (str): Path to save the normalized features (NumPy format).
        output_labels (str): Path to save the labels (NumPy format).
    """
    data = pd.read_csv(csv_file)
    features = data.iloc[:, 1:-1].values  # Exclude 'Frame' and 'Label' columns
    labels = data['Label'].values  # Extract the 'Label' column

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Save features and labels
    np.save(output_features, features)
    np.save(output_labels, labels)
    print(f"Features saved to {output_features}")
    print(f"Labels saved to {output_labels}")

if __name__ == "__main__":
    csv_file = './data/dataset.csv'
    output_features = './data/features.npy'
    output_labels = './data/labels.npy'
    prepare_training_data(csv_file, output_features, output_labels)
