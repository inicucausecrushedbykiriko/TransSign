import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths to your data
asl_features_path = 'data/features/asl_features'
csl_features_path = 'data/features/csl_features'

def load_data(folder_path):
    data = []
    labels = []
    for label in range(1, 12):
        file_path = os.path.join(folder_path, f'{label}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None)
            data.append(df.values)
            labels.append(np.full((df.shape[0],), label - 1))  # Labels from 0 to 10
    return np.vstack(data), np.hstack(labels)

# Load ASL and CSL data separately
asl_data, asl_labels = load_data(asl_features_path)
csl_data, csl_labels = load_data(csl_features_path)

# Optionally, you can combine ASL and CSL data if you want to train on both
combined_data = np.vstack((asl_data, csl_data))
combined_labels = np.hstack((asl_labels, csl_labels))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_data, combined_labels, test_size=0.2, random_state=42)

# Save the prepared data
np.savez('data/prepared_data/prepared_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
