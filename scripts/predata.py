import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_data(folder):
    data = []
    labels = []
    for i in range(1, 12):
        filename = os.path.join(folder, f"{i}.csv")
        df = pd.read_csv(filename, header=None)
        data.append(df.values)
        labels.extend([i-1] * len(df))
    data = np.vstack(data)
    labels = np.array(labels)
    return data, labels

asl_data, asl_labels = load_data('data/features/asl_features')
csl_data, csl_labels = load_data('data/features/csl_features')

# Combine ASL and CSL data
data = np.vstack((asl_data, csl_data))
labels = np.hstack((asl_labels, csl_labels))

# Normalize the data
data = data / np.max(data)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Ensure the save directory exists
os.makedirs('data/prepared_data', exist_ok=True)

# Save the prepared data
np.savez('data/prepared_data/prepared_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
