import pandas as pd

def prepare_dataset(features_dir, lang):
    csv_file = f'{features_dir}/{lang}_features.csv'
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y
