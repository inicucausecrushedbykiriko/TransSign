import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model(X, y, lang):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"{lang.upper()} Model Accuracy: {accuracy}")
    model_dir = f'../models/{lang}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(model, f'{model_dir}/sign_language_model.pkl')

if __name__ == "__main__":
    for lang in ['asl', 'csl']:
        X = np.load(f'../data/X_{lang}.npy')
        y = np.load(f'../data/y_{lang}.npy')
        train_model(X, y, lang)

