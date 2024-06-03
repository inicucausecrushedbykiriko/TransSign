import os
import numpy as np
from scripts.extract_features import extract_features

def prepare_dataset(processed_dir):
    X = []
    y = []
    for lang in ['asl', 'csl']:
        lang_dir = os.path.join(processed_dir, lang)
        for label in os.listdir(lang_dir):
            label_dir = os.path.join(lang_dir, label)
            for image_file in os.listdir(label_dir):
                if image_file.endswith('.jpg'):
                    image_path = os.path.join(label_dir, image_file)
                    features = extract_features(image_path)
                    if features is not None:
                        X.append(features)
                        y.append(int(label))
    return np.array(X), np.array(y)

if __name__ == "__main__":
    processed_dir = '../data/processed_frames'
    X, y = prepare_dataset(processed_dir)
    np.save('../data/X.npy', X)
    np.save('../data/y.npy', y)
