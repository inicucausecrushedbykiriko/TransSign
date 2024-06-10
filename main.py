import argparse
import numpy as np
from scripts.preprocess_videos import preprocess_videos
from scripts.prepare_dataset import prepare_dataset
from scripts.train_model import train_model

def main():
    parser = argparse.ArgumentParser(description='Sign Language Translator')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess videos')
    parser.add_argument('--prepare', action='store_true', help='Prepare dataset')
    parser.add_argument('--train', action='store_true', help='Train model')

    args = parser.parse_args()

    if args.preprocess:
        data_dir = './data/sign_videos'
        processed_dir = './data/sign_images'
        features_dir = './data/features'
        print(f"Preprocessing videos from {data_dir} to {processed_dir}")
        preprocess_videos(data_dir, processed_dir, features_dir)
    elif args.prepare:
        features_dir = './data/features'
        for lang in ['asl', 'csl']:
            X, y = prepare_dataset(features_dir, lang)
            np.save(f'./data/X_{lang}.npy', X)
            np.save(f'./data/y_{lang}.npy', y)
    elif args.train:
        for lang in ['asl', 'csl']:
            X = np.load(f'./data/X_{lang}.npy')
            y = np.load(f'./data/y_{lang}.npy')
            train_model(X, y, lang)
    else:
        print("No valid arguments provided. Use --help for more information.")

if __name__ == "__main__":
    main()