import mediapipe as mp
import cv2
import numpy as np
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extract_features(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        features = []
        for lm in landmarks.landmark:
            features.append([lm.x, lm.y])
        return features
    return None

def save_features_to_csv(features, output_csv):
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(features)

if __name__ == "__main__":
    image_path = '../data/processed_frames/asl/1/1.png'
    output_csv = '../data/features.csv'
    features = extract_features(image_path)
    if features:
        save_features_to_csv(features, output_csv)
