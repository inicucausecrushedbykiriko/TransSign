import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from evaluate import SignModel
import time
import mediapipe as mp

# Initialize MediaPipe hands in static mode for image processing
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image {image_path} not found")

    # Convert image to RGB for Mediapipe processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_world_landmarks:
        print(f"No hand landmarks detected in image {image_path}, recording zeros.")
        return np.zeros(226)  # Return 226 zeros to match model input size

    # Extract hand landmark data
    hand_data = []
    for hand_landmarks in results.multi_hand_world_landmarks:
        for landmark in hand_landmarks.landmark:
            hand_data.extend([landmark.x, landmark.y, landmark.z])

    # Pad to 226 features if less data is detected
    if len(hand_data) < 226:
        hand_data.extend([0] * (226 - len(hand_data)))

    return np.array(hand_data)

def predict_digit(image_path, model_path, scaler_path):
    print("Processing image...")
    start = time.time()
    features = process_image(image_path).reshape(1, -1)  # Extracted features
    print("Image processed in", time.time() - start, "seconds.")

    print("Loading model and scaler...")
    start = time.time()
    model = SignModel(input_size=226)  # Adjust input size to match training
    model.load_state_dict(torch.load(model_path))
    model.eval()
    scaler = torch.load(scaler_path)  # Load pre-fitted scaler
    features = scaler.transform(features)
    print("Model and scaler loaded in", time.time() - start, "seconds.")

    print("Predicting digit...")
    start = time.time()
    features = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(features)
        probabilities = nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
    print("Prediction completed in", time.time() - start, "seconds.")

    predicted_digit = predicted.item() + 1
    return predicted_digit, probabilities.numpy()

if __name__ == "__main__":
    image_path = './data/extest/hand_gesture.jpg'  # Path to the hand gesture image
    model_path = './models/asl_model.pth'
    scaler_path = './models/scaler_asl.pth'

    try:
        digit, probabilities = predict_digit(image_path, model_path, scaler_path)
        print(f"Predicted Digit: {digit}")
        print(f"Probabilities: {probabilities}")
    except Exception as e:
        print(f"Error: {e}")
