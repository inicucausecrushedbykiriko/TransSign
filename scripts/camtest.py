import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from evaluate import SignModel
import mediapipe as mp
import os

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def process_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_world_landmarks:
        print("No hand landmarks detected, recording zeros.")
        return np.zeros(226)  # Adjusted to match model's input size

    hand_data = []
    for hand_landmarks in results.multi_hand_world_landmarks:
        for landmark in hand_landmarks.landmark:
            hand_data.extend([landmark.x, landmark.y, landmark.z])

    if len(hand_data) == 63:
        hand_data.extend([0] * (226 - 63))  # Padding to reach 226 features

    return np.array(hand_data)

def predict_digit(frame, model, scaler, device):
    features = process_frame(frame).reshape(1, -1)
    features = scaler.transform(features)
    features = torch.tensor(features, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(features)
        probabilities = nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)

    predicted_digit = predicted.item() + 1
    return predicted_digit, probabilities.cpu().numpy()

def load_model_and_scaler(language, device):
    model_path = f'./models/{language}_model.pth'
    scaler_path = f'./models/scaler_{language}.pth'

    model = SignModel(input_size=226).to(device)  # Ensure input size matches the model used in training
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scaler = torch.load(scaler_path)
    return model, scaler

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_type = input("Enter the type of input (asl/csl): ").strip().lower()

    if input_type not in ['asl', 'csl']:
        print("Invalid input type. Please enter 'asl' or 'csl'.")
        return

    model, scaler = load_model_and_scaler(input_type, device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            digit, probabilities = predict_digit(frame, model, scaler, device)
            print(f"Predicted Digit: {digit}")
            print(f"Probabilities: {probabilities}")

            translation_path = f'./data/show/{("csl" if input_type == "asl" else "asl")}/{digit}.png'
            translation_image = cv2.imread(translation_path)

            if translation_image is not None:
                translation_image = cv2.resize(translation_image, (200, 200))
                frame[0:200, 0:200] = translation_image
            else:
                print(f"Warning: Translation image for digit {digit} not found.")

            cv2.putText(frame, f"Predicted Digit: {digit}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Hand Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Video stream closed.")

if __name__ == "__main__":
    main()
