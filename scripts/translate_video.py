import os
import cv2
import torch
from utils.model import SignLanguageModel
from utils.dataset import SignLanguageDataset

def translate(input_video, output_video, direction):
    model = SignLanguageModel()
    model.load_state_dict(torch.load('models/sign_language_model.pth'))
    model.eval()

    cap = cv2.VideoCapture(input_video)
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0)
        output = model(frame_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
        # Translate label and overlay on frame
        translated_label = translate_label(predicted_label, direction)
        cv2.putText(frame, translated_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        out.write(frame)
    
    cap.release()
    out.release()

def translate_label(label, direction):
    asl_to_csl = {0: '0_CSL', 1: '1_CSL', 2: '2_CSL', 3: '3_CSL', 4: '4_CSL', 5: '5_CSL', 6: '6_CSL', 7: '7_CSL', 8: '8_CSL', 9: '9_CSL'}
    csl_to_asl = {0: '0_ASL', 1: '1_ASL', 2: '2_ASL', 3: '3_ASL', 4: '4_ASL', 5: '5_ASL', 6: '6_ASL', 7: '7_ASL', 8: '8_ASL', 9: '9_ASL'}
    if direction == 'asl_to_csl':
        return asl_to_csl[label]
    elif direction == 'csl_to_asl':
        return csl_to_asl[label]
