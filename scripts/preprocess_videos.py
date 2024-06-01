import os
import cv2

def preprocess():
    data_dir = 'data'
    processed_dir = 'data/processed'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    for lang in ['asl', 'csl']:
        lang_dir = os.path.join(data_dir, lang)
        if os.path.isdir(lang_dir):
            for label in os.listdir(lang_dir):
                label_dir = os.path.join(lang_dir, label)
                if os.path.isdir(label_dir):
                    for video_file in os.listdir(label_dir):
                        if video_file.endswith('.mp4'):
                            video_path = os.path.join(label_dir, video_file)
                            process_video(video_path, os.path.join(processed_dir, f"{lang}_{label}_{video_file}"))

def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_file = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_file, frame)
        frame_count += 1
    cap.release()
