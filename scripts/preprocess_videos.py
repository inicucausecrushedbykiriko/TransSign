import os
import cv2
from scripts.extract_features import extract_features, save_features_to_csv

def create_if_not_exsit(target_dir, msg):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"{msg}: {target_dir}")

def preprocess_a_video(data_dir, processed_dir, features_dir, lang, output_csv):
    lang_dir = os.path.join(data_dir, lang)
    processed_lang_dir = os.path.join(processed_dir, lang)
    create_if_not_exsit(processed_lang_dir, "Created processed language directory")
    for video_file in os.listdir(lang_dir):
        if video_file.endswith('.mp4'):
            processed_label_dir = os.path.join(processed_lang_dir, video_file.split('.')[0])
            create_if_not_exsit(processed_label_dir, "Created processed label directory")
            video_path = os.path.join(lang_dir, video_file)
            print(f"Processing video: {video_path}")
            process_video(video_path, processed_label_dir, output_csv)

def preprocess_videos(data_dir, processed_dir, features_dir):
    create_if_not_exsit(processed_dir, "Created processed directory")
    create_if_not_exsit(features_dir, "Created features directory")
    for lang in ['asl', 'csl']:
        output_csv = os.path.join(features_dir, f'{lang}_features.csv')
        preprocess_a_video(data_dir, processed_dir, features_dir, lang, output_csv)

def process_video(video_path, output_dir, output_csv):
    print(f"Processing video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        return
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_file = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_file, frame)
        print(f"Saved frame: {frame_file}")
        
        # Extract features and save to CSV
        features = extract_features(frame_file)
        if features:
            print(f"Extracted features from frame: {frame_file}")
            save_features_to_csv(features, output_csv)
        
        frame_count += 1
    cap.release()
    print(f"Processed {frame_count} frames from {video_path}")

if __name__ == "__main__":
    data_dir = './data/sign_videos'
    processed_dir = './data/sign_images'
    features_dir = './data/features'
    preprocess_videos(data_dir, processed_dir, features_dir)
