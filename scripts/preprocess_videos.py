import os
import cv2

def preprocess_videos(data_dir, processed_dir):
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"Created processed directory: {processed_dir}")

    for lang in ['asl', 'csl']:
        lang_dir = os.path.join(data_dir, lang)
        processed_lang_dir = os.path.join(processed_dir, lang)
        if not os.path.exists(processed_lang_dir):
            os.makedirs(processed_lang_dir)
            print(f"Created processed language directory: {processed_lang_dir}")

        for label in os.listdir(lang_dir):
            label_dir = os.path.join(lang_dir, label)
            processed_label_dir = os.path.join(processed_lang_dir, label)
            if os.path.isdir(label_dir):
                if not os.path.exists(processed_label_dir):
                    os.makedirs(processed_label_dir)
                    print(f"Created processed label directory: {processed_label_dir}")
                for video_file in os.listdir(label_dir):
                    if video_file.endswith('.mp4'):
                        video_path = os.path.join(label_dir, video_file)
                        print(f"Processing video: {video_path}")
                        process_video(video_path, processed_label_dir)

def process_video(video_path, output_dir):
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
        frame_count += 1
    cap.release()
    print(f"Processed {frame_count} frames from {video_path}")

if __name__ == "__main__":
    data_dir = './data/sign_videos'
    processed_dir = './data/processed_frames'
    preprocess_videos(data_dir, processed_dir)
