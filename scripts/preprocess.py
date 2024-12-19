import mediapipe as mp
import cv2
import csv
import os
import numpy as np
from multiprocessing import Pool, cpu_count

# Initialize Mediapipe modules globally (to avoid re-initializing in each process)
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
mp_pose = mp.solutions.pose.Pose(static_image_mode=True)

def extract_keypoints(image_path):
    """Extract hand, face, and shoulder key points from an image."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize keypoints with placeholders
    hand_keypoints = [(0, 0, 0)] * 42  # 21 points per hand
    face_keypoints = [(0, 0, 0)] * 15  # 15 points for face
    shoulder_keypoints = [(0, 0, 0)] * 2  # 2 points for shoulders

    # Hand landmarks
    hand_results = mp_hands.process(image_rgb)
    if hand_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            if i == 0:  # First hand
                hand_keypoints[:21] = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            elif i == 1:  # Second hand
                hand_keypoints[21:] = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

    # Face landmarks
    face_results = mp_face.process(image_rgb)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            selected_indices = [1, 33, 133, 159, 145, 160, 144, 362, 263, 387, 373, 388, 380, 78, 308]
            face_keypoints = [(face_landmarks.landmark[i].x,
                               face_landmarks.landmark[i].y,
                               face_landmarks.landmark[i].z) for i in selected_indices]

    # Shoulder landmarks
    pose_results = mp_pose.process(image_rgb)
    if pose_results.pose_landmarks:
        for idx, landmark in zip([11, 12], range(2)):  # Left and right shoulders
            lm = pose_results.pose_landmarks.landmark[idx]
            shoulder_keypoints[landmark] = (lm.x, lm.y, lm.z)

    return hand_keypoints + face_keypoints + shoulder_keypoints

def process_video(video_image_folder, output_csv, max_frames, label):
    """Process all frames in a single video folder."""
    frame_data = []
    frame_files = sorted(os.listdir(video_image_folder))

    for frame_file in frame_files:
        frame_path = os.path.join(video_image_folder, frame_file)
        if os.path.isfile(frame_path):
            keypoints = extract_keypoints(frame_path)
            if keypoints:
                frame_data.append([frame_file] + [val for kp in keypoints for val in kp])

    # Handle padding
    num_features = len(frame_data[0]) - 1 if frame_data else 59 * 3  # Exclude 'Frame' column
    while len(frame_data) < max_frames:
        frame_data.append(["PAD"] + [0] * num_features)  # Add padding frames

    # Add label to each frame
    for row in frame_data:
        row.append(label)

    # Save to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame'] + [f'Point_{i}_{axis}' for i in range(1, 60) for axis in ['X', 'Y', 'Z']] + ['Label'])
        writer.writerows(frame_data)
    print(f"Processed {video_image_folder} and saved features to {output_csv}")

def calculate_max_frames(image_dir):
    """Calculate the maximum number of frames in all videos."""
    max_frames = 0
    for lang in ['asl', 'csl']:
        lang_folder = os.path.join(image_dir, lang)
        for digit_folder in os.listdir(lang_folder):
            digit_image_folder = os.path.join(lang_folder, digit_folder)
            if not os.path.isdir(digit_image_folder):
                continue

            for video_folder in os.listdir(digit_image_folder):
                video_image_folder = os.path.join(digit_image_folder, video_folder)
                if not os.path.isdir(video_image_folder):
                    continue

                num_frames = len([f for f in os.listdir(video_image_folder) if os.path.isfile(os.path.join(video_image_folder, f))])
                max_frames = max(max_frames, num_frames)
    return max_frames

def process_all_images_parallel(image_dir, output_features_dir):
    """Process all video folders in parallel."""
    max_frames = calculate_max_frames(image_dir)  # Automatically determine max frames
    print(f"Maximum frames determined: {max_frames}")

    tasks = []
    for lang in ['asl', 'csl']:
        lang_folder = os.path.join(image_dir, lang)
        features_folder = os.path.join(output_features_dir, f"{lang}_features")
        os.makedirs(features_folder, exist_ok=True)

        for digit_folder in os.listdir(lang_folder):
            digit_image_folder = os.path.join(lang_folder, digit_folder)
            if not os.path.isdir(digit_image_folder):
                continue

            label = int(digit_folder)  # Use folder name as label (e.g., '1', '2', etc.)
            for video_folder in os.listdir(digit_image_folder):
                video_image_folder = os.path.join(digit_image_folder, video_folder)
                if not os.path.isdir(video_image_folder):
                    continue

                output_csv = os.path.join(features_folder, f"{video_folder}.csv")
                tasks.append((video_image_folder, output_csv, max_frames, label))

    # Use multiprocessing to parallelize
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(process_video, tasks)

    print(f"Processing completed. Maximum frames used: {max_frames}")

if __name__ == "__main__":
    data_dir = './data'
    process_all_images_parallel(
        image_dir=os.path.join(data_dir, 'sign_images'),
        output_features_dir=os.path.join(data_dir, 'features')
    )
