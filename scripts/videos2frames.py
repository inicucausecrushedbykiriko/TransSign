import os
import cv2

def extract_frames(video_path, image_folder, fps=30):
    """Extract frames from a video and save them to a folder with local numbering."""
    local_frame_id = 0  # Reset local frame ID for each video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print(f"Error: FPS for video {video_path} is 0")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    skip_interval = max(1, int(video_fps / fps))  # Ensure skip_interval is at least 1
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if the video is longer than 4 seconds
        if duration > 4 and processed_frames % (skip_interval * 2) != 0:
            processed_frames += 1
            continue

        # Save each frame with local numbering
        frame_filename = os.path.join(image_folder, f"frame_{local_frame_id:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        local_frame_id += 1
        processed_frames += 1

    cap.release()
    print(f"Processed video: {video_path} into {local_frame_id} frames.")

def process_all_videos(video_dir, output_image_dir):
    """Process all videos in a directory structure and extract frames."""
    folder_counter = {}  # Counter for naming folders per digit

    for lang in ['asl', 'csl']:
        lang_folder = os.path.join(video_dir, lang)
        for digit_folder in os.listdir(lang_folder):
            video_folder = os.path.join(lang_folder, digit_folder)
            if not os.path.isdir(video_folder):
                continue

            # Initialize folder counter for each digit
            if digit_folder not in folder_counter:
                folder_counter[digit_folder] = 0

            for video_file in os.listdir(video_folder):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(video_folder, video_file)

                    # Generate a new folder name based on digit and counter
                    new_folder_name = f"{digit_folder}.{folder_counter[digit_folder]}"
                    folder_counter[digit_folder] += 1

                    unique_image_folder = os.path.join(output_image_dir, lang, digit_folder, new_folder_name)
                    os.makedirs(unique_image_folder, exist_ok=True)

                    # Extract frames for the current video
                    extract_frames(video_path, unique_image_folder)

if __name__ == "__main__":
    data_dir = './data'
    process_all_videos(
        video_dir=os.path.join(data_dir, 'sign_videos'),
        output_image_dir=os.path.join(data_dir, 'sign_images')
    )
