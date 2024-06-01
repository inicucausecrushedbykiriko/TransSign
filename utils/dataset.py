import os
import cv2
import torch
from torch.utils.data import Dataset

class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for video_file in os.listdir(label_dir):
                    if video_file.endswith('.mp4'):
                        self.data.append(os.path.join(label_dir, video_file))
                        self.labels.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data[idx]
        frames = self.load_video_frames(video_path)
        label = self.labels[idx]

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        return frames, label

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
