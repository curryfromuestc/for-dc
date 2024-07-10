from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torch
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        """
        video_dir: 存放视频文件的目录
        transform: 应用于视频帧的预处理函数
        """
        self.video_dir = video_dir
        self.transform = transform
        self.videos = os.listdir(video_dir)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.videos[idx])
        # 使用OpenCV加载视频
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        # 假设你的模型期望固定数量的帧，这里简单地选择前N帧
        N = 10  # Replace 10 with the desired number of frames
        frames = frames[:N]  # N是你选择的帧数
        # 将帧列表转换为模型期望的张量形式
        frames_tensor = torch.stack(frames)
        return frames_tensor

# 实例化数据集和数据加载器
#video_dataset = VideoDataset(video_dir="你的视频目录路径")
#data_loader = DataLoader(video_dataset, batch_size=4, shuffle=True)

# 接下来，你可以使用这个data_loader在训练循环中加载数据