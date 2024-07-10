import torch
import torch.nn as nn

class Conv_fc(nn.Module):
    def __init__(self, *args, T, image_size, **kwargs) -> None:
        super().__init__(*args,T,image_size, **kwargs)
        self.T = T
        self.image_size = image_size
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*self.image_size*self.image_size*self.T, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    