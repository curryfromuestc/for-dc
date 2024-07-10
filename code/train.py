import torch
import torch.nn as nn
from model import *
import tqdm
from vedeio_loader import *

model = Conv_fc(T=40, image_size=64)
optmizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#data_loader是一个自己的MP4视频数据集
verdeio_dataloader = VideoDataset(video_dir="你的视频目录路径")
dataloader = DataLoader(verdeio_dataloader, batch_size=4, shuffle=True)

def train(model,data_loader,optmizer,criterion,device):
    model.train()
    running_loss = 0
    for i, data in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optmizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optmizer.step()
        running_loss += loss.item()
    return running_loss/len(data_loader)

for epoch in range(10):
    loss = train(model,dataloader,optmizer,criterion,device)
    print("Epoch: {} Loss: {}".format(epoch, loss))