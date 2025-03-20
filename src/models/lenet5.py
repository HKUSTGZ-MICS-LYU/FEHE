'''LeNet5 in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch import nn

class LeNet5(nn.Module):  # 使用nn.Module而不是Module
    def __init__(self):
        super(LeNet5, self).__init__()  # 修正super调用
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        
        # 添加调试语句，计算正确的展平维度
        self.fc1 = nn.Linear(16*5*5, 120)  # 使用计算出的256
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        
        # 在展平前打印特征图尺寸，用于调试
        # print(f"Before flatten: {y.shape}")
        
        y = y.view(y.shape[0], -1)
        
        # 展平后打印尺寸，确认维度
        # print(f"After flatten: {y.shape}")
        
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y