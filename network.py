import torch
import torch.nn as nn
import torch.optim as optimizer
import argparse
import torch.nn.functional as F



import numpy as np

class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, downsample=None):
        super(Net, self).__init__()
        self.conv1= nn.Conv1d(10, 16, kernel_size=1)
        self.v = nn.Sequential(nn.Linear(16, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(16, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 2), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 2), nn.Softplus())
        self.conv_drop = nn.Dropout2d()
        self.apply(self._weights_init)
        self.downsample=downsample

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x= x.view(-1,10,1)
        residual = x
        if(self.downsample is not None):
            residual = self.downsample(residual)
        x = F.relu(F.max_pool1d(self.conv_drop(self.conv1(x)),kernel_size=1))+residual
        x = x.view(-1, 16)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1 

        return (alpha, beta), v
