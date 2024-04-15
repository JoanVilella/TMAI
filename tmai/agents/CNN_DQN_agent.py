import numpy as np
import torch
import torch.nn as nn
from tmai.agents.agent import Agent
import os
from datetime import datetime

class CNN_DQN(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(CNN_DQN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(32*4*4, 256)
        self.fc2 = nn.Linear(256, output_size)            


    def forward(self, observation):
        x = torch.Tensor(observation).to("cuda" if torch.cuda.is_available() else "cpu")
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x