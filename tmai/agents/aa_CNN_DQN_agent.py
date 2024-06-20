import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tmai.agents.agent import Agent
import os
from datetime import datetime

class CNN_DQN(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(CNN_DQN, self).__init__()

        # Primera capa convolucional
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Segunda capa convolucional
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Tercera capa convolucional
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # Capa completamente conectada
        self.fc1 = nn.Linear(128 * 16 * 16, 128)  # Ajustar tamaño si la imagen de entrada cambia
        self.fc2 = nn.Linear(128, output_size)     


    def forward(self, observation):
        x = torch.Tensor(observation).to("cuda" if torch.cuda.is_available() else "cpu")
        x = self.pool(F.relu(self.conv1(x)))  # Capa convolucional 1 con ReLU y Max Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Capa convolucional 2 con ReLU y Max Pooling
        x = self.pool(F.relu(self.conv3(x)))  # Capa convolucional 3 con ReLU y Max Pooling
        x = x.view(-1, 128 * 16 * 16)  # Aplanar la salida
        x = F.relu(self.fc1(x))  # Capa completamente conectada con ReLU
        x = self.fc2(x)  # Capa de salida
        return x