import torch
import torch.nn as nn
from tmai.agents.aa_DQN_base import DQNBase

class Linear_DQN(DQNBase):
    def __init__(self, input_size, output_size, mid_size=32, p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, mid_size)
        self.fc2 = nn.Linear(mid_size, output_size)
        self.dropout = nn.Dropout(p)

    def forward(self, observation):
        x = torch.Tensor(observation).to("cuda" if torch.cuda.is_available() else "cpu")
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x