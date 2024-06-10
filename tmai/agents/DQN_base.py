import torch
import torch.nn as nn

class DQNBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, observation):
        raise NotImplementedError("This method should be overridden by subclasses")
